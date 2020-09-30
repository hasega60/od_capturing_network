# coding:utf-8
import copy
import datetime
import math
from ast import literal_eval
import networkx as NX
from gurobipy import *
from tqdm import tqdm
from inputimeout import inputimeout, TimeoutOccurred
import pandas as pd


EPS = 1.e-6
timeLimit = 600
# 解の精度
setParam("MIPGap", 0.00)
setParam("TimeLimit", timeLimit)
setParam("MIPFocus", 1)

# パラメータ設定
basedir = 'C:/Projects/Gurobi/createRoute/joso/'
resultdir = 'result/'
debugdir = 'debug/'
nodefile = "nodes.csv"
edgefile = "edges.csv"
flowfile = "flows.csv"
dMatrixfile = "distanceMatrixbyNetwork.csv"
hubListfille = "hub.csv"

#カヴァー済ノードリスト ここに含まれていたらフロー量を0にする
coveredNodeList=[]
#abiko
#coveredNodeList = [3,4,5,25,28,29,30,31,32,33,36,42,43,44,45,46,48,52,53,55,56,57,58,59,60,63,64,65,66,67,68,69,70,71,74,75,76,77,78,79,81,82,83,85,86,87,88,90,102,103,105,106,108,114,115,117,118,119,120,122,144,158,160,161,162,163,165,166,168]


# モデルの解出力ファイル名
mstFile_loop = "sol_model_loop.sol"
mstFile_path = "sol_model_path.sol"

bWeight = True  # pmedianでノードの重みを考慮するか
bSolveKShortestPath = False  # k-shortestPathを構築するか
bTSPRoute = False  # TSProuteを構築するか
bMSTRoute = False  # MSTrouteを構築するか
bReversedRoute = False  # ハブ間路線を構築するときに逆方向路線を作成するか
numHubs = 7  # ハブの数
numMainHubs = 0  # ハブのうち，メインハブの数 残りはサブハブとする 0ならばp-medianのみ

# 路線作成時のパラメータ　循環路線
maxloopLength = 20000  # ループ最大延長
minLength = 5000  # 最小路線長
span = 2500  # 路線候補を作る間隔
# maxNodeSize_createRoute_loop_Flow = 60  # 循環線を作成する場合のノード上限数 重みが大きく，距離が小さいノードから順に追加する

maxTotalLength = 100000  # 最大路線長
minTotalLength = 5000  # 最小路線長
totalLengthSpan = 5000  # 路線候補を作る間隔

# 拠点間路線
minPathRatio = 1.1  # 最短経路から距離，フロー量制約を増加させる割合の最小量
maxPathRatio = 3.0  # 最短経路から距離，フロー量制約を増加させる割合の最大量
pathRatioSpan = 0.1  # 割合増加間隔
bMinLength = False  # 拠点間ルートは最短経路で作成するか falseなら最大フロー

now = datetime.datetime.now()
routeFileName = "routeList" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"
routeCombiFileName = "routeCombination" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"

df = pd.DataFrame()

pd.read_csv('***.csv')
N = None
E = None
D = None
G = None
E_id = None


def createDistanceMatrix(G, N, E):
    # このネットワーク（edges）を用いた距離行列を作成する　一回だけ行って，後はcsvを読み込めばいい
    dists = pd.DataFrame(index=[],
                         columns=["originID",
                                  "destID",
                                  "distance"])

    for i in N:
        for j in N:
            source = i
            target = j

            if source != target:
                series = pd.Series([source, target, NX.dijkstra_path_length(G, source, target)], index=dists.columns)
                dists = dists.append(series, ignore_index=True)
            else:
                series = pd.Series([source, target, 0], index=dists.columns)
                dists = dists.append(series, ignore_index=True)

    dists.to_csv(basedir + "distanceMatrixbyNetwork.csv", index=False)
    print("distance matrix exported!")


def load_data(nodeData, edgeData, distanceMatrixData, flowData=None):
    # 基準データを読み込み
    nodes, edges, flows, dists = {}, {}, {}, {}
    N, E, D, F, E_id = {}, {}, {}, {}, {}
    # nodeデータのxyはメートルであること
    nodes = pd.read_csv(nodeData,
                        encoding='Shift-JIS')
    edges = pd.read_csv(edgeData,
                        encoding='Shift-JIS')
    if flowData is not None:
        flows = pd.read_csv(flowData,
                            encoding='Shift-JIS')
        for i, v in flows.iterrows():
            #カヴァー済ノードリストに含まれていた場合はフロー量を0にする
            if coveredNodeList is not None and v["originID"] in coveredNodeList and v["destID"] in coveredNodeList:
                F[(v["originID"], v["destID"])] = 0
            else:
                F[(v["originID"], v["destID"])] = v["flow"]

        print("flow data loaded!")

    totalF = 0
    for i, v in nodes.iterrows():
        # N[v["id"]] = (v["x"], v["y"], v["weight"], v["pop"], v["storeArea"], v["hospitalNum"])
        N[v["id"]] = (v["x"], v["y"], v["weight"])

    print("node data loaded!")
    for i, v in edges.iterrows():
        E[(v["originID"], v["destID"])] = v["distance"]
        E_id[(v["originID"], v["destID"])] = v["edgeID"]

    G = NX.Graph()
    G.add_nodes_from(N)
    # エッジデータ作成
    for key in E:
        value = E[key]
        G.add_edge(key[0], key[1], weight=value)

    print("edge data loaded!")

    import os
    if os.path.exists(distanceMatrixData):
        dists = pd.read_csv(distanceMatrixData,
                            encoding='Shift-JIS')
    else:
        createDistanceMatrix(G, N, E)
        dists = pd.read_csv(distanceMatrixData, encoding='Shift-JIS')

    """
    for i, v in dists.iterrows():
    D[(v["originID"], v["destID"])] = v[
        "distance"]"""

    for o, d, dist in zip(dists.originID, dists.destID, dists.distance):
        D[(o, d)] = dist

    print("distance matrix loaded!")

    return N, E, D, F, G, E_id


def input_with_timeout(prompt='', timeout=120.0):
    import msvcrt
    import time
    begin = time.monotonic()
    end = begin + timeout
    for c in prompt:
        msvcrt.putwch(c)
    line = ''
    is_timeout = True
    while time.monotonic() < end:
        if msvcrt.kbhit():
            c = msvcrt.getwch()
            msvcrt.putwch(c)
            if c == '\r' or c == '\n':
                is_timeout = False
                break
            if c == '\003':
                raise KeyboardInterrupt
            if c == '\b':
                line = line[:-1]
            else:
                line = line + c
        time.sleep(0.05)
    msvcrt.putwch('\r')
    msvcrt.putwch('\n')
    if is_timeout:
        raise TimeoutOccurred
    return line


def pmedian(N, D, p, bWeight):
    model = Model("p-median")
    # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y = {}, {}
    for j in N:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in N:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N:
        model.addConstr(quicksum(x[i, j] for j in N) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in N:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
    model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個

    if bWeight:
        model.setObjective(quicksum(D[i, j] * x[i, j] * N[i][2] for i in N for j in N), GRB.MINIMIZE)
    else:
        model.setObjective(quicksum(D[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

    model.update()
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("pmedian:実行不可能で終了_" + str(p))
        # outIIS(model_rmf)
        return None

    model.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    select = [j for j in x if x[j].X > EPS]
    print("obj:"+str(model.objVal))
    print("hubs:" + str(hubs))
    return hubs, select


# メモリ不足となる
def p_path_median(N, G, D, p, mainNode, bWeight):
    model = Model("p-path-median")
    x, y, z, s = {}, {}, {}, {}

    # y_jkは事前に作成
    print("1:create S " + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for k in N:
        s[k] = NX.dijkstra_path(G, k, mainNode)

    NS = copy.deepcopy(N)  # 出発地を除いたノード集合
    NS.pop(mainNode)
    # ノードの重みが0のところも除く
    for i in N:
        if N[i][2] == 0:
            NS.pop(i)

    print("2:create Y " + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for j in NS:
        for k in NS:
            if j in s[k]:
                y[j, k] = 1
            else:
                y[j, k] = 0

    print("3:create const " + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for k in NS:
        z[k] = model.addVar(vtype="B", name="z(%s)" % k)

    for j in NS:
        for i in NS:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model.update()
    for k in NS:
        for j in NS:
            model.addConstr(y[j, k] <= z[k])
            for i in NS:
                model.addConstr(x[i, j] <= y[j, k])

    for i in NS:
        model.addConstr(quicksum(x[i, j] for j in NS) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる

    model.addConstr(quicksum(z[k] for k in NS) == p, "Facilities")  # 終点の数はp個
    if bWeight:
        model.setObjective(quicksum(D[i, j] * x[i, j] * NS[i][2] for i in NS for j in NS), GRB.MINIMIZE)
    else:
        model.setObjective(quicksum(D[i, j] * x[i, j] for i in NS for j in NS for k in NS), GRB.MINIMIZE)

    model.update()
    print("4:optimize " + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("pmedian:実行不可能で終了_" + str(p))
        # outIIS(model_rmf)
        return None

    model.__data = y, z
    hubs = [j for j in z if z[j].X > EPS]
    return hubs


def pcenter(N, D, p):
    model = Model("p-center")
    # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y = {}, {}
    z = model.addVar(vtype="C")
    for j in N:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in N:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N:
        model.addConstr(quicksum(x[i, j] for j in N) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in N:
            model.addConstr(x[i, j] <= y[j])
            model.addConstr(D[i, j] * x[i, j] <= z)
    model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個
    model.setObjective(z, GRB.MINIMIZE)

    model.update()
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("p-center:実行不可能で終了_" + str(p))
        # outIIS(model_rmf)
        return None

    model.__data = x, y

    hubs = [j for j in y if y[j].X > EPS]
    return hubs


def pcenter_covering(N, D, p):
    model = Model("p-center-covering")
    # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y, z = {}, {}, {}
    for i in N:
        y[i] = model.addVar(vtype="B", name="y(%s)" % i)
        z[i] = model.addVar(vtype="B", name="z(%s)" % i)

    for i in N:
        for j in N:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N:
        model.addConstr(quicksum(x[i, j] for j in N) + z[i] == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in N:
            model.addConstr(x[i, j] <= y[j])
    model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個
    model.setObjective(quicksum(z[i] for i in N), GRB.MINIMIZE)

    model.update()
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("p-center:実行不可能で終了_" + str(p))
        # outIIS(model_rmf)
        return None

    model.__data = x, y, z

    hubs = [j for j in y if y[j].X > EPS]
    return hubs


# 一晩やっても解けなかった
def pcenter_binarySearch(N, D, p):
    model = Model("p-center-covering-b")
    # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y, z = {}, {}, {}
    for i in N:
        y[i] = model.addVar(vtype="B", name="y(%s)" % i)
        z[i] = model.addVar(vtype="B", name="z(%s)" % i)

    for i in N:
        for j in N:
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N:
        model.addConstr(quicksum(x[i, j] for j in N) + z[i] == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in N:
            model.addConstr(x[i, j] <= y[j])
    model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個
    model.setObjective(quicksum(z[i] for i in N), GRB.MINIMIZE)
    model.update()
    model.__data = x, y, z
    model.Params.Cntoff = .1
    facilities, edges = [], []
    LB = 0
    UB = max(D[i, j] for (i, j) in D)
    while UB - LB > 1.e-4:
        theta = (UB + LB) / 2.
        # print "\n\ncurrent theta:", theta
        for j in N:
            for i in N:
                if D[i, j] > theta:
                    x[i, j].UB = 0
                else:
                    x[i, j].UB = 1.0
        model.update()
        # model.Params.OutputFlag = 0 # silent mode
        model.Params.Cutoff = .1
        model.optimize()

        if model.status == GRB.Status.OPTIMAL:
            # infeasibility = sum([z[i].X for i in I])
            # print "infeasibility=",infeasibility
            UB = theta
            facilities = [j for j in y if y[j].X > .5]
            edges = [(i, j) for (i, j) in x if x[i, j].X > .5]
            # print "updated solution:"
            # print "facilities",facilities
            # print "edges",edges
        else:  # infeasibility > 0:
            LB = theta

    return facilities, edges


def pmedian_existing(N, D, yExt, q, bWeight):
    model2 = Model("p-median-ext")
    I = N
    J = N
    r = 0  # 廃止施設は考えない
    x, y, ye, z = {}, {}, {}, {}

    for j in J:
        y[j] = model2.addVar(vtype="B", name="y(%s)" % j)
        ye[j] = model2.addVar(vtype="B", name="ye(%s)" % j)
        z[j] = model2.addVar(vtype="B", name="z(%s)" % j)
        for i in I:
            x[i, j] = model2.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model2.update()

    for i in I:
        model2.addConstr(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in J:
            model2.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    model2.addConstr(quicksum(y[j] for j in J) == -r + q, "Facilities_New")
    model2.addConstr(quicksum(z[j] for j in J) == r + q, "Changed")

    for j in J:
        for e in yExt:
            if j == e:
                model2.addConstr(-1 * z[j] <= y[j] - 1)
                model2.addConstr(1 - ye[j] <= z[j])
            else:
                model2.addConstr(-1 * z[j] <= y[j])
                model2.addConstr(ye[j] <= z[j])

    # 距離行列の更新
    cNew = updateDistance(I, J, D, yExt)
    if bWeight:
        # 重みを考慮するか
        model2.setObjective(quicksum(cNew[i, j] * x[i, j] * N[i][2] for i in I for j in J), GRB.MINIMIZE)
    else:
        model2.setObjective(quicksum(cNew[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

    model2.update()
    model2.optimize()
    if model2.Status == GRB.INFEASIBLE:
        print("pmedian_ext:実行不可能で終了_" + "_" + str(q))
        # outIIS(model_rmf)
        return None

    model2.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]

    # pcenter用初期解作成
    model2.write("pmedian_ext_sol_out.mst")

    return hubs


def pcenter_existing(N, D, yExt, q, bWeight):
    # いったんp-median-existを解いてから，それを初期解として使う
    pmedian_existing(N, D, yExt, q, bWeight)

    r = 0
    model3 = Model("p-center-ext")
    a = model3.addVar(vtype="C", name="a")
    I = N
    J = N
    x, y, ye, z = {}, {}, {}, {}
    for j in J:
        y[j] = model3.addVar(vtype="B",
                             name="y(%s)" % j)
        ye[j] = model3.addVar(vtype="B",
                              name="ye(%s)" % j)
        z[j] = model3.addVar(vtype="B",
                             name="z(%s)" % j)
        for i in I:
            x[i, j] = model3.addVar(vtype="B",
                                    name="x(%s,%s)" % (
                                        i, j))
    model3.update()

    # 距離行列の更新
    cNew = updateDistance(I, J, D, yExt)

    for i in I:
        model3.addConstr(
            quicksum(x[i, j] for j in J) == 1,
            "Assign(%s)" % i)
        model3.addConstr(quicksum(
            cNew[i, j] * x[i, j] for j in J) <= a,
                         "Max_x(%s)" % (i))

        for j in J:
            model3.addConstr(x[i, j] <= y[j],
                             "Strong(%s,%s)" % (
                                 i, j))

    model3.addConstr(
        quicksum(y[j] for j in J) == -r + q,
        "Facilities")

    for j in J:
        for e in yExt:
            if j == e:
                model3.addConstr(
                    -1 * z[j] <= y[j] - 1)
                model3.addConstr(
                    1 - ye[j] <= z[j])
            else:
                model3.addConstr(
                    -1 * z[j] <= y[j])
                model3.addConstr(ye[j] <= z[j])

    model3.setObjective(a, GRB.MINIMIZE)
    model3.update()

    # 初期解読み込み
    model3.read("pmedian_ext_sol_out.mst")

    model3.optimize()
    if model3.Status == GRB.INFEASIBLE:
        print("pcenter_ext:実行不可能で終了_" + "_" + str(q))
        return None

    model3.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    return hubs


def k_shortest_paths(G, F, source, target, limitLength=10000, limitCalcTime=12000, limitNum=100, weight='weight'):
    import KShortestPaths

    def get_path_length(G, path, weight='weight'):
        length = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]

                length += G.edge[u][v].get(weight, 1)

        return length

    def kpMain(lengths, paths, calcTimes):
        import time
        startTime = time.time()
        midTime = startTime
        KP = KShortestPaths.KShortestPaths(G, weight)
        flowPath = 0
        path = KShortestPaths.findFirstShortestPath(KP, source, target)
        for i in path.nodeList:
            for j in path.nodeList:
                flowPath += F[(i, j)]
        length = path.cost
        pbar = tqdm(total=limitNum)
        len_pbar = length
        count = 0
        while True:
            currentTime = time.time()
            # timeout
            if currentTime - startTime >= limitCalcTime:
                pbar.close()
                print("timeout!")
                return lengths, paths, True, currentTime - startTime

            if length < limitLength and count < limitNum:
                path = KShortestPaths.getNextShortestPath(KP)
                kp_fpath = 0

                for i in path.nodeList:
                    for j in path.nodeList:
                        kp_fpath += F[(i, j)]
                if kp_fpath > flowPath:
                    count += 1
                    pbar.set_description(
                        "Processing k_shortest_path " + str(source) + "to" + str(target) + "_%s" % length)
                    pbar.update(1)
                    flowPath = kp_fpath
                    len_pbar = path.cost

                    length = path.cost
                    paths.append(path)
                    lengths.append(length)
                    calcTimes.append(currentTime - midTime)
                    midTime = currentTime
            else:
                pbar.close()
                return lengths, paths, False, calcTimes

    lengths, paths, calcTimes = [], [], []
    try:
        lengths, paths, isTimeout, calcTimes = kpMain(lengths, paths, calcTimes)
        return lengths, paths, isTimeout, calcTimes
    except:
        import traceback
        print(traceback.format_exc())
        print("timeout_KP")
        return lengths, paths, True, calcTimes


def mst_weightedNodes(nodeList, D):
    edges = []
    for i in nodeList:
        for j in nodeList:
            if i != j:
                dist = D[(i, j)]
                edges.append((i, j, dist))

    mst_G = NX.Graph()
    mst_G.add_weighted_edges_from(edges)
    mst_edges = list(NX.minimum_spanning_edges(mst_G))
    length = 0
    for e in mst_edges:
        length += e[2]["weight"]

    return length, mst_edges


def tsp_weightedNodes(nodeList, D):
    def tsp():
        V = nodeList
        """tsp -- model for solving the traveling salesman problem with callbacks
           - start with assignment model
           - add cuts until there are no sub-cycles
        Parameters:
            - V: set/list of nodes in the graph
            - c[i,j]: cost for traversing edge (i,j)
        Returns the optimum objective value and the list of edges used.
        """

        EPS = 1.e-6

        def tsp_callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return

            edges = []
            for (i, j) in x:
                if model.cbGetSolution(x[i, j]) > EPS:
                    edges.append((i, j))

            tsp_G = NX.Graph()
            tsp_G.add_edges_from(edges)
            if NX.number_connected_components(tsp_G) == 1:
                return

            Components = NX.connected_components(tsp_G)
            for S in Components:
                model.cbLazy(quicksum(x[i, j] for i in S for j in S if j > i) <= len(S) - 1)
                # print "cut: len(%s) <= %s" % (S,len(S)-1)
            return

        model = Model("tsp")
        # model.Params.OutputFlag = 0 # silent/verbose mode
        x = {}
        for i in V:
            for j in V:
                if j > i:
                    x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
        model.update()

        for i in V:
            model.addConstr(quicksum(x[j, i] for j in V if j < i) + \
                            quicksum(x[i, j] for j in V if j > i) == 2, "Degree(%s)" % i)

        model.setObjective(quicksum(D[i, j] * x[i, j] for i in V for j in V if j > i), GRB.MINIMIZE)

        model.update()
        model.__data = x
        return model, tsp_callback

    model, tsp_callback = tsp()
    model.params.DualReductions = 0
    model.params.LazyConstraints = 1
    model.optimize(tsp_callback)
    x = model.__data
    EPS = 1.e-6
    edges = []
    for (i, j) in x:
        if x[i, j].X > EPS:
            edges.append((i, j))
    return model.ObjVal, edges, model.Runtime, model.MIPGap


def updateDistance(I, J, c, yExt):
    ce = {}
    for i in I:
        for j in J:
            dist = c[i, j]
            for k in yExt:
                distNew = c[i, k]
                if distNew < dist:
                    dist = distNew

                ce[i, j] = dist

    return ce


def outIIS(model):
    model.computeIIS()
    model.write("outputIISLog.ilp")


def createTerminalCombination(hubs, D, minLength):
    list = []
    for hub1 in hubs:
        for hub2 in hubs:
            if list not in [(hub1, hub2)]:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 == hub2 or D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list


def createTerminalCombinationOnlyloop(hubs, list=None):
    if list is None:
        list = []

    for hub1 in hubs:
        for hub2 in hubs:
            # 同一ハブ
            if hub1 == hub2:
                list.append([hub1, hub2])
    return list


def createTerminalCombinationNoloop(hubs, D, minLength, list=None):
    if list is None:
        list = []

    for hub1 in hubs:
        for hub2 in hubs:
            if [hub1, hub2] not in list and [hub2, hub1] not in list:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 != hub2 and D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list


def createRouteList(N, E, D, F, G, listTerminal):
    routes_condition, routes_node, routes_edge, routes_edge_id, routes_length, routes_flow, routes_CalcTime, routes_MIPGap = {}, {}, {}, {}, {}, {}, {}, {}
    count = 0
    print("createRoute:" + str(listTerminal))

    numNodeMin = 5
    numNodeSpan = 5
    numNodeMax = 50
    conditionList, lengthList, selectNodeList, selectEdgeList, selectEdgeIDlist, calcTimeList, mipGapList = [], [], [], [], [], [], []

    # 降順でソート
    if bTSPRoute or bMSTRoute:
        print("-----重要度の高いものを巡回する路線(TSP)-----")
        print("-----重要度の高いものを結ぶ最小木(MST)-----")
        sortF = sorted(F.items(), key=lambda x: x[1], reverse=True)
        for num in range(numNodeMin, numNodeMax + numNodeSpan, numNodeSpan):
            conditionList.append("TSP_" + str(num))
            nodeList = []
            # フローの多いノードたちからピックアップ
            for f in sortF:
                s = f[0][0]
                t = f[0][1]
                if s not in nodeList:
                    nodeList.append(s)
                if t not in nodeList:
                    nodeList.append(t)
                if len(nodeList) >= num:
                    break

            if bTSPRoute:
                length, tspEdge, calcTime, mipGap = tsp_weightedNodes(nodeList, D)
                selectN, selectE, selectE_id = [], [], []

                for (s, t) in tspEdge:
                    # 選ばれたノードの組み合わせをエッジ化
                    nodes = NX.dijkstra_path(G, s, t)
                    i = -1
                    for n in nodes:
                        if n not in selectN:
                            selectN.append(n)
                        if i != -1:
                            selectE.append((i, n))
                            selectE_id.append(E_id[(i, n)])
                        i = n

                lengthList.append(length)
                selectNodeList.append(selectN)
                selectEdgeList.append(selectE)
                selectEdgeIDlist.append(selectE_id)
                calcTimeList.append(calcTime)
                mipGapList.append(mipGap)

            if bMSTRoute:
                length, mstEdge = mst_weightedNodes(nodeList, D)
                # TODO MSTのときに次数が1のノード（端部）をカウント
                selectN, selectE, selectE_id = [], [], []
                for e in mstEdge:
                    # 選ばれたノードの組み合わせをエッジ化
                    nodes = NX.dijkstra_path(G, e[0], e[1])
                    i = -1
                    for n in nodes:
                        if n not in selectN:
                            selectN.append(n)
                        if i != -1:
                            selectE.append((i, n))
                            selectE_id.append(E_id[(i, n)])
                        i = n

                conditionList.append("MST_" + str(num))
                lengthList.append(length)
                selectNodeList.append(selectN)
                selectEdgeList.append(selectE)
                selectEdgeIDlist.append(selectE_id)
                calcTimeList.append(0)
                mipGapList.append(-1)

        for i in range(len(conditionList)):
            f = 0.000
            for u in selectNodeList[i]:
                for t in selectNodeList[i]:
                    f += F[(u, t)]

            routes_condition[count] = conditionList[i]
            routes_node[count] = selectNodeList[i]
            routes_edge[count] = selectEdgeList[i]
            routes_edge_id[count] = selectEdgeIDlist[i]
            routes_length[count] = lengthList[i]
            routes_flow[count] = f
            routes_CalcTime[count] = calcTimeList[i]
            routes_MIPGap[count] = mipGapList[i]
            count += 1
        # TODO 都度出力
        outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length, routes_flow,
                        routes_CalcTime, routes_MIPGap)

    for hub in tqdm(listTerminal):
        #TODO debug
        if hub[0] == 25:
            continue

        shortestLength = 0
        model = None
        import os
        # mstファイルを削除
        if os.path.exists(mstFile_loop):
            os.remove(mstFile_loop)

        if os.path.exists(mstFile_path):
            os.remove(mstFile_path)

        if hub[0] == hub[1]:
            # ---------------------------------------ハブ循環路線---------------------------------------
            print("-----循環路線(modelB-2)-----")
            flow_b = 0
            bFinish = False
            for length in range(minLength, maxloopLength + span, span):
                print("---------------------------↓" + "solve:" + str(hub[0]) + ", " + str(hub[1]) + "," + str(
                    length) + "↓-------------------------")

                # ハブから一定範囲内(路線長制限の半分)のエッジを対象にする
                NS, ES = select_insideBoundary(N, E, hub[0], length / 2)
                print("size_node:" + str(len(NS)) + " edge:" + str(len(ES)))
                selectF, selectE, selectE_id, routeLength, model = createRoute_loop_Flow(NS, ES, D, F, hub[0], length)

                if selectE is not None and len(selectE) > 1:
                    if flow_b > model.objVal:
                        bFinish = True

                    selectN = edgeConvertnodes(selectE)
                    f = 0.000
                    for i in selectN:
                        for j in selectN:
                            f += F[(i, j)]

                    routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str(length) + "_" + str(
                        "loop")
                    routes_node[count] = selectN
                    routes_edge[count] = selectE
                    routes_edge_id[count] = selectE_id
                    routes_length[count] = routeLength
                    # routes_flow[count] = model.objVal
                    routes_flow[count] = f
                    routes_CalcTime[count] = model.Runtime
                    routes_MIPGap[count] = model.MIPGap
                    flow_b = model.objVal
                else:
                    bFinish = True
                    selectN = []
                    routes_node[count] = []
                    routes_edge[count] = []
                    routes_edge_id[count] = []
                    routes_length[count] = -1
                    routes_flow[count] = -1
                    routes_CalcTime[count] = -1
                    routes_MIPGap[count] = -1
                count += 1
                # TODO 都度出力
                outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length, routes_flow,
                                routes_CalcTime, routes_MIPGap)

                if bFinish:
                    # 実行結果が前回の制約条件の厳しい結果より悪い,実行不可能な場合はそれ以降の計算をしない
                    break

        else:
            # ---------------------------------------ハブ間路線--------------------------------------

            print("-----最短経路-----")
            shortestLength = NX.dijkstra_path_length(G, hub[0], hub[1])
            flowShortestPath = 0
            sPath = NX.dijkstra_path(G, hub[0], hub[1])
            for i in sPath:
                for j in sPath:
                    flowShortestPath += F[(i, j)]

            print("shortestLength_(" + str(hub[0]) + ", " + str(hub[1]) + "):" + str(shortestLength))
            routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str("spath")
            # TODO 都度出力
            s = -1
            sEdge, sEdge_id = [], []
            for p in sPath:
                if s != -1:
                    sEdge.append((s, p))
                    sEdge_id.append(E_id[(s, p)])
                s = p

            routes_node[count] = sPath
            routes_edge[count] = sEdge
            routes_edge_id[count] = sEdge_id
            routes_length[count] = shortestLength
            routes_flow[count] = flowShortestPath
            routes_CalcTime[count] = 0
            routes_MIPGap[count] = 0
            count += 1

            if (bSolveKShortestPath):
                print("-----経路長制約k-shortestpath-----")
                calcTimeLimit = int(timeLimit * ((maxPathRatio - minPathRatio) / pathRatioSpan))  # 最大計算時間
                countKpath = 0
                maxLength = math.ceil(maxPathRatio * shortestLength)
                # limitNumが回数10回を超えたあたりから計算時間，メモリ消費が倍々で増える
                lengths, paths, bTimeout, calcTimes = k_shortest_paths(G.copy(), F, hub[0], hub[1],
                                                                       limitLength=maxLength,
                                                                       limitCalcTime=calcTimeLimit, limitNum=10)
                for wpath in paths:
                    path = wpath.nodeList
                    sEdge, sEdge_id = [], []
                    flowPath = 0
                    s = -1
                    for i in path:
                        if s != -1:
                            sEdge.append((s, i))
                            sEdge_id.append(E_id[(s, i)])
                        s = i
                        for j in path:
                            flowPath += F[(i, j)]

                    routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str(
                        "K-shortest_" + str(countKpath))
                    routes_node[count] = path
                    routes_edge[count] = sEdge
                    routes_edge_id[count] = sEdge_id
                    routes_length[count] = lengths[countKpath]
                    routes_flow[count] = flowPath
                    routes_CalcTime[count] = calcTimes[countKpath]
                    routes_MIPGap[count] = -1
                    count += 1
                    countKpath += 1

                # 出力
                outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length,
                                routes_flow, routes_CalcTime, routes_MIPGap)

            print("-----路線調整役フロー最大化路線(modelB-1)-----")
            # 拠点から逆方向にいかないように，ノードとエッジを限定する
            N_s, E_s = select_insideHubsBoundary(N, E, hub)

            length_b = 0
            flow_b = 0
            bFinish = False
            for ratio in drange(minPathRatio, maxPathRatio + pathRatioSpan * 2, pathRatioSpan):
                if bMinLength:
                    # 路線長最小化 フロー制約
                    flow = ratio * flowShortestPath
                    model = None
                    print("solve:" + str(hub[0]) + ", " + str(hub[1]) + ",_flow:" + str(flow))
                    print("nodeSize:" + str(len(N_s)) + " edgeSize:" + str(len(E_s)))
                    # 最大路線長を超えないパターンを作成
                    selectE, selectE_id, routeFlow, model = createRoute_minLength(N_s, E_s, hub[0], hub[1], flow, F)
                    routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + '{:.1f}'.format(
                        ratio) + "_" + str(
                        "F")
                    if selectE is not None:
                        if length_b > model.objVal:
                            bFinish = True

                        selectN = edgeConvertnodes(selectE)
                        routes_node[count] = selectN
                        routes_edge[count] = selectE
                        routes_edge_id[count] = selectE_id
                        routes_length[count] = model.objVal
                        routes_flow[count] = routeFlow
                        routes_CalcTime[count] = model.Runtime
                        routes_MIPGap[count] = model.MIPGap
                        length_b = model.objVal

                    else:
                        # 実行不可能解
                        bFinish = True
                        selectN_r = []
                        routes_node[count] = []
                        routes_edge[count] = []
                        routes_edge_id[count] = []
                        routes_length[count] = -1
                        routes_flow[count] = -1
                        routes_CalcTime[count] = -1
                        routes_MIPGap[count] = -1

                    count += 1
                    # 出力
                    outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length,
                                    routes_flow, routes_CalcTime, routes_MIPGap)

                    if bFinish:
                        # 実行結果が前回の制約条件の厳しい結果より悪い,実行不可能な場合はそれ以降の計算をしない
                        break

                    if bReversedRoute and selectE is not None:
                        # すでに上記路線が選ばれた場合の路線を作成する
                        N_selected = updateNodeWeight(N, selectN)
                        F_selected = updateFlow(F, selectN)
                        selectE_r, selectE_id_r, routeFlow_r, model_r = createRoute_minLength(N_selected, E, hub[0],
                                                                                              hub[1], flow,
                                                                                              F_selected)

                        routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + '{:.1f}'.format(
                            ratio) + "_" + str(
                            "R")
                        if selectE_r is not None:

                            selectN_r = edgeConvertnodes(selectE_r)
                            routes_node[count] = selectN_r
                            routes_edge[count] = selectE_r
                            routes_edge_id[count] = selectE_id_r
                            routes_length[count] = model_r.objVal
                            routes_flow[count] = routeFlow_r
                            routes_CalcTime[count] = model_r.Runtime
                            routes_MIPGap[count] = model_r.MIPGap
                        else:
                            selectN_r = []
                            routes_node[count] = []
                            routes_edge[count] = []
                            routes_edge_id[count] = []
                            routes_length[count] = -1
                            routes_flow[count] = -1
                            routes_CalcTime[count] = -1
                            routes_MIPGap[count] = -1

                        count += 1
                        # TODO 都度出力
                        outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length,
                                        routes_flow, routes_CalcTime, routes_MIPGap)
                else:
                    # フロー最大化　路線長制約
                    length = math.ceil(ratio * shortestLength)
                    model = None
                    print("solve:" + str(hub[0]) + ", " + str(hub[1]) + "_length:" + '{:.1f}'.format(ratio))
                    print("nodeSize:" + str(len(N_s)) + " edgeSize:" + str(len(E_s)))
                    # 最大路線長を超えないパターンを作成
                    if shortestLength < length:
                        # selectE, routeLength, model = createRoute_maxWeight(N_s, E_s, hub[0], hub[1], length)
                        selectN, selectE, selectE_id, routeLength, model = createRoute_maxFlow(N_s, E_s, hub[0], hub[1],
                                                                                               length, F)
                        routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + '{:.1f}'.format(
                            ratio) + "_" + str(
                            "F")
                        if selectE is not None and len(selectE) >= 1:
                            if flow_b > model.objVal:
                                bFinish = True

                            selectN = edgeConvertnodes(selectE)
                            f = 0.000
                            for i in selectN:
                                for j in selectN:
                                    f += F[(i, j)]

                            routes_node[count] = selectN
                            routes_edge[count] = selectE
                            routes_edge_id[count] = selectE_id
                            routes_length[count] = routeLength
                            routes_flow[count] = f
                            routes_CalcTime[count] = model.Runtime
                            routes_MIPGap[count] = model.MIPGap
                            flow_b = model.objVal

                        else:
                            # 実行不可能解
                            routes_node[count] = []
                            routes_edge[count] = []
                            routes_edge_id[count] = []
                            routes_length[count] = -1
                            routes_flow[count] = -1
                            routes_CalcTime[count] = -1
                            routes_MIPGap[count] = -1

                        count += 1
                        outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length,
                                        routes_flow, routes_CalcTime, routes_MIPGap)
                        if bFinish:
                            # 実行結果が前回の制約条件の厳しい結果より悪い,実行不可能な場合はそれ以降の計算をしない
                            break

                        if bReversedRoute and selectE is not None:
                            # すでに上記路線が選ばれた場合の路線を作成する
                            # N_selected = updateNodeWeight(N, selectN)
                            F_selected = updateFlow(F, selectN)
                            selectN_r, selectE_r, selectE_id_r, routeLength_r, model_r = createRoute_maxFlow(N_s, E_s,
                                                                                                             hub[0],
                                                                                                             hub[1],
                                                                                                             length,
                                                                                                             F_selected)
                            routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + '{:.1f}'.format(
                                ratio) + "_" + str(
                                "R")
                            if selectE_r is not None and len(selectE) >= 1:
                                selectN_r = edgeConvertnodes(selectE_r)
                                f = 0.000
                                for i in selectN_r:
                                    for j in selectN_r:
                                        f += F[(i, j)]

                                routes_node[count] = selectN_r
                                routes_edge[count] = selectE_r
                                routes_edge_id[count] = selectE_id_r
                                routes_length[count] = routeLength_r
                                # routes_flow[count] = model_r.objVal
                                routes_flow[count] = f
                                routes_CalcTime[count] = model_r.Runtime
                                routes_MIPGap[count] = model_r.MIPGap
                            else:
                                selectN_r = []
                                routes_node[count] = []
                                routes_edge[count] = []
                                routes_edge_id[count] = []
                                routes_length[count] = -1
                                routes_flow[count] = -1
                                routes_CalcTime[count] = -1
                                routes_MIPGap[count] = -1

                            count += 1
                            # TODO 都度出力
                            outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length,
                                            routes_flow, routes_CalcTime, routes_MIPGap)

    return routes_condition, routes_node, routes_edge, routes_length, routes_flow


def createRoute_minLength(nodes, edges, s, t, minFlow, F):
    bInit = False
    model_rmd = Model("route_minDist")

    x, y, u = {}, {}, {}
    for (i, j) in edges:
        x[i, j] = model_rmd.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    for i in nodes:
        y[i] = model_rmd.addVar(vtype="B", name="y(%s)" % i)
        u[i] = model_rmd.addVar(vtype="C", name="u(%s)" % i)
        # for j in nodes:
        #   f[i, j] = model_rmf.addVar(vtype="C", name="f(%s,%s)" % (i, j))

    model_rmd.update()

    # 順番制約
    for v in nodes:
        for e in edges:
            if v == e[0]:
                if v == s:
                    model_rmd.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == 1)
                elif v == t:
                    model_rmd.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == -1)
                else:
                    model_rmd.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == 0)

    # xとyの関係
    for i in nodes:
        model_rmd.addConstr(quicksum(x[i, j] for j in nodes if (i, j) in edges) >= y[i])
        model_rmd.addConstr(quicksum(x[j, i] for j in nodes if (i, j) in edges) >= y[i])

    for i in nodes:
        model_rmd.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for (i, j) in edges:
        if i != s and j != s:
            model_rmd.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    # フロー量制約 最後に追加
    model_rmd.addConstr(quicksum(y[i] * y[j] * F[(i, j)] for i in nodes for j in nodes if i != j) >= minFlow, "Flow")
    # model_rmf.setObjective(quicksum(x[i, j] * N[i][2] for (i,j) in E), GRB.MAXIMIZE)
    model_rmd.setObjective(quicksum(edges[i, j] * x[i, j] for (i, j) in edges), GRB.MINIMIZE)

    import os
    if os.path.exists(mstFile_path):
        model_rmd.read(mstFile_path)  # 前回の解を許容界として与える

    model_rmd.update()
    model_rmd.optimize()
    # model_rmd.write(mstFile_path)  # 解を出力
    if model_rmd.Status == GRB.INFEASIBLE or model_rmd.SolCount == 0:
        return None, None, None, None

    model_rmd.__data = x, y
    selectE = [j for j in x if x[j].X > EPS]
    selectN = [j for j in y if y[j].X > EPS]

    selectE_id = []
    for (i, j) in selectE:
        selectE_id.append(E_id[(i, j)])

    flow = 0.0
    for i in selectN:
        for j in selectN:
            flow += F[(i, j)]

    return selectE, selectE_id, flow, model_rmd


def createRoute_maxFlow(nodes, edges, s, t, dist, F):
    bInit = False
    model_rmf = Model("route_maxFlow")

    x, y, u = {}, {}, {}
    for (i, j) in edges:
        x[i, j] = model_rmf.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    for i in nodes:
        u[i] = model_rmf.addVar(vtype="C", name="u(%s)" % i)
        for j in nodes:
            y[i, j] = model_rmf.addVar(vtype="B", name="y(%s,%s)" % (i, j))

    model_rmf.update()

    # 順番制約
    for v in nodes:
        for e in edges:
            if v == e[0]:
                if v == s:
                    model_rmf.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == 1)
                elif v == t:
                    model_rmf.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == -1)
                else:
                    model_rmf.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == 0)

    # xとyの関係
    for i in nodes:
        for k in nodes:
            model_rmf.addConstr(quicksum(x[i, j] for j in nodes if (i, j) in edges) >= y[i, k],
                                "Assign(%s,%s)" % (i, k))
            model_rmf.addConstr(quicksum(x[j, k] for j in nodes if (j, k) in edges) >= y[i, k],
                                "Assign(%s,%s)" % (i, k))
    """
    for k in nodes:
        for k in nodes:
            model_rmf.addConstr(quicksum(x[i, j] for j in nodes if (i, j) in edges) >= y[i])
            model_rmf.addConstr(quicksum(x[j, i] for j in nodes if (i, j) in edges) >= y[i])
    """
    # 同じエッジを使わない
    for (i, j) in edges:
        model_rmf.addConstr(x[i, j] + x[j, i] <= 1)

    for i in nodes:
        model_rmf.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for (i, j) in edges:
        if i != s and j != s:
            model_rmf.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    # 路線長制約 最後に追加
    model_rmf.addConstr(quicksum(edges[i, j] * x[i, j] for (i, j) in edges) <= dist, "Length")

    model_rmf.setObjective(quicksum(y[i, j] * F[(i, j)] for i in nodes for j in nodes), GRB.MAXIMIZE)

    import os
    if os.path.exists(mstFile_path):
        model_rmf.read(mstFile_path)  # 前回の解を許容界として与える

    model_rmf.update()
    model_rmf.optimize()
    # model_rmf.write(mstFile_path)  # 解を出力
    if model_rmf.Status == GRB.INFEASIBLE:
        return None, None, None, None, None

    model_rmf.__data = x, y
    selectE = [j for j in x if x[j].X > EPS]
    selectY = [j for j in y if y[j].X > EPS]
    selectN = []
    for (i, j) in selectY:
        if i not in selectN:
            selectN.append(i)
        if j not in selectN:
            selectN.append(j)

    length = 0
    selectE_id = []
    for e in selectE:
        length += edges[e]
        selectE_id.append(E_id[e])

    return selectN, selectE, selectE_id, length, model_rmf


def createRoute_maxWeight(nodes, edges, s, t, dist, F=None):
    bInit = False
    model_rmf = Model("route_maxFlow")

    x, y, u = {}, {}, {}
    for (i, j) in edges:
        x[i, j] = model_rmf.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    for i in nodes:
        y[i] = model_rmf.addVar(vtype="B", name="y(%s)" % i)
        u[i] = model_rmf.addVar(vtype="C", name="u(%s)" % i)
        # for j in nodes:
        #   f[i, j] = model_rmf.addVar(vtype="C", name="f(%s,%s)" % (i, j))

    model_rmf.update()

    # 順番制約
    for v in nodes:
        for e in edges:
            if v == e[0]:
                if v == s:
                    model_rmf.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == 1)
                elif v == t:
                    model_rmf.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == -1)
                else:
                    model_rmf.addConstr(quicksum(x[i, j] for (i, j) in edges if i == v) - quicksum(
                        x[j, i] for (j, i) in edges if i == v) == 0)

    # xとyの関係
    for i in nodes:
        model_rmf.addConstr(quicksum(x[i, j] for j in nodes if (i, j) in edges) >= y[i])
        model_rmf.addConstr(quicksum(x[j, i] for j in nodes if (i, j) in edges) >= y[i])

    for i in nodes:
        model_rmf.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for (i, j) in edges:
        if i != s and j != s:
            model_rmf.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    # 路線長制約 最後に追加
    model_rmf.addConstr(quicksum(edges[i, j] * x[i, j] for (i, j) in edges) <= dist, "Length")
    # model_rmf.setObjective(quicksum(x[i, j] * N[i][2] for (i,j) in E), GRB.MAXIMIZE)
    if F == None:
        model_rmf.setObjective(quicksum(y[i] * nodes[i][2] for i in nodes), GRB.MAXIMIZE)
    else:
        model_rmf.setObjective(quicksum(y[i] * y[j] * F[(i, j)] for i in nodes for j in nodes if i != j), GRB.MAXIMIZE)

    import os
    if os.path.exists(mstFile_path):
        model_rmf.read(mstFile_path)  # 前回の解を許容界として与える

    model_rmf.update()
    model_rmf.optimize()
    # model_rmf.write(mstFile_path)  # 解を出力
    if model_rmf.Status == GRB.INFEASIBLE:
        return None, None, None

    model_rmf.__data = x, y
    selectE = [j for j in x if x[j].X > EPS]
    selectN = [j for j in y if y[j].X > EPS]

    length = 0
    for e in selectE:
        length += edges[e]

    return selectE, length, model_rmf


# 循環路線作成用2
def createRoute_loop(nodes, D, s, maxdist):
    model_loop = Model("route_loop")
    x, y, u = {}, {}, {}

    # ノードの重みが0のところも除く
    nodes = copy.deepcopy(N)
    for i in N:
        if N[i][2] == 0:
            nodes.pop(i)

    for i in nodes:
        y[i] = model_loop.addVar(vtype="B", name="y(%s)" % (i))
        u[i] = model_loop.addVar(vtype="C", name="u(%s)" % i)
        for j in nodes:
            if i != j:
                x[i, j] = model_loop.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model_loop.update()

    for j in nodes:
        model_loop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == quicksum(x[j, k] for k in nodes if k != j))
        model_loop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == y[j])
        model_loop.addConstr(quicksum(x[j, i] for i in nodes if i != j) == y[j])

    model_loop.addConstr(quicksum(x[i, j] for i in nodes for j in nodes if i != j) == quicksum(y[i] for i in nodes),
                         "numNode&Link")
    model_loop.addConstr(quicksum(D[i, j] * x[i, j] for i in nodes for j in nodes if i != j) <= maxdist, "Length")
    model_loop.addConstr(y[s] == 1)

    for i in nodes:
        model_loop.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for i in nodes:
        for j in nodes:
            if i != j and i != s and j != s:
                model_loop.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    model_loop.update()
    # model_loop.setObjective(quicksum(y[i] * nodes[i][2] for i in nodes), GRB.MAXIMIZE)
    model_loop.setObjective(quicksum(y[i] * nodes[i][2] for i in nodes), GRB.MAXIMIZE)
    model_loop.update()
    model_loop.optimize()

    model_loop.__data = x, y
    stops = [j for j in y if y[j].X > EPS]
    selectLink = [j for j in x if x[j].X > EPS]
    return stops, selectLink, model_loop


def createRoute_loop_Flow(nodes, edges, D, F, s, maxdist):
    model_loop = Model("route_loop")

    x, y, u = {}, {}, {}

    for i in nodes:
        u[i] = model_loop.addVar(vtype="C", name="u(%s)" % i)
        for j in nodes:
            y[i, j] = model_loop.addVar(vtype="B", name="y(%s,%s)" % (i, j))
    for (i, j) in edges:
        x[i, j] = model_loop.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model_loop.update()

    # 順番制約
    for j in nodes:
        model_loop.addConstr(
            quicksum(x[i, j] for i in nodes if (i, j) in edges) == quicksum(x[j, k] for k in nodes if (j, k) in edges))
        # model_loop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == y[j])
        # model_loop.addConstr(quicksum(x[j, i] for i in nodes if i != j) == y[j])

    # xとyの関係
    for i in nodes:
        for k in nodes:
            model_loop.addConstr(quicksum(x[i, j] for j in nodes if (i, j) in edges) >= y[i, k],
                                 "Assign(%s,%s)" % (i, k))
            model_loop.addConstr(quicksum(x[j, k] for j in nodes if (j, k) in edges) >= y[i, k],
                                 "Assign(%s,%s)" % (i, k))

    # 同じエッジを使わない
    for (i, j) in edges:
        model_loop.addConstr(x[i, j] + x[j, i] <= 1)

    # model_loop.addConstr(quicksum(x[i, j] for i in nodes for j in nodes if i != j) == quicksum(y[i] for i in nodes),                         "numNode&Link")
    model_loop.addConstr(quicksum(D[i, j] * x[i, j] for (i, j) in edges) <= maxdist, "Length")

    # 起点終点制約
    model_loop.addConstr(quicksum(x[s, j] for j in nodes if (s, j) in edges) == 1)
    model_loop.addConstr(quicksum(x[i, s] for i in nodes if (i, s) in edges) == 1)

    for i in nodes:
        model_loop.addConstr(u[i] >= 0)

    # 部分巡回路制約
    countN = len(nodes)
    for (i, j) in edges:
        if i != s and j != s:
            model_loop.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    model_loop.update()
    model_loop.setObjective(quicksum(y[i, j] * F[(i, j)] for i in nodes for j in nodes), GRB.MAXIMIZE)
    model_loop.update()
    import os
    if os.path.exists(mstFile_loop):
        model_loop.read(mstFile_loop)  # 前回の解を許容界として与える

    model_loop.optimize()
    # model_loop.write(mstFile_loop)  # 解を出力
    model_loop.__data = x, y
    selectF = [j for j in y if y[j].X > EPS]
    selectE = [j for j in x if x[j].X > EPS]

    routeLength = 0
    selectE_id = []
    for (i, j) in selectE:
        routeLength += D[(i, j)]
        selectE_id.append(E_id[(i, j)])

    return selectF, selectE, selectE_id, routeLength, model_loop


def createRoute_loop_weight(N, D, F, s, maxdist):
    model_loop = Model("route_loop")

    x, y, u = {}, {}, {}

    # ノードの重みが0のところも除く
    nodes = copy.deepcopy(N)
    for i in N:
        if N[i][2] == 0:
            nodes.pop(i)

    for i in nodes:
        y[i] = model_loop.addVar(vtype="B", name="y(%s)" % (i))
        u[i] = model_loop.addVar(vtype="C", name="u(%s)" % i)
        for j in nodes:
            if i != j:
                x[i, j] = model_loop.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model_loop.update()

    for j in nodes:
        model_loop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == quicksum(x[j, k] for k in nodes if k != j))
        model_loop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == y[j])
        model_loop.addConstr(quicksum(x[j, i] for i in nodes if i != j) == y[j])

    model_loop.addConstr(quicksum(x[i, j] for i in nodes for j in nodes if i != j) == quicksum(y[i] for i in nodes),
                         "numNode&Link")
    model_loop.addConstr(quicksum(D[i, j] * x[i, j] for i in nodes for j in nodes if i != j) <= maxdist, "Length")
    model_loop.addConstr(y[s] == 1)

    for i in nodes:
        model_loop.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for i in nodes:
        for j in nodes:
            if i != j and i != s and j != s:
                model_loop.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    model_loop.update()
    model_loop.setObjective(quicksum(y[i] * y[j] * F[(i, j)] for i in nodes for j in nodes if i != j), GRB.MAXIMIZE)
    model_loop.update()
    import os
    if os.path.exists(mstFile_loop):
        model_loop.read(mstFile_loop)  # 前回の解を許容界として与える

    model_loop.optimize()
    # model_loop.write(mstFile_loop)  # 解を出力
    model_loop.__data = x, y
    selectN = [j for j in y if y[j].X > EPS]
    selectE = [j for j in x if x[j].X > EPS]

    routeLength = 0
    for (i, j) in selectE:
        routeLength += D[(i, j)]

    return selectN, selectE, routeLength, model_loop


def updateNodeWeight(N, selectN):
    NS = copy.deepcopy(N)
    for n in NS.keys():
        if n in selectN[0]:
            # 選択ノードに含まれていたら重み0にする
            NS[n] = (NS[n][0], NS[n][1], 0)
    return NS


def updateFlow(F, selectN):
    FS = copy.deepcopy(F)
    for (i, j) in FS.keys():
        if i in selectN and j in selectN:
            # 選択ノードに含まれていたら重み0にする
            FS[(i, j)] = 0
    return FS


def edgeConvertnodes(edges):
    GA = NX.Graph()
    GA.add_edges_from(edges)
    # 連結成分分解
    return list(list(NX.connected_components(GA))[0])


def edgeConvertnodeswithWeight(edges, nodes):
    GA = NX.Graph()
    GA.add_edges_from(edges)
    # 連結成分分解
    nodeList = list(NX.connected_components(GA))
    weight = 0
    for i in nodeList[0]:
        weight += nodes[i][2]
    return nodeList, weight


def addcut(E, cut_edges, model, x):
    G = NX.Graph()
    G.add_edges_from(cut_edges)
    # 連結成分分解
    Components = list(NX.connected_components(G))
    if len(Components) == 1:
        return False

    print("subtour! path count: " + str(
        len(Components)))
    for S in Components:
        model.addConstr(quicksum(
            x[i, j] for i in S for j in S if
            (i, j) in E) <= len(S) - 1, "cut")
        print("cut: len(%s) <= %s" % (
            S, len(S) - 1))

    return True


def calcBestRouteCombination(N, E, routes_condition, routes_node, routes_edge, routes_length, total_length, mainNode,
                             F=None):
    model_brc = Model("bestRouteCombination")
    R = routes_node.keys()

    x, y, u = {}, {}, {}

    # x_irは事前に作成
    for r in R:
        for i in N:
            if i in routes_node[r][0]:
                x[i, r] = 1
            else:
                x[i, r] = 0

    for i in N:
        y[i] = model_brc.addVar(vtype="B", name="y(%s)" % i)

    for r in R:
        u[r] = model_brc.addVar(vtype="B", name="u(%s)" % r)

    model_brc.update()
    for i in N:
        model_brc.addConstr(quicksum(x[i, r] * u[r] for r in R) >= y[i])
        model_brc.addConstr(y[i] <= 1)

    for r in R:
        # 路線長制約
        model_brc.addConstr(sum(routes_length[r] * u[r] for r in R) <= total_length)

    if mainNode is not None:
        H = mainNode[0]
        # mainNodeが選ばれない路線は選ばない
        for r in R:
            model_brc.addConstr(quicksum(x[h, r] for h in H) >= u[r])

    # 目的関数　ノードの重み取得最大化
    if F is None:
        model_brc.setObjective(quicksum(y[i] * N[i][2] for i in N), GRB.MAXIMIZE)
    else:
        model_brc.setObjective(quicksum(y[i] * y[j] * F[(i, j)] for i in N for j in N if i != j), GRB.MAXIMIZE)
    model_brc.update()
    model_brc.optimize()
    model_brc.__data = y, u
    # selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
    selectNode = []
    for r in selectRoute:
        selectNode.append(routes_node[r][0])
    length = 0
    selectHubs = []
    for r in selectRoute:
        length += routes_length[r]
        lst = routes_condition[r].split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print("case:" + str(total_length) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs


def calcRouteCombination_Tree(N, E, routes_node, routes_edge, routes_length, total_length):
    model_brc = Model("routeCombinationTree")
    R = routes_node.keys()
    x, y, v, z, u, w = {}, {}, {}, {}, {}, {}

    # x_ir, z_ijrは事前に作成
    for r in R:
        for i in N:
            if i in routes_node[r][0]:
                x[i, r] = 1
            else:
                x[i, r] = 0

    for r in R:
        for (i, j) in E:
            if (i, j) in routes_edge[r][0]:
                v[i, j, r] = 1
            else:
                v[i, j, r] = 0

    for i in N:
        y[i] = model_brc.addVar(vtype="B", name="y(%s)" % i)
        w[i] = model_brc.addVar(vtype="B", name="w(%s)" % i)

    for (i, j) in E:
        z[i, j] = model_brc.addVar(vtype="B", name="z(%s,%s)" % (i, j))

    for r in R:
        u[r] = model_brc.addVar(vtype="B", name="u(%s)" % r)

    model_brc.update()

    for i in N:
        model_brc.addConstr(quicksum(x[i, r] * u[r] for r in R) >= y[i])
        model_brc.addConstr(y[i] <= 1)

    for (i, j) in E:
        model_brc.addConstr(quicksum(v[i, j, r] * u[r] for r in R) >= z[i, j])
        model_brc.addConstr(z[i, j] <= 1)

    for r in R:
        model_brc.addConstr(sum(routes_length[r] * u[r] for r in R) <= total_length)

    # 目的関数　ノードの重み取得最大化
    model_brc.setObjective(quicksum(y[i] * N[i][2] for i in N), GRB.MAXIMIZE)
    model_brc.update()

    cutoff = 0
    while True:
        cutoff += 1
        model_brc.optimize()
        if model_brc.Status == GRB.INFEASIBLE:
            model_brc.computeIIS()
            print("実行不可能で終了")
            import time
            time.sleep(10)
            model_brc.Write("model.ilp");
            sys.exit()

        edges = [j for j in z if z[j].X > EPS]
        if addcut(E, edges, model_brc, z) == False:
            if model_brc.IsMIP:  # integer variables, components connected: solution found
                break
            elif cutoff > 1000:
                print("部分巡回路除去　回数制限到達")
                break

    model_brc.__data = y, u
    selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
    length = 0
    for r in selectRoute:
        length += routes_length[r]

    print("case " + str(total_length) + ": node:" + str(selectNode) + "_route:" + str(selectRoute) + "_length:" + str(
        length))
    return selectNode, selectRoute, length, model_brc


def select_insideBoundary(N, E, s, range):
    nodes, edges = {}, {}
    for i in N:
        if distance(N[i][0], N[i][1], N[s][0], N[s][1]) <= range:
            nodes[i] = (N[i][0], N[i][1], N[i][2])

    for (i, j) in E:
        if i in nodes and j in nodes:
            edges[i, j] = E[i, j]

    return nodes, edges


def select_insideHubsBoundary(N, E, hubs):
    nodes, edges = {}, {}
    start = hubs[0]
    end = hubs[1]
    minX = 0.0
    maxX = 0.0
    minY = 0.0
    maxY = 0.0
    if N[start][0] < N[end][0]:
        minX = N[start][0] - 1
        maxX = N[end][0] + 1
    else:
        minX = N[end][0] - 1
        maxX = N[start][0] + 1

    if N[start][1] < N[end][1]:
        minY = N[start][1] - 1
        maxY = N[end][1] + 1
    else:
        maxY = N[start][1] + 1
        minY = N[end][1] - 1

    for i in N:
        if N[i][0] >= minX and N[i][0] <= maxX and N[i][1] >= minY and N[i][1] <= maxY:
            # hubのxとyの範囲内に入るノード
            nodes[i] = (N[i][0], N[i][1], N[i][2])

    for (i, j) in E:
        if i in nodes and j in nodes:
            edges[i, j] = E[i, j]

    return nodes, edges


def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def outputRouteList(routes_condition, routes_node, routes_edge, routes_edge_id, routes_length, routes_objVal,
                    routes_CalcTime, routes_MIPGap):
    outRouteList = pd.DataFrame(index=[],
                                columns=["condition",
                                         "nodes",
                                         "edges",
                                         "edgesID",
                                         "length",
                                         "flow", "calcTime", "mipGap"])

    for key, condition in routes_condition.items():
        series = pd.Series(
            [condition, routes_node[key], routes_edge[key], routes_edge_id[key], routes_length[key], routes_objVal[key],
             routes_CalcTime[key], routes_MIPGap[key]],
            index=outRouteList.columns)
        outRouteList = outRouteList.append(series,
                                           ignore_index=True)

    outRouteList.to_csv(basedir + resultdir + routeFileName, index=False)


def outputRouteCombination(totalLength, routes_id, routes_hubs, routes_node, routes_length, routes_objVal,
                           routes_CalcTime):
    outRouteList = pd.DataFrame(index=[],
                                columns=["totalLength",
                                         "routeID",
                                         "hubs",
                                         "nodes",
                                         "routeLength",
                                         "objVal", "calcTime"])

    for key, routeId in routes_id.items():
        series = pd.Series(
            [totalLength[key], routeId, routes_hubs[key], routes_node[key], routes_length[key], routes_objVal[key],
             routes_CalcTime[key]],
            index=outRouteList.columns)
        outRouteList = outRouteList.append(series,
                                           ignore_index=True)

    outRouteList.to_csv(basedir + resultdir + routeCombiFileName, index=False)


def readRouteListCSV(path):
    routes = pd.read_csv(path, encoding='Shift-JIS')
    # データ整形
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime = {}, {}, {}, {}, {}, {}
    totalF = 0
    # 文字列をlist化
    routes.nodes = routes.nodes.apply(literal_eval)
    routes.edges = routes.edges.apply(literal_eval)
    for i, v in routes.iterrows():
        if v["condition"] != -1:  # -1の場合は実行不能
            routes_condition[i] = v["condition"]
            routes_node[i] = v["nodes"]
            routes_edge[i] = v["edges"]
            routes_length[i] = v["length"]
            routes_objVal[i] = v["objVal"]

    return routes_condition, routes_node, routes_edge, routes_length, routes_objVal


def drange(begin, end, step):
    n = begin
    while n + step < end:
        yield n
        n += step


if __name__ == "__main__":
    import sys

    print("---------------------------↓Load_Data↓-------------------------")
    N, E, D, F, G, E_id = load_data(basedir + nodefile, basedir + edgefile, basedir + dMatrixfile, basedir + flowfile)

    print("node count=" + str(len(N)))
    print("edge count=" + str(len(E)))

    outSummaryData = pd.DataFrame(index=[],
                                  columns=["p",
                                           "q",
                                           "limitDistance",
                                           "hubs",
                                           "obj_hubs",
                                           "routes",
                                           "Selected_nodes",
                                           "Length",
                                           "Length_identify",
                                           "cFlow"])

    print("---------------------------↓1.create_Hub↓-------------------------")
    # hubの組合せを作成
    # ハブの数を選択
    #hubs, selects = pmedian(N, D, numHubs, bWeight)

    # mainNode = pmedian(N, D, 1, bWeight)

    """
    # 最短経路の路線に近いところを抽出 メモリ不足となった
    # まずはmainhubを抽出
    mainNode = 315
    print("mainNode:" + str(mainNode))
    hubs = p_path_median(N, G, D, numHubs - 1, mainNode, bWeight)
    print("subhubs:" + str(hubs))
    sys.exit()
    """
    # center問題　解けなかった
    # hubs_center = pcenter(N, D, numHubs)
    # print("hub_center:" + str(hubs_center))
    # TODO debug
    # hubs = [84, 102, 236, 315, 342, 362, 458, 584, 688, 780, 859]

    #常総の主要施設
    hubs = [22,120,123,129,151]
    print("---------------------------↓2.create_Route↓-------------------------")

    # listTerminal = createTerminalCombination(hubs, D, minLength)
    # 環状線のみパターン
    #listTerminal = createTerminalCombinationOnlyloop(hubs)

    # 環状線なしパターン
    listTerminal = createTerminalCombinationNoloop(hubs, D, minLength)
    listTerminal = createTerminalCombinationOnlyloop(hubs,listTerminal)

    # 路線パターン作成
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal = createRouteList(N, E, D, F, G,
                                                                                               listTerminal)
    # routes_condition, routes_node, routes_edge, routes_length, routes_objVal = readRouteListCSV(basedir + debugdir + "routeList_debug.csv")

    # TODO 以下の過程はvrpファイルにて実行する

    print("---------------------------↓3.route_Combination↓-------------------------")
    # 作成した路線パターンを組合せて，全体最適となるパターンを作成する

    totalLengthList, routes_hubs_b, routes_id_b, routes_node_b, routes_length_b, routes_objVal_b, routes_CalcTime_b = {}, {}, {}, {}, {}, {}, {}
    routes_b_totalLength = 0
    count = 0

    mainNode = None
    ls = copy.deepcopy(routes_length)
    for l in ls:
        if routes_length[l] == -1:
            # 実行出来なかった解は除去
            routes_condition.pop(l)
            routes_node.pop(l)
            routes_edge.pop(l)
            routes_length.pop(l)
            routes_objVal.pop(l)

    for length in range(minTotalLength, maxTotalLength + totalLengthSpan, totalLengthSpan):
        selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_condition,
                                                                                     routes_node, routes_edge,
                                                                                     routes_length, length, mainNode, F)
        if mainNode is None:
            mainNode = selectHubs

        if id is not None:
            totalLengthList[count] = length
            routes_id_b[count] = id
            routes_hubs_b[count] = selectHubs
            routes_node_b[count] = selectNodes
            routes_length_b[count] = routeLength
            routes_objVal_b[count] = model_b.objVal
            routes_CalcTime_b[count] = model_b.Runtime
            count += 1

            # TODO 都度出力
            outputRouteCombination(totalLengthList, routes_id_b, routes_hubs_b, routes_node_b, routes_length_b,
                                   routes_objVal_b, routes_CalcTime_b)
