# coding:utf-8
import copy
import datetime
import math
from ast import literal_eval
import numpy as np
import networkx as NX
import pandas as pd
from gurobipy import *

EPS = 1.e-6
# 解の精度
setParam("MIPGap", 0.00)
setParam("TimeLimit", 1200)
setParam("MIPFocus", 1)

# パラメータ設定
basedir = 'C:/Projects/Gurobi/createRoute/joso/'
resultdir = 'result/'
debugdir = 'debug/'
nodefile = "nodes.csv"
edgefile = "edges.csv"
flowfile = "flows.csv"
routelistfile = "routeList_DRTFlow.csv"
routeListfiles = ["routeList_r1_2_F.csv", ""]
dMatrixfile = "distanceMatrixbyNetwork.csv"

now = datetime.datetime.now()
routeCombiFileName = "routeCombination" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"
vrpResultFileName = "VRPresult" + now.strftime("%Y%m%d%H%M%S") + ".csv"
vrpDemandFileName = "VRPdemand" + now.strftime("%Y%m%d%H%M%S") + ".csv"

# 路線組合せ時のパラメータ
maxTotalLength = 30000  # 最大路線長
minTotalLength = 10000  # 最小路線長
totalLengthSpan = 500  # 路線候補を作る間隔
ratioFeederRoute = 2  # 循環路線に対する拠点間路線の路線長割合　往復を考慮する場合は2を割り当てる

# VRPパラメータ
Q = 3  # 車両の最大乗車人数
dLimit = 30000  # 車両一台の最大距離
tripUnit = 0.008552  # トリップ原単位
ratioShopping = 0.6218  # 買物へ行く確率
ratioHospital = 0.3782  # 通院確率
K = 10  # 試行回数
hour = 10  # 一日の時間カウント　作成したパターンをいくつに分割するか
distTripMin = 1000  # 一定距離以上の需要発生

# VRP計算用　発生トリップ確率の設定
minRatio = 1
maxRatio = 1.5
ratioSpan = 0.5


def drange(begin, end, step):
    n = begin
    while n + step < end:
        yield n
        n += step


def load_data(nodeData, edgeData, distanceMatrixData, flowData=None):
    # 基準データを読み込み
    nodes, edges, flows, dists = {}, {}, {}, {}
    N, E, D, F = {}, {}, {}, {}
    # nodeデータのxyはメートルであること
    nodes = pd.read_csv(nodeData,
                        encoding='Shift-JIS')
    edges = pd.read_csv(edgeData,
                        encoding='Shift-JIS')
    if flowData is not None:
        flows = pd.read_csv(flowData,
                            encoding='Shift-JIS')
        for i, v in flows.iterrows():
            F[(v["originID"], v["destID"])] = v["flow"]

        print("flow data loaded!")

    totalF = 0
    for i, v in nodes.iterrows():
        # N[v["id"]] = (v["x"], v["y"], v["weight"], v["pop"], v["storeArea"], v["hospitalNum"])
        N[v["id"]] = (v["x"], v["y"], v["weight"])

    print("node data loaded!")
    for i, v in edges.iterrows():
        E[(v["originID"], v["destID"])] = v["distance"]

    G = NX.Graph()
    G.add_nodes_from(N)
    # エッジデータ作成
    for key in E:
        value = E[key]
        G.add_edge(key[0], key[1], weight=value)

    print("edge data loaded!")
    dists = pd.read_csv(distanceMatrixData, encoding='Shift-JIS')
    for o, d, dist in zip(dists.originID, dists.destID, dists.distance):
        D[(o, d)] = dist

    print("distance matrix loaded!")

    return N, E, D, F, G


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
    print("hubs:" + str(hubs))
    return hubs


def calcBestRouteCombination(N, E, routes_condition, routes_node, routes_edge, routes_length, total_length, mainNode,
                             F=None):
    model_brc = Model("bestRouteCombination")
    R = routes_node.keys()

    x, y, u = {}, {}, {}

    # x_irは事前に作成
    for r in R:
        for i in N:
            # if i in routes_node[r][0]:
            if i in routes_node[r]:
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
        selectNode.extend(routes_node[r])
    length = 0
    selectHubs = []
    for r in selectRoute:
        length += routes_length[r]
        lst = routes_condition[r].split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print("case:" + str(total_length) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs


def calcBestRouteCombination_ext(N, E, routes_condition, routes_node, routes_edge, routes_length, total_length,
                                 mainNode,
                                 F=None):
    model_brc = Model("bestRouteCombination")
    R = routes_node.keys()

    x, y, z, u = {}, {}, {}, {}

    # x_irは事前に作成
    for r in R:
        for i in N:
            # if i in routes_node[r][0]:
            if i in routes_node[r]:
                x[i, r] = 1
            else:
                x[i, r] = 0

    for i in N:
        z[i] = model_brc.addVar(vtype="B", name="z(%s)" % i)

    for i in N:
        for j in N:
            y[i, j] = model_brc.addVar(vtype="B", name="y(%s,%s)" % (i, j))

    for r in R:
        u[r] = model_brc.addVar(vtype="B", name="u(%s)" % r)

    model_brc.update()
    for i in N:
        model_brc.addConstr(quicksum(x[i, r] * u[r] for r in R) >= z[i])
        # model_brc.addConstr(y[i] <= 1)

    for r in R:
        # 路線長制約
        model_brc.addConstr(sum(routes_length[r] * u[r] for r in R) <= total_length)

    for i in N:
        for j in N:
            model_brc.addConstr(z[i] + z[j] - 1 <= y[i, j])
            model_brc.addConstr(y[i, j] <= (z[i] + z[j]) / 2)

    # 目的関数　ノードの重み取得最大化
    model_brc.setObjective(quicksum(y[i, j] * F[(i, j)] for i in N for j in N), GRB.MAXIMIZE)
    model_brc.update()
    model_brc.optimize()

    if model_brc.Status == GRB.INFEASIBLE:
        model_brc.computeIIS()
        model_brc.write("test1.ilp")
        print("実行不可能で終了")
        sys.exit()

    model_brc.__data = y, u, z
    # selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
    selectNode = []
    for r in selectRoute:
        selectNode.extend(routes_node[r])
    length = 0
    selectHubs = []
    for r in selectRoute:
        length += routes_length[r]
        lst = routes_condition[r].split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print(
        "case:" + str(total_length) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
            selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs


# VRP
def vrp(demand, c, Q, dLimit):
    """solve_vrp -- solve the vehicle routing problem.
       - start with assignment model (depot has a special status)
       - add cuts until all components of the graph are connected
    Parameters:
        - V: set/list of nodes in the graph
        - c[i,j]: cost for traversing edge (i,j)
        - m: number of vehicles available
        - q[i]: demand for customer i
        - Q: vehicle capacity
    Returns the optimum objective value and the list of edges used.
    """

    def vrp_callback(model, where):
        """vrp_callback: add constraint to eliminate infeasible solutions
        Parameters: gurobi standard:
            - model: current model
            - where: indicator for location in the search
        If solution is infeasible, adds a cut using cbLazy
        """
        # remember to set     model.params.DualReductions = 0     before using!
        # remember to set     model.params.LazyConstraints = 1     before using!
        if where != GRB.callback.MIPSOL:
            return
        edges = []
        for (i, j) in x:
            if model.cbGetSolution(x[i, j]) > .5:
                if i != V[0] and j != V[0]:
                    edges.append((i, j))
        G = NX.Graph()
        G.add_edges_from(edges)
        Components = NX.connected_components(G)
        for S in Components:
            S_card = len(S)
            # q_sum = sum(q[i] for i in S)
            q_sum = len(S)
            NS = int(math.ceil(float(q_sum) / (2 * Q)))  # 乗車降車のポイントなので定員に二倍
            S_edges = [(i, j) for i in S for j in S if i < j and (i, j) in edges]
            length = sum(c[demand[i], demand[j]] for i in S for j in S if i < j and (i, j) in edges)
            if S_card >= 3 and (len(S_edges) >= S_card or NS > 1 or length > dLimit):
                model.cbLazy(quicksum(x[i, j] for i in S for j in S if j > i) <= S_card - NS)
                # print("adding cut for", S_edges)
        return

    model = Model("vrp")
    m = model.addVar(vtype="I", name="m")
    x = {}
    V = range(len(demand))
    for i in V:
        for j in V:
            if j > i and i == V[0]:  # depot
                x[i, j] = model.addVar(ub=2, vtype="I", name="x(%s,%s)" % (i, j))
            elif j > i:
                x[i, j] = model.addVar(ub=1, vtype="I", name="x(%s,%s)" % (i, j))
    model.update()

    model.addConstr(quicksum(x[V[0], j] for j in V[1:]) == 2 * m, "DegreeDepot")
    for i in V[1:]:
        model.addConstr(quicksum(x[j, i] for j in V if j < i) +
                        quicksum(x[i, j] for j in V if j > i) == 2, "Degree(%s)" % i)
    for i in V:
        for j in V:
            if j > i:
                if demand[i] == 54 and demand[j] is None:
                    print(str(demand[i]) + "_" + str(demand[j]))
    model.addConstr(quicksum(c.get((demand[i], demand[j])) * x[i, j] for i in V for j in V if j > i) <= dLimit * m)
    # 目的関数の下界を設定　需要をキャパシティで割ったもの
    # model.addConstr(m >= math.ceil((len(demand) / 2) / Q))
    # 目的関数の上界を設定　一人だけ乗った場合
    model.addConstr(m <= math.ceil((len(demand) / 2)))

    # model.setObjective(quicksum(c[i, j] * x[i, j] for i in V for j in V if j > i), GRB.MINIMIZE)
    print(str(demand))

    model.setObjective(quicksum(c.get((demand[i], demand[j])) * x[i, j] for i in V for j in V if j > i), GRB.MINIMIZE)

    # model.setObjective(m, GRB.MINIMIZE)
    model.update()
    model.__data = m, x
    return model, vrp_callback


def createDRTDemand(nodes, D, pattern, hour, tripUnit, distTripMin, routeNode, demands, busRouteLength, ratio):
    np.random.seed()
    for p in range(pattern):
        dem = []  # 一回需要パターン
        for i in nodes:
            # 出発地ループ
            # 人口×原単位で平均移動人数を計算
            # TODO 人口を増加させる検証
            pop = nodes[i][3]
            if pop > 0:
                lHospital, lStore = {}, {}
                acHospital, acStore = 0, 0  # 累積
                # 目的地の候補を作成
                for j in nodes:
                    # 一定距離以上のメッシュ別の病院数,商業面積をカウント
                    if D[(i, j)] >= distTripMin:
                        hCount = nodes[j][4]
                        sCount = nodes[j][5]
                        if hCount > 0:
                            lHospital[j] = (hCount, acHospital, acHospital + hCount)
                            acHospital += hCount

                        if sCount > 0:
                            lStore[j] = (sCount, acStore, acStore + sCount)
                            acStore += sCount

                for k in range(pop):
                    # 乱数を発生させて人数決定
                    randP = np.random.random()
                    if randP <= (tripUnit / hour):

                        randC = np.random.random()
                        # 通院or買物を乱数発生で決定　3割通院
                        bHospital = False
                        if randC <= ratioHospital:
                            bHospital = True

                        randF = np.random.random()
                        if bHospital:
                            # 病院へ向かう
                            randF1 = randF * acHospital
                            for f in lHospital:
                                if routeNode is not None:
                                    if i in routeNode and f in routeNode:
                                        # TODO 迂回率の高さでコントロール?
                                        continue  # バスがカヴァーされているところは対象外として飛ばす

                                # 該当したら需要パターンに追加
                                if lHospital[f][1] <= randF1 and randF1 <= lHospital[f][2]:
                                    series = pd.Series([busRouteLength, ratio, p, i, f], index=demands.columns)
                                    demands = demands.append(series, ignore_index=True)
                                    break

                        else:
                            # 買物へ向かう
                            randF1 = randF * acStore
                            for f in lStore:
                                if routeNode is not None:
                                    if i in routeNode and f in routeNode:
                                        # TODO 迂回率の高さでコントロール?
                                        continue  # バスがカヴァーされているところは対象外として飛ばす
                                # 該当したら需要パターンに追加
                                if lStore[f][1] <= randF1 and randF1 <= lStore[f][2]:
                                    series = pd.Series([busRouteLength, ratio, p, i, f], index=demands.columns)
                                    demands = demands.append(series, ignore_index=True)
                                    break

    return demands


def createDRTDemandbyFlow(nodes, D, pattern, hour, routeNode, demands, busRouteLength, ratio, F):
    np.random.seed()
    for p in range(pattern):
        for (i, j) in F:
            if routeNode is not None:
                if i in routeNode and j in routeNode:
                    continue  # バスがカヴァーされているところは対象外として飛ばす

            # 平均がij間フローのポアソン分布に従って需要発生
            randF = np.random.poisson(lam=F[(i, j)] * ratio / hour)
            if randF > 0:
                for n in range(randF):
                    series = pd.Series([busRouteLength, ratio, p, i, j], index=demands.columns)
                    demands = demands.append(series, ignore_index=True)

    return demands


def sepDemand(I, numSep):
    I_All = None
    I_part = None
    for i in I.keys():
        if i != 0 and i % numSep == 0:  # numSep人以上なら分割
            if I_All is None:
                I_All = [I_part]
            else:
                I_All.append(I_part)

            I_part = None
        else:
            if I_part is None:
                try:
                    I_part = np.array([I[i]])
                except:
                    import traceback
                    print(traceback.format_exc())
            else:
                I_part = np.append(I_part, I[i])

    # 最後の需要パターン追加
    if I_All is None:
        I_All = [I_part]
    elif I_part is not None:
        I_All.append(I_part)

    return I_All


def solveVRP_numVehicles(nodes, D, route_nodes, pattern, depot, tripUnit, distTripMin, hour, Q, dLimit, busRouteLength,
                         ratio, F=None):
    print("---------------------------↓create_demand_pattern↓-------------------------")
    min = 99
    max = 0
    demand = pd.DataFrame(index=[], columns=["busRouteLength", "ratio", "pid", "from", "to"])
    if F is None:
        demand = createDRTDemand(nodes, D, pattern, hour, tripUnit * ratio, distTripMin, route_nodes, demand,
                                 busRouteLength, ratio)
    else:
        demand = createDRTDemandbyFlow(nodes, D, pattern, hour, route_nodes, demand, busRouteLength, ratio, F)

    # 需要のファイル出力
    # fileName = ("VRPDemand_busRouteLength_" + str(busRouteLength) + "_ratio_" + str(ratio) + ".csv")
    # demand.to_csv(basedir + resultdir + fileName, index=False)

    # 需要データの確認
    min_customer = demand.groupby(["pid"]).size().min()
    max_customer = demand.groupby(["pid"]).size().max()
    ave_customer = demand.groupby(["pid"]).size().mean()
    print("pattans:" + str(pattern) + " min:" + str(min_customer) + " max:" + str(max_customer) + " ave:" + str(
        ave_customer))
    print("---------------------------↓solve_VRP↓-------------------------")
    count = 0
    vehicles, listX, lengths, selectNodes, selectEdge = [], [], [], [], []
    for i in range(pattern):
        dem2 = demand.copy()
        # TODO 需要を一定数に分割して解く

        I = dem2.loc[dem2["pid"] == i]["from"]
        J = dem2.loc[dem2["pid"] == i]["to"]
        print("-----------pattern customers:" + str(len(I)) + "-------------")
        """
        nodes_taxi = np.hstack((I.values, J.values))
        # デポ（mainhubひとつめ）を追加
        nodes_taxi = np.insert(nodes_taxi, 0, depot)
        """

        # TODO 需要を一定数に分割して解く
        I_ALL = sepDemand(I, 16)
        J_ALL = sepDemand(J, 16)
        numV = 0
        vlength = 0
        for i in range(len(I_ALL)):
            I_part = I_ALL[i]
            J_part = J_ALL[i]
            if I_part is None or J_part is None:
                continue
            nodes_taxi = np.hstack((I_part, J_part))
            # デポ（mainhubひとつめ）を追加
            nodes_taxi = np.insert(nodes_taxi, 0, depot)
            # 順番維持しつつ重複削除
            """
            li_uniq = []
            for x in nodes_taxi:
                if x not in li_uniq:
                    li_uniq.append(x)
            """
            li_uniq = nodes_taxi
            # 解ける範囲（）まで需要を分割

            oVal = -1
            while True:
                model_vrp, vrp_callback = vrp(li_uniq, D, Q, dLimit)
                # model_vrp = vrp_MTZModel(li_uniq, D, Q, m, dLimit)

                model_vrp.params.DualReductions = 0
                model_vrp.params.LazyConstraints = 1
                model_vrp.optimize(vrp_callback)

                m, x = model_vrp.__data
                if model_vrp.Status == GRB.INFEASIBLE:
                    print("m:" + str(v) + "_infeasible")
                else:
                    v = int(math.ceil(float(m.X)))

                    if oVal == -1 or oVal > model_vrp.objVal:
                        oVal = model_vrp.objVal
                        print("m:" + str(v) + "_length:" + str(oVal))
                        break
                    else:
                        break
            numV += v
            vlength += oVal
            sid = [j for j in x if x[j].X > EPS]
            for (i, j) in sid:
                selectEdge.append((li_uniq[i], li_uniq[j]))

        print("customers:" + str(len(I)) + " total m:" + str(numV) + " length:" + str(vlength))
        vehicles.append(numV)
        lengths.append(vlength)

        numCustomers = demand.groupby(["pid"]).size().tolist()
        """
        #xの出力
              outX = pd.DataFrame(index=[], columns=["i", "j", "Assign"])
              for a in x:
                  outX = outX.append(pd.Series([a[0], a[1], int(x[a].x)], index=outX.columns), ignore_index=True)
              fileName = ("X_ratio" + str(0) + ".csv")
              outX = outX.sort_values(["i", "j", "k"], ascending=True)
              outX.to_csv(basedir + resultdir + fileName, index=False)
              """

    print("customers:" + str(numCustomers))
    print("vehicles:" + str(vehicles))
    print("lengths:" + str(lengths))

    return demand, numCustomers, lengths, vehicles


def readRouteListCSV(path):
    routes = pd.read_csv(path, encoding='Shift-JIS')
    # データ整形
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime = {}, {}, {}, {}, {}, {}
    totalF = 0
    # 文字列をlist化
    routes.nodes = routes.nodes.apply(literal_eval)
    routes.edges = routes.edges.apply(literal_eval)
    for i, v in routes.iterrows():
        if v["condition"] != -1 and v["flow"] != -1:  # -1の場合は実行不能
            routes_condition[i] = v["condition"]
            routes_node[i] = v["nodes"]
            routes_edge[i] = v["edges"]
            # 循環路線の場合はそのままの路線長で，そうではない場合は路線長を一定倍にする
            if v["condition"].find("_loop") > -1 or v["condition"].find("_roop") > -1:
                routes_length[i] = v["length"]
            else:
                routes_length[i] = v["length"] * ratioFeederRoute

            routes_objVal[i] = v["flow"]

    return routes_condition, routes_node, routes_edge, routes_length, routes_objVal


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


if __name__ == "__main__":
    import sys

    print("---------------------------↓Load_Data↓-------------------------")
    N, E, D, F, G = load_data(basedir + nodefile, basedir + edgefile, basedir + dMatrixfile,
                              basedir + flowfile)

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

    print("---------------------------↓3.route_Combination↓-------------------------")
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal = readRouteListCSV(
        basedir + debugdir + routelistfile)
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
        """        
        selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_condition,
                                                                                     routes_node, routes_edge,
                                                                                     routes_length, length, mainNode, F)
        """
        selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_condition,
                                                                                     routes_node, routes_edge,
                                                                                     routes_length, length,
                                                                                     mainNode, F)

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
    sys.exit()

    print("---------------------------↓4.solveVRP↓-------------------------")
    # ルートの組合せでVRP回す
    mainNode = pmedian(N, D, 1, True)
    mainhub = mainNode[0]
    outVRPData = pd.DataFrame(index=[], columns=["routeLength", "tripRatio", "VRPcustomers", "VRPlengths", "vehicles"])
    outdemand = pd.DataFrame(index=[], columns=["busRouteLength", "ratio", "pid", "from", "to"])

    for ratio in drange(minRatio, maxRatio + ratioSpan, ratioSpan):
        # バス路線無しver
        demand, numCustomers, lengths, vehicles = solveVRP_numVehicles(N, D, None, K, mainhub, tripUnit, distTripMin,
                                                                       hour, Q, dLimit, 0, ratio, F)
        series = pd.Series([0, ratio, numCustomers, lengths, vehicles], index=outVRPData.columns)
        outdemand = outdemand.append(demand)
        outVRPData = outVRPData.append(series, ignore_index=True)
        outVRPData.to_csv(basedir + resultdir + vrpResultFileName, index=False)
        outdemand.to_csv(basedir + resultdir + vrpDemandFileName, index=False)
        # 路線がある場合
        for s in routes_node_b:
            demand, numCustomers, lengths, vehicles = solveVRP_numVehicles(N, D, routes_node_b[s], K, mainhub, tripUnit,
                                                                           distTripMin, hour, Q, dLimit,
                                                                           totalLengthList[s], ratio, F)
            series = pd.Series([totalLengthList[s], ratio, numCustomers, lengths, vehicles], index=outVRPData.columns)
            outVRPData = outVRPData.append(series, ignore_index=True)
            outdemand = outdemand.append(demand)
            outVRPData.to_csv(basedir + resultdir + vrpResultFileName, index=False)
            outdemand.to_csv(basedir + resultdir + vrpDemandFileName, index=False)

            # outVRPData.to_csv(basedir + resultdir + vrpResultFileName, index=False)
