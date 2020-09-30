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
dMatrixfile = "distanceMatrixbyNetwork.csv"

routelistfile = "routeList_r1_2_F.csv"

routeListfiles = ["routeList_r1_2_F.csv", "routeList_r1_5_F.csv", "routeList_r2_F.csv", "routeList_r3_F.csv"]

# debug
#routeListfiles = ["routeList_r3_F.csv"]

now = datetime.datetime.now()
routeCombiSummaryFileName = "routeCombination_S_" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"


#カヴァー済ノードリスト ここに含まれていたらフロー量を0にする
coveredNodeList=[]
#abiko
#coveredNodeList = [3,4,5,25,28,29,30,31,32,33,36,42,43,44,45,46,48,52,53,55,56,57,58,59,60,63,64,65,66,67,68,69,70,71,74,75,76,77,78,79,81,82,83,85,86,87,88,90,102,103,105,106,108,114,115,117,118,119,120,122,144,158,160,161,162,163,165,166,168]


# 路線組合せ時のパラメータ
maxTotalLength = 100000  # 最大路線長
minTotalLength = 5000  # 最小路線長
totalLengthSpan = 5000  # 路線候補を作る間隔
ratioFeederRoute = 1  # 循環路線に対する拠点間路線の路線長割合　1か2を割り当てる
isMIP = False  # TrueならF_ij*z_ijの整数計画  FalseならF_ij*z_i*z_jの二次計画
bSubtour = True  # Trueならルートの分断を許容する（路線網グラフの連結要素が2以上でもOK），Falseなら巡回路除去制約を入れる
bMainNode = True  # TrueならMainNodeにつながる路線のみ考慮

N = None
E = None
D = None
G = None
E_id = None


def drange(begin, end, step):
    n = begin
    while n + step < end:
        yield n
        n += step


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
            # カヴァー済ノードリストに含まれていた場合はフロー量を0にする
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
    dists = pd.read_csv(distanceMatrixData, encoding='Shift-JIS')
    for o, d, dist in zip(dists.originID, dists.destID, dists.distance):
        D[(o, d)] = dist

    print("distance matrix loaded!")

    return N, E, D, F, G, E_id


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

    def addcut(cut_edges):
        G = NX.Graph()
        G.add_edges_from(cut_edges)
        Components = NX.connected_components(G)

        if NX.number_connected_components(G) == 1:
            return False
        for S in Components:
            model_brc.addConstr(quicksum(y[i] * y[j] for i in S for j in S if j > i) <= len(S) - 1)
            print("cut: len(%s) <= %s" % (S, len(S) - 1))
        return True

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
    EPS = 1.e-6

    if bSubtour:
        model_brc.optimize()
    else:
        while True:
            model_brc.optimize()

            selectRoute = [j for j in u if u[j].X > EPS]
            edges = []
            for r in selectRoute:
                for e in routes_edge[r]:
                    edges.append(e)

            if addcut(edges) == False:
                if model_brc.IsMIP:  # integer variables, components connected: solution found
                    break
                for i in y:  # all components connected, switch to integer model
                    y[i].VType = "B"

                model_brc.update()

    model_brc.__data = y, u
    # selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
    selectNode = []
    selectEdge_feeder = []
    selectEdge_Loop = []
    for r in selectRoute:
        selectNode.extend(routes_node[r])
        for e in routes_edge[r]:
            if routes_condition[r].find('roop') != -1 or routes_condition[r].find('loop') != -1:
                selectEdge_Loop.append(E_id[e])
            else:
                selectEdge_feeder.append(E_id[e])

    length = 0
    selectHubs = []

    for r in selectRoute:
        length += routes_length[r]
        lst = routes_condition[r].split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print("case:" + str(total_length) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs, selectEdge_feeder, selectEdge_Loop


def calcBestRouteCombination_ext(N, E, routes_condition, routes_node, routes_edge, routes_length, total_length,
                                 mainNode,
                                 F=None):
    model_brc = Model("bestRouteCombination")
    R = routes_node.keys()
    x, y, z, u = {}, {}, {}, {}

    def addcut(cut_edges):
        G = NX.Graph()
        G.add_edges_from(cut_edges)
        Components = NX.connected_components(G)

        if NX.number_connected_components(G) == 1:
            return False
        for S in Components:
            model_brc.addConstr(quicksum(y[i, j] for i in S for j in S if j > i) <= len(S) - 1)
            print("cut: len(%s) <= %s" % (S, len(S) - 1))
        return True

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

    if mainNode is not None:
        H = mainNode[0]
        # mainNodeが選ばれない路線は選ばない
        for r in R:
            model_brc.addConstr(quicksum(x[h, r] for h in H) >= u[r])

    for i in N:
        for j in N:
            model_brc.addConstr(z[i] + z[j] - 1 <= y[i, j])
            model_brc.addConstr(y[i, j] <= (z[i] + z[j]) / 2)

    # 目的関数　ノードの重み取得最大化
    model_brc.setObjective(quicksum(y[i, j] * F[(i, j)] for i in N for j in N), GRB.MAXIMIZE)
    model_brc.update()

    EPS = 1.e-6
    if bSubtour:
        model_brc.optimize()
    else:
        while True:
            model_brc.optimize()

            selectRoute = [j for j in u if u[j].X > EPS]
            edges = []
            for r in selectRoute:
                for e in routes_edge[r]:
                    edges.append(e)

            if addcut(edges) == False:
                if model_brc.IsMIP:  # integer variables, components connected: solution found
                    break
                for (i, j) in y:  # all components connected, switch to integer model
                    y[i, j].VType = "B"

                model_brc.update()

    model_brc.__data = y, u, z
    # selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
    selectNode = []
    selectEdge_feeder = []
    selectEdge_Loop = []
    for r in selectRoute:
        selectNode.extend(routes_node[r])
        for e in routes_edge[r]:
            if routes_condition[r].find('roop') != -1 or routes_condition[r].find('loop') != -1:
                selectEdge_Loop.append(E_id[e])
            else:
                selectEdge_feeder.append(E_id[e])

    length = 0
    selectHubs = []
    for r in selectRoute:
        length += routes_length[r]
        lst = routes_condition[r].split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print(
        "case:" + str(total_length) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
            selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs, selectEdge_feeder, selectEdge_Loop


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


def outputRouteCombination(totalLength, routes_id, routes_hubs, routes_node, routes_edge_b_feeder, routes_edge_b_loop,
                           routes_length, routes_objVal,
                           routes_CalcTime):
    outRouteList = pd.DataFrame(index=[],
                                columns=["totalLength",
                                         "routeID",
                                         "hubs",
                                         "nodes",
                                         "edges_feeder",
                                         "edges_loop",
                                         "routeLength",
                                         "objVal", "calcTime"])

    for key, routeId in routes_id.items():
        series = pd.Series(
            [totalLength[key], routeId, routes_hubs[key], routes_node[key], routes_edge_b_feeder[key],
             routes_edge_b_loop[key], routes_length[key], routes_objVal[key], routes_CalcTime[key]],
            index=outRouteList.columns)
        outRouteList = outRouteList.append(series,
                                           ignore_index=True)

    outRouteList.to_csv(basedir + resultdir + routeCombiFileName, index=False)


if __name__ == "__main__":
    import sys

    print("---------------------------↓Load_Data↓-------------------------")
    N, E, D, F, G, E_id = load_data(basedir + nodefile, basedir + edgefile, basedir + dMatrixfile,
                                    basedir + flowfile)

    print("node count=" + str(len(N)))
    print("edge count=" + str(len(E)))

    outSummaryData = pd.DataFrame(index=[],
                                  columns=["fileName",
                                           "length",
                                           "nodes",
                                           "edges_feeder",
                                           "edges_loop",
                                           "routeLength",
                                           "objVal"])

    print("---------------------------↓3.route_Combination↓-------------------------")
    if routeListfiles is not None:
        for routelistfile in routeListfiles:
            routeCombiFileName = "routeCombination_" + routelistfile
            routes_condition, routes_node, routes_edge, routes_length, routes_objVal = readRouteListCSV(
                basedir + debugdir + routelistfile)
            totalLengthList, routes_hubs_b, routes_id_b, routes_node_b, routes_edge_b_feeder, routes_edge_b_loop, routes_length_b, routes_objVal_b, routes_CalcTime_b = {}, {}, {}, {}, {}, {}, {}, {}, {}
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

                if isMIP:
                    selectNodes, id, routeLength, model_b, selectHubs, selectEdge_feeder, selectEdge_Loop = calcBestRouteCombination_ext(
                        N, E,
                        routes_condition,
                        routes_node,
                        routes_edge,
                        routes_length,
                        length, mainNode,
                        F)
                else:
                    selectNodes, id, routeLength, model_b, selectHubs, selectEdge_feeder, selectEdge_Loop = calcBestRouteCombination(
                        N, E, routes_condition,
                        routes_node,
                        routes_edge,
                        routes_length, length,
                        mainNode, F)

                if bMainNode and mainNode is None:
                    if len(selectHubs) > 0:
                        mainNode = selectHubs

                if id is not None:
                    totalLengthList[count] = length
                    routes_id_b[count] = id
                    routes_hubs_b[count] = selectHubs
                    routes_node_b[count] = selectNodes
                    routes_edge_b_feeder[count] = selectEdge_feeder
                    routes_edge_b_loop[count] = selectEdge_Loop
                    routes_length_b[count] = routeLength
                    routes_objVal_b[count] = model_b.objVal
                    routes_CalcTime_b[count] = model_b.Runtime
                    count += 1

                    series = pd.Series(
                        [routelistfile, length, selectNodes, selectEdge_feeder,
                         selectEdge_Loop, routeLength, model_b.objVal],
                        index=outSummaryData.columns)
                    outSummaryData = outSummaryData.append(series,
                                                           ignore_index=True)
                    # TODO 都度出力
                    outputRouteCombination(totalLengthList, routes_id_b, routes_hubs_b, routes_node_b,
                                           routes_edge_b_feeder, routes_edge_b_loop, routes_length_b,
                                           routes_objVal_b, routes_CalcTime_b)

        outSummaryData.to_csv(basedir + resultdir + routeCombiSummaryFileName, index=False)
