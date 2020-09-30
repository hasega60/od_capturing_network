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
# setParam("MIPGap", 0.0001)
setParam("TimeLimit", 600)
setParam("MIPFocus", 1)

# パラメータ設定
basedir = 'C:/Projects/Gurobi/createRoute/kamisu/'
resultdir = 'result/'
debugdir = 'debug/'
nodefile = "nodes.csv"
edgefile = "edges.csv"
flowfile = "flows.csv"
dMatrixfile = "distanceMatrixbyNetwork.csv"

#モデルの解出力ファイル名
mstFile_roop = "sol_model_roop.sol"
mstFile_path= "sol_model_path.sol"

bWeight = True  # pmedianでノードの重みを考慮するか
numHubs = 7  # ハブの数

# 路線作成時のパラメータ
maxLength = 30000  # 最大路線長
maxRoopLength = 20000  # ループ最大延長
minLength = 5000  # 最小路線長
span = 5000  # 路線候補を作る間隔
bMinLength = False #拠点間ルートは最短経路で作成するか
maxNodeSize_createRoute_roop_Flow = 60  # 循環線を作成する場合のノード上限数 重みが大きく，距離が小さいノードから順に追加する


# 路線組合せ時のパラメータ
maxTotalLength = 50000  # 最大路線長
minTotalLength = 5000  # 最小路線長
totalLengthSpan = 5000  # 路線候補を作る間隔

# VRPパラメータ
Q = 4  # 車両の最大乗車人数
dLimit = 30000  # 車両一台の最大距離
tripUnit = 0.008552  # トリップ原単位
ratioShopping = 0.6218  # 買物へ行く確率
ratioHospital = 0.3782  # 通院確率
K = 10  # 試行回数
hour = 10  # 一日の時間カウント　作成したパターンをいくつに分割するか
distTripMin = 1000  # 一定距離以上の需要発生

# VRP計算用　発生トリップ確率の設定
minRatio = 1
maxRatio = 3
ratioSpan = 1
bModal = False  # VRPでバス乗り継ぎを考慮するか エリアわけを行って

now = datetime.datetime.now()
routeFileName = "routeList" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"
routeCombiFileName = "routeCombination" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"
vrpResultFileName = "VRPresult" + now.strftime("%Y%m%d%H%M%S") + ".csv"
vrpDemandFileName = "VRPdemand" + now.strftime("%Y%m%d%H%M%S") + ".csv"

N = None
E = None
D = None
G = None


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

    return hubs


def pcenter_existing(N, D, yExt, q):
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
    model3.__data = x, y

    return model3


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


def createTerminalCombinationNoRoop(hubs, D, minLength):
    list = []
    for hub1 in hubs:
        for hub2 in hubs:
            if list not in [(hub1, hub2)]:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 != hub2 and D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list


def createRouteList(N, E, D, F, G, listTerminal, minLength, maxLength, maxRoopLength, span):
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime = {}, {}, {}, {}, {}, {}
    count = 0
    print("createRoute:" + str(listTerminal))
    for hub in listTerminal:
        shortestLength = 0
        model = None
        import os
        # mstファイルを削除
        if os.path.exists(mstFile_roop):
            os.remove(mstFile_roop)

        if os.path.exists(mstFile_path):
                os.remove(mstFile_path)

        if hub[0] == hub[1]:
            for length in range(minLength, maxRoopLength + span, span):
                print("---------------------------↓" + "solve:" + str(hub[0]) + ", " + str(hub[1]) + "," + str(
                    length) + "↓-------------------------")

                # ハブから一定範囲内(路線長制限の半分)のエッジを対象にする
                NS, ES = select_insideBoundary(N, E, hub[0], length / 2)
                print("size_node:" + str(len(NS)) + " edge:" + str(len(ES)))
                selectN, selectE, routeLength, model = createRoute_roop_Flow(NS, D, F, hub[0], length)

                if selectN is not None:
                    routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str(length) + "_" + str(
                        "roop")
                    routes_node[count] = selectN
                    routes_edge[count] = selectE
                    routes_length[count] = routeLength
                    routes_objVal[count] = model.objVal
                    routes_CalcTime[count] = model.Runtime
                else:
                    selectN = []
                    routes_node[count] = []
                    routes_edge[count] = []
                    routes_length[count] = -1
                    routes_objVal[count] = -1
                    routes_CalcTime[count] = -1
                count += 1
                # TODO 都度出力
                outputRouteList(routes_condition, routes_node, routes_edge, routes_length, routes_objVal,
                                routes_CalcTime)

        else:
            # ハブ間路線
            # 最短経路
            shortestLength = NX.dijkstra_path_length(G, hub[0], hub[1])
            weightShortestPath = 0
            sPath = NX.dijkstra_path(G, hub[0], hub[1])
            for i in sPath:
                weightShortestPath += N[i][2]

            print("shortestLength_(" + str(hub[0]) + ", " + str(hub[1]) + "):" + str(shortestLength))
            routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str("spath")
            # TODO 都度出力

            routes_node[count] = sPath
            routes_edge[count] = []
            routes_length[count] = shortestLength
            routes_objVal[count] = weightShortestPath
            routes_CalcTime[count] = 0
            count += 1
            outputRouteList(routes_condition, routes_node, routes_edge, routes_length,
                            routes_objVal, routes_CalcTime)

            # hub間路線網パターン作成

            for length in range(minLength, maxLength + span, span):
                model = None
                print("solve:" + str(hub[0]) + ", " + str(hub[1]) + "," + str(length))
                # 最大路線長を超えないパターンを作成
                if shortestLength < length:
                    # selectE, routeLength, model = createRoute_maxWeight(N, E, hub[0], hub[1], length)
                    selectE, routeLength, model = createRoute_maxWeight(N, E, hub[0], hub[1], length, F)
                    routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str(
                        length) + "_" + str(
                        "F")
                    if selectE is not None:
                        selectN = edgeConvertnodes(selectE)
                        routes_node[count] = selectN
                        routes_edge[count] = selectE
                        routes_length[count] = routeLength
                        routes_objVal[count] = model.objVal
                        routes_CalcTime[count] = model.Runtime

                    else:
                        # 実行不可能解
                        selectN_r = []
                        routes_node[count] = []
                        routes_edge[count] = []
                        routes_length[count] = -1
                        routes_objVal[count] = -1
                        routes_CalcTime[count] = -1

                    count += 1
                    outputRouteList(routes_condition, routes_node, routes_edge, routes_length,
                                    routes_objVal, routes_CalcTime)

                    if selectE is not None:
                        # すでに上記路線が選ばれた場合の路線を作成する
                        N_selected = updateNodeWeight(N, selectN)
                        F_selected = updateFlow(F, selectN)
                        selectE_r, routeLength_r, model_r = createRoute_maxWeight(N_selected, E, hub[0],
                                                                                  hub[1],
                                                                                  length,
                                                                                  F_selected)
                        routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str(
                            length) + "_" + str(
                            "R")
                        if selectE_r is not None:
                            selectN_r = edgeConvertnodes(selectE_r)
                            routes_node[count] = selectN_r
                            routes_edge[count] = selectE_r
                            routes_length[count] = routeLength_r
                            routes_objVal[count] = model_r.objVal
                            routes_CalcTime[count] = model_r.Runtime
                        else:
                            selectN_r = []
                            routes_node[count] = []
                            routes_edge[count] = []
                            routes_length[count] = -1
                            routes_objVal[count] = -1
                            routes_CalcTime[count] = -1

                        count += 1
                        # TODO 都度出力
                        outputRouteList(routes_condition, routes_node, routes_edge, routes_length,
                                        routes_objVal, routes_CalcTime)

    return routes_condition, routes_node, routes_edge, routes_length, routes_objVal


def createRoute_minDist(nodes, edges, s, t, minWeight):
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
    for (i, j) in E:
        if i != s and j != s:
            model_rmf.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    # 路線長制約 最後に追加
    model_rmf.addConstr(quicksum(y[i] * nodes[i][2] for i in nodes) >= minWeight, "minWeight")
    model_rmf.setObjective(quicksum(edges[i, j] * x[i, j] for (i, j) in edges), GRB.MINIMIZE)

    model_rmf.update()
    model_rmf.optimize()
    model_rmf.__data = x, y
    selectE = [j for j in x if x[j].X > EPS]
    selectN = [j for j in y if y[j].X > EPS]

    length = 0
    for e in selectE:
        length += edges[e]

    return selectE, length, model_rmf


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
    for (i, j) in E:
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
    #model_rmf.write(mstFile_path)  # 解を出力
    if model_rmf.Status == GRB.INFEASIBLE:
        return None, None, None

    model_rmf.__data = x, y
    selectE = [j for j in x if x[j].X > EPS]
    selectN = [j for j in y if y[j].X > EPS]

    length = 0
    for e in selectE:
        length += edges[e]

    return selectE, length, model_rmf


def createRoute(N, E, s, t, dist, model):
    bInit = False
    if model is None:
        model_rmf = Model("route_maxFlow")
        bInit = True
    else:
        model_rmf = model

    if bInit:
        # 初回はすべての制約を設定
        x, y = {}, {}
        for (i, j) in E:
            x[i, j] = model_rmf.addVar(vtype="B", name="x(%s,%s)" % (i, j))

        for i in N:
            y[i] = model_rmf.addVar(vtype="B", name="y(%s)" % i)
        """
        for i in N:
            for j in N:
                y[i, j] = model_rmf.addVar(vtype="B", name="y(%s,%s)" % (i, j))
        """
        model_rmf.update()

        for v in N:
            for e in E:
                if v == e[0]:
                    if v == s:
                        model_rmf.addConstr(quicksum(x[i, j] for (i, j) in E if i == v) - quicksum(
                            x[j, i] for (j, i) in E if i == v) == 1)
                    elif v == t:
                        model_rmf.addConstr(quicksum(x[i, j] for (i, j) in E if i == v) - quicksum(
                            x[j, i] for (j, i) in E if i == v) == -1)
                    else:
                        model_rmf.addConstr(quicksum(x[i, j] for (i, j) in E if i == v) - quicksum(
                            x[j, i] for (j, i) in E if i == v) == 0)
        """
        for (i, j) in E:
            model_rmf.addConstr(x[i, j] >= 0, "Assign(%s,%s)" % (i, j))

        for i in N:
            model_rmf.addConstr(y[i] >= 0, "Assign(%s)" % i)
        """
        # model_rmf.setObjective(quicksum(y[i] * N[i][2] for i in N), GRB.MAXIMIZE)

        # 路線長制約 最後に追加
        model_rmf.addConstr(quicksum(E[i, j] * x[i, j] for (i, j) in E) <= dist, "length(%s,%s)" % (i, j))
        model_rmf.setObjective((quicksum(x[i, j] * (N[i][2] + N[j][2]) for (i, j) in E) + (N[s][2] + N[t][2])) / 2,
                               GRB.MAXIMIZE)

        # model_rmf.setObjective(quicksum(y[i, j] * N[i][2] + y[i, j] * N[j][2] for i in N for j in N if (i, j) in y), GRB.MAXIMIZE)
        model_rmf.update()

    else:
        # 路線長制約を変更
        x, y = model_rmf.__data
        model_rmf.remove(model.getConstrs()[len(model.getConstrs()) - 1])
        model_rmf.addConstr(quicksum(E[i, j] * x[i, j] for (i, j) in E) <= dist, "maxLength" + str(dist))
        model_rmf.update()

    model_rmf.optimize()
    if model_rmf.Status == GRB.INFEASIBLE:
        print("createRoute:実行不可能で終了_" + str(s) + "_" + str(t) + "_" + str(dist))
        # outIIS(model_rmf)
        return None, None, model_rmf
    else:
        model_rmf.__data = x, y
        # selectN = [j for j in y if y[j].X > EPS]
        selectE = [j for j in x if x[j].X > EPS]
        length = 0
        for e in selectE:
            length += E[e]

        return selectE, length, model_rmf


def createTSPModel(N, E, s, dist, model):
    def tsp_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        edges = []
        for (i, j) in x:
            if model.cbGetSolution(x[i, j]) > EPS:
                edges.append((i, j))

        Ge = NX.Graph()
        Ge.add_edges_from(edges)
        Components = list(NX.connected_components(Ge))

        if len(Components) == 1:
            selectE = [j for j in x if x[j].X > EPS]
            return selectE
        for S in Components:
            model.cbLazy(quicksum(x[i, j] for i in S for j in S if j > i) <= len(S) - 1, "cut")
            # print "cut: len(%s) <= %s" % (S,len(S)-1)
        return

    model_rTSP = Model("Model_rTSP")
    x = {}
    for (i, j) in E:
        x[i, j] = model_rTSP.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    for v in N:
        for e in E:
            if v == e[0]:
                # 始点から終点に流れる
                model_rTSP.addConstr(
                    quicksum(x[i, j] for (i, j) in E if i == v) - quicksum(x[j, i] for (j, i) in E if i == v) == 0)

    # エッジは一回しか選ばれない
    for (i, j) in E:
        model_rTSP.addConstr((x[i, j] + x[j, i]) <= 1)

    # ハブを含むエッジは必ず一つ選ばれる
    model_rTSP.addConstr(quicksum(x[i, j] for (i, j) in E if i == s) == 1)
    model_rTSP.addConstr(quicksum(x[i, j] for (i, j) in E if j == s) == 1)

    model_rTSP.update()

    # 路線長制約 最後に追加
    model_rTSP.addConstr(quicksum(E[i, j] * x[i, j] for (i, j) in E) <= dist, "Length")

    model_rTSP.setObjective((quicksum(x[i, j] * (N[i][2] + N[j][2]) for (i, j) in E)) / 2, GRB.MAXIMIZE)
    model_rTSP.update()
    model_rTSP.__data = x

    return model_rTSP, tsp_callback


# 実行に非常に時間がかかるためsaveFlowに移行
def createRoute_TSP(N, E, s, dist, model):
    model_rTSP, tsp_callback = createTSPModel(N, E, s, dist, model)

    # 制約を追加後，MIPを解きなおす形
    while True:
        model_rTSP.optimize()
        x = model_rTSP.__data
        if model_rTSP.Status == GRB.INFEASIBLE:
            print("createRouteTSP:実行不可能で終了_" + str(s) + "_" + str(dist))
            # outIIS(model_rmf)
            return None, None, model_rTSP

        edges = []
        for (i, j) in x:
            if x[i, j].X > EPS:
                edges.append((i, j))

        if addcut(E, edges, model_rTSP, x) == False:
            if model_rTSP.IsMIP:  # integer variables, components connected: solution found
                model_rTSP.__data = x
                break
            for (i, j) in x:  # all components connected, switch to integer model
                x[i, j].VType = "B"
                model_rTSP.update()
    """

    # 分枝限定法を適用して解く
    model_rTSP.params.DualReductions = 0
    model_rTSP.params.LazyConstraints = 1
    model_rTSP.optimize(tsp_callback)
    """

    x = model_rTSP.__data
    selectE = [j for j in x if x[j].X > EPS]
    length = 0
    for e in selectE:
        length += E[e]

    return selectE, length, model_rTSP


def createRoute_saveFlow(N, E, D, s, dist, model):
    model_rSF = Model("Model_rSF")
    x, y, f = {}, {}, {}
    for i in N:
        y[i] = model_rSF.addVar(vtype="B", name="y(%s)" % (i))
        for j in N:
            x[i, j] = model_rSF.addVar(vtype="B", name="x(%s,%s)" % (i, j))
            f[i, j] = model_rSF.addVar(vtype=GRB.INTEGER, name="f(%s,%s)" % (i, j))

    model_rSF.update()
    # x_ijとy_iの関係式
    for i in N:
        model_rSF.addConstr(quicksum(x[i, j] for j in N if i != j) == y[i])
        model_rSF.addConstr(quicksum(x[j, i] for j in N if i != j) == y[i])
        for j in N:
            if i != j:
                # ij間の移動が無い場合はfは0となる
                model_rSF.addConstr(f[i, j] <= len(N) * x[i, j])

    NS = copy.deepcopy(N)  # 出発地を除いたノード集合
    NS.pop(s)
    for i in NS:
        # ノードiの訪問でフロー量が1増加していく
        model_rSF.addConstr(quicksum(f[h, i] for h in N if h != i) + y[i] == quicksum(f[i, j] for j in N if j != i))
        # 出発地から出た直後はのfは0
        model_rSF.addConstr(f[s, i] == 0)

    # 出発地は必ず選ぶ
    model_rSF.addConstr(y[s] == 1)

    model_rSF.update()

    # 路線長制約 最後に追加
    model_rSF.addConstr(quicksum(D[i, j] * x[i, j] for i in N for j in N if j != i) <= dist, "Length")

    model_rSF.setObjective(quicksum(y[i] * N[i][2] for i in N), GRB.MAXIMIZE)
    model_rSF.update()
    model_rSF.optimize()
    if model_rSF.Status == GRB.INFEASIBLE:
        outIIS(model_rSF)
        print("実行不可能で終了")
        return None, None, model_rSF
    elif model_rSF.Status == GRB.TIME_LIMIT:
        print("timeLimit")
        return None, None, model_rSF
    else:
        model_rSF.__data = x, y
        selectN = [j for j in y if y[j].X > EPS]
        selectX = [j for j in x if x[j].X > EPS]
        selectE = []
        length = 0
        for x in selectX:
            length += D[x]
            if x in E:
                selectE.append(x)
            else:
                dPath = NX.dijkstra_path(G, x[0], x[1])
                for i in range(len(dPath)):
                    if i + 1 < len(dPath):
                        selectE.append((dPath[i], dPath[i + 1]))

        return selectN, selectE, length, model_rSF


# 循環路線作成用2
def createRoute_roop(N, D, s, maxdist):
    model_roop = Model("route_roop")
    x, y, u = {}, {}, {}

    # ノードの重みが0のところも除く
    nodes = copy.deepcopy(N)
    for i in N:
        if N[i][2] == 0:
            nodes.pop(i)

    for i in nodes:
        y[i] = model_roop.addVar(vtype="B", name="y(%s)" % (i))
        u[i] = model_roop.addVar(vtype="C", name="u(%s)" % i)
        for j in nodes:
            if i != j:
                x[i, j] = model_roop.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model_roop.update()

    for j in nodes:
        model_roop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == quicksum(x[j, k] for k in nodes if k != j))
        model_roop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == y[j])
        model_roop.addConstr(quicksum(x[j, i] for i in nodes if i != j) == y[j])

    model_roop.addConstr(quicksum(x[i, j] for i in nodes for j in nodes if i != j) == quicksum(y[i] for i in nodes),
                         "numNode&Link")
    model_roop.addConstr(quicksum(D[i, j] * x[i, j] for i in nodes for j in nodes if i != j) <= maxdist, "Length")
    model_roop.addConstr(y[s] == 1)

    for i in nodes:
        model_roop.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for i in nodes:
        for j in nodes:
            if i != j and i != s and j != s:
                model_roop.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    model_roop.update()
    # model_roop.setObjective(quicksum(y[i] * nodes[i][2] for i in nodes), GRB.MAXIMIZE)
    model_roop.setObjective(quicksum(y[i] * nodes[i][2] for i in nodes), GRB.MAXIMIZE)
    model_roop.update()
    model_roop.optimize()

    model_roop.__data = x, y
    stops = [j for j in y if y[j].X > EPS]
    selectLink = [j for j in x if x[j].X > EPS]
    return stops, selectLink, model_roop


def createRoute_roop_Flow(N, D, F, s, maxdist):
    model_roop = Model("route_roop")
    # model_roop.setParam("MIPFocus", 2)  # 上界を下げる形で実装

    x, y, u = {}, {}, {}

    # ノードの重みが0のところも除く
    nodes = copy.deepcopy(N)
    for i in N:
        if N[i][2] == 0:
            nodes.pop(i)
    """
    # ノードのサイズが最大値を超えたら重みが大きい順に格納
    if len(nodes) > maxNodeSize_createRoute_roop_Flow:
        con = 0
        model_roop.setParam("TimeLimit", 600)
        for k, v in sorted(nodes.items(), key=lambda x: x[1][2], reverse=True):
            if con <= maxNodeSize_createRoute_roop_Flow:
                con += 1
            else:
                nodes.pop(k)
    """
    for i in nodes:
        y[i] = model_roop.addVar(vtype="B", name="y(%s)" % (i))
        u[i] = model_roop.addVar(vtype="C", name="u(%s)" % i)
        for j in nodes:
            if i != j:
                x[i, j] = model_roop.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model_roop.update()

    for j in nodes:
        model_roop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == quicksum(x[j, k] for k in nodes if k != j))
        model_roop.addConstr(quicksum(x[i, j] for i in nodes if i != j) == y[j])
        model_roop.addConstr(quicksum(x[j, i] for i in nodes if i != j) == y[j])

    model_roop.addConstr(quicksum(x[i, j] for i in nodes for j in nodes if i != j) == quicksum(y[i] for i in nodes),
                         "numNode&Link")
    model_roop.addConstr(quicksum(D[i, j] * x[i, j] for i in nodes for j in nodes if i != j) <= maxdist, "Length")
    model_roop.addConstr(y[s] == 1)

    for i in nodes:
        model_roop.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(nodes)
    for i in nodes:
        for j in nodes:
            if i != j and i != s and j != s:
                model_roop.addConstr(u[i] - u[j] + countN * x[i, j] <= countN - 1)

    model_roop.update()
    model_roop.setObjective(quicksum(y[i] * y[j] * F[(i, j)] for i in nodes for j in nodes if i != j), GRB.MAXIMIZE)
    model_roop.update()
    import os
    if os.path.exists(mstFile_roop):
        model_roop.read(mstFile_roop)  # 前回の解を許容界として与える

    model_roop.optimize()
    #model_roop.write(mstFile_roop)  # 解を出力
    model_roop.__data = x, y
    selectN = [j for j in y if y[j].X > EPS]
    selectE = [j for j in x if x[j].X > EPS]

    routeLength = 0
    for (i, j) in selectE:
        routeLength += D[(i, j)]

    return selectN, selectE, routeLength, model_roop


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
        if i in selectN[0] and j in selectN[0]:
            # 選択ノードに含まれていたら重み0にする
            FS[(i, j)] = 0
    return FS


def edgeConvertnodes(edges):
    GA = NX.Graph()
    GA.add_edges_from(edges)
    # 連結成分分解
    return list(NX.connected_components(GA))


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
    selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
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
        if i in nodes or j in nodes:
            edges[i, j] = E[i, j]

    return nodes, edges


def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def outputRouteList(routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime):
    outRouteList = pd.DataFrame(index=[],
                                columns=["condition",
                                         "nodes",
                                         "edges",
                                         "length",
                                         "objVal", "calcTime"])

    for key, condition in routes_condition.items():
        series = pd.Series(
            [condition, routes_node[key], routes_edge[key], routes_length[key], routes_objVal[key],
             routes_CalcTime[key]],
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
            NS = int(math.ceil(float(q_sum) / Q))
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

    model.addConstr(quicksum(c[demand[i], demand[j]] * x[i, j] for i in V for j in V if j > i) <= dLimit * m)
    # 目的関数の下界を設定　需要をキャパシティで割ったもの
    model.addConstr(m >= math.ceil((len(demand) / 2) / Q))
    # 目的関数の上界を設定　一人だけ乗った場合
    model.addConstr(m <= math.ceil((len(demand) / 2)))

    # model.setObjective(quicksum(c[i, j] * x[i, j] for i in V for j in V if j > i), GRB.MINIMIZE)
    # model.setObjective(quicksum(c[demand[i], demand[j]] * x[i, j] for i in V for j in V if j > i), GRB.MINIMIZE)
    model.setObjective(m, GRB.MINIMIZE)
    model.update()
    model.__data = m, x
    return model, vrp_callback


def vrp_MTZModel(demand, D, Q, m, dLimit):
    model_vrpM = Model("Model_vrpM")
    m = model_vrpM.addVar(vtype="C", name="m")
    x, u, q = {}, {}, {}
    V = range(len(demand))
    for i in V:
        u[i] = model_vrpM.addVar(vtype="I", name="u(%s)" % i)
        q[i] = model_vrpM.addVar(vtype="I", name="q(%s)" % i)
        for j in V:
            x[i, j] = model_vrpM.addVar(vtype="B", name="x(%s,%s)" % (i, j))

    model_vrpM.update()

    for i in V:
        if i == V[0]:
            model_vrpM.addConstr(quicksum(x[j, i] for j in V if j < i) == m)
            model_vrpM.addConstr(quicksum(x[i, j] for j in V if j < i) == m)
        else:
            model_vrpM.addConstr(quicksum(x[j, i] for j in V if j < i) == 1)
            model_vrpM.addConstr(quicksum(x[i, j] for j in V if j < i) == 1)

    for i in V:
        for j in V:
            if j != i:
                model_vrpM.addConstr(u[i] - u[j] + (Q - q[i] - q[j]) * x[i, j] <= Q - q[j])  # 容量制約

    for i in V:
        model_vrpM.addConstr(q[i] <= u[i])
        model_vrpM.addConstr(u[i] <= Q)

    model_vrpM.setObjective(quicksum(D[demand[i], demand[j]] * x[i, j] for i in V for j in V if j > i),
                            GRB.MINIMIZE)
    model_vrpM.update()

    model_vrpM.__data = m, x
    return model_vrpM


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


# VRP需要作成flow考慮
def createDRTDemandbyFlow(nodes, D, pattern, hour, tripUnit, distTripMin, routeNode, demands, busRouteLength, ratio, F):
    np.random.seed()
    for p in range(pattern):
        for (i, j) in F:
            # 乱数を発生させて人数決定
            randF = np.random.random()
            if randF <= F[(i, j)] * ratio / hour:
                if routeNode is not None:
                    if i in routeNode and j in routeNode:
                        continue  # バスがカヴァーされているところは対象外として飛ばす
                series = pd.Series([busRouteLength, ratio, p, i, j], index=demands.columns)
                demands = demands.append(series, ignore_index=True)

    return demands


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
        demand = createDRTDemandbyFlow(nodes, D, pattern, hour, tripUnit * ratio, distTripMin, route_nodes, demand,
                                       busRouteLength, ratio, F)

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
    # TODO 需要を一定数に分割して解く
    vehicles, listX, lengths, selectNodes = [], [], [], []
    for i in range(pattern):
        dem2 = demand.copy()
        I = dem2.loc[dem2["pid"] == i]["from"]
        J = dem2.loc[dem2["pid"] == i]["to"]
        nodes_taxi = np.hstack((I.values, J.values))
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
                print("m:" + str(m) + "_infeasible")
            elif oVal == -1 or oVal > model_vrp.objVal:
                oVal = model_vrp.objVal
                print("m:" + str(m) + "_length:")
                break
            else:
                break

        # vehicles.append(m)
        vehicles.append(model_vrp.objVal)
        sid = [j for j in x if x[j].X > EPS]
        selectEdge = []
        length = 0
        for (i, j) in sid:
            selectEdge.append((li_uniq[i], li_uniq[j]))
            length += D[li_uniq[i], li_uniq[j]]

        lengths.append(length)

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


if __name__ == "__main__":
    import sys

    print("---------------------------↓Load_Data↓-------------------------")
    N, E, D, F, G = load_data(basedir + nodefile, basedir + edgefile, basedir + dMatrixfile, basedir + flowfile)

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

    hubs = []

    # メインハブ・サブハブで考える場合
    """
    for p in range(1, 4):
        q = numHubs - p
        mainhubs = pmedian(N, D, p, bWeight)
        subhubs = pcenter_existing(N, D, mainhubs, q)
        if mainhubs is not None or subhubs is not None:
            hubs.append([mainhubs, subhubs])
            print("hubs:" + str(mainhubs) + "_" + str(subhubs))
    print(str(hubs))"""

    # 通常のpmedian
    hubs = pmedian(N, D, numHubs, bWeight)
    mainNode = pmedian(N, D, 1, bWeight)

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

    print("---------------------------↓2.create_Route↓-------------------------")

    listTerminal = createTerminalCombination(hubs, D, minLength)
    # 環状線なしパターン
    # listTerminal = createTerminalCombinationNoRoop(hubs, D, minLength)
    # 路線パターン作成
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal = createRouteList(N, E, D, F, G,
                                                                                               listTerminal, minLength,
                                                                                               maxLength, maxRoopLength,
                                                                                               span)

    # routes_condition, routes_node, routes_edge, routes_length, routes_objVal = readRouteListCSV(basedir + debugdir + "routeList_debug.csv")

    sys.exit()
    #TODO 以下の過程はvrpファイルにて実行する

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

    print("---------------------------↓4.solveVRP↓-------------------------")
    # ルートの組合せでVRP回す
    mainhub = mainNode[0][0]
    outVRPData = pd.DataFrame(index=[], columns=["routeLength", "tripRatio", "VRPcustomers", "VRPlengths", "vehicles"])
    outdemand = pd.DataFrame(index=[], columns=["busRouteLength", "ratio", "pid", "from", "to"])
    for ratio in range(minRatio, maxRatio + ratioSpan, ratioSpan):
        # バス路線無しver
        demand, numCustomers, lengths, vehicles = solveVRP_numVehicles(N, D, None, K, mainhub, tripUnit, distTripMin,
                                                                       hour, Q, dLimit, 0, ratio, F)
        series = pd.Series([0, ratio, numCustomers, lengths, vehicles], index=outVRPData.columns)
        outdemand = outdemand.append(demand)
        outVRPData = outVRPData.append(series, ignore_index=True)

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
