# coding:utf-8
import copy
import datetime
import math
from ast import literal_eval
import networkx as NX
import pandas as pd
from gurobipy import *
from tqdm import tqdm
from inputimeout import inputimeout, TimeoutOccurred

EPS = 1.e-6
timeLimit = 1200
# 解の精度
setParam("MIPGap", 0.00)
setParam("TimeLimit", timeLimit)
setParam("MIPFocus", 1)

# パラメータ設定
basedir = 'C:/Projects/Gurobi/createRoute/tikusei/'
resultdir = 'result/'
debugdir = 'debug/'
nodefile = "nodes.csv"
edgefile = "edges.csv"
flowfile = "flows.csv"
dMatrixfile = "distanceMatrixbyNetwork.csv"

# モデルの解出力ファイル名
mstFile_loop = "sol_model_loop.sol"
mstFile_path = "sol_model_path.sol"

bWeight = True  # pmedianでノードの重みを考慮するか
numHubs = 7  # ハブの数
numMainHubs = 0  # ハブのうち，メインハブの数 残りはサブハブとする 0ならばp-medianのみ
maxMainHubs = 3

now = datetime.datetime.now()
summaryFileData = "hubs" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"
routeFileName = "routeList" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"
routeCombiFileName = "routeCombination" + str(now.strftime("%Y%m%d%H%M%S")) + ".csv"

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
    print("hubs:" + str(hubs))
    return hubs, select


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
        return None, None

    model2.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    select = [j for j in x if x[j].X > EPS]

    # pcenter用初期解作成
    model2.write("pmedian_ext_sol_out.mst")

    return hubs, select


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
    select = [j for j in x if x[j].X > EPS]
    return hubs, select


if __name__ == "__main__":
    import sys

    print("---------------------------↓Load_Data↓-------------------------")
    N, E, D, F, G, E_id = load_data(basedir + nodefile, basedir + edgefile, basedir + dMatrixfile, basedir + flowfile)

    print("node count=" + str(len(N)))
    print("edge count=" + str(len(E)))

    outSummaryData = pd.DataFrame(index=[],
                                  columns=["numHubs",
                                           "numMainHubs",
                                           "mainhubs",
                                           "subhubs",
                                           "hubs",
                                           "distanceToHubs",
                                           "distanceToHubs_Max",
                                           "distanceToHubs_weighted"])

    print("---------------------------↓1.create_Hub↓-------------------------")
    # hubの組合せを作成
    # ハブの数を選択

    maxHubs = 15
    for m in range(7, maxHubs + 1):
        hubs = []
        numMainHubs = m
        numHubs = m
        hubs, select = pmedian(N, D, numHubs, bWeight)
        dist = 0
        wdist = 0
        distMax = 0
        for s in select:
            dist += D[s]
            wdist += D[s] * N[s[0]][2]
            if distMax < D[s]:
                distMax = D[s]

        series = pd.Series(
            [numHubs, numMainHubs, hubs, [], hubs, dist, distMax, wdist],
            index=outSummaryData.columns)
        outSummaryData = outSummaryData.append(series,
                                               ignore_index=True)

        if numHubs > 1:

            for n in range(1, maxMainHubs):
                print("case:" + str(n) + "_" + str(numHubs - n))
                numMainHubs = n
                numSubHubs = numHubs - numMainHubs
                if n >= numHubs or numMainHubs > numSubHubs:
                    break

                mainHubs, subHubs = [], []
                mainHubs, select1 = pmedian(N, D, numMainHubs, bWeight)
                # subHubs, select2 = pcenter_existing(N, D, mainHubs, numSubHubs, bWeight)
                subHubs, select2 = pmedian_existing(N, D, mainHubs, numSubHubs, bWeight)
                dist = 0
                wdist = 0
                distMax = 0
                if subHubs is not None:
                    for i in range(len(select1)):
                        s1 = select1[i]
                        s2 = select2[i]
                        wdist1 = D[s1] * N[s1[0]][2]
                        wdist2 = D[s2] * N[s1[0]][2]
                        if wdist1 < wdist2:
                            dist += D[s1]
                            wdist += D[s1] * N[s1[0]][2]
                            if distMax < D[s1]:
                                distMax = D[s1]
                        else:
                            dist += D[s2]
                            wdist += D[s2] * N[s2[0]][2]
                            if distMax < D[s2]:
                                distMax = D[s2]

                    hubs = []
                    for h in mainHubs:
                        hubs.append(h)
                    for h in subHubs:
                        hubs.append(h)
                    series = pd.Series(
                        [numHubs, numMainHubs, mainHubs, subHubs, hubs, dist, distMax, wdist],
                        index=outSummaryData.columns)
                    outSummaryData = outSummaryData.append(series,
                                                           ignore_index=True)

        outSummaryData.to_csv(basedir + resultdir + summaryFileData, index=False)
