from gurobipy import *
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx
import copy
import select_hub

EPS = 1.e-6

setParam("TimeLimit", 360)
setParam("MIPFocus", 1)

def outIIS(model):
    model.computeIIS()
    model.write("outputIISLog.ilp")

def createTerminalCombinationNoRoop(hubs, D, minLength):
    list = []
    for hub1 in hubs:
        for hub2 in hubs:
            if list not in [(hub1, hub2)]:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 != hub2 and D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list


def edgeConvertnodes(edges):
    GA = nx.Graph()
    GA.add_edges_from(edges)
    # 連結成分分解
    return list(list(nx.connected_components(GA))[0])


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


def updateFlow(F, selectN):
    FS = copy.deepcopy(F)
    for (i, j) in FS.keys():
        if i in selectN[0] and j in selectN[0]:
            # 選択ノードに含まれていたら重み0にする
            FS[(i, j)] = 0
    return FS


def updateNodeWeight(N, selectN):
    NS = copy.deepcopy(N)
    for n in NS.keys():
        if n in selectN[0]:
            # 選択ノードに含まれていたら重み0にする
            NS[n] = (NS[n][0], NS[n][1], 0)
    return NS


def route_minDist_bulk(N, E, h, alfa, R, D, F):
    """
    一括で路線を作成する
    """

    bInit = False
    model_rmd = Model("route_minDist")
    x, y, z, u = {}, {}, {}, {}
    for r in R:
        for (i, j) in E:
            x[i, j, r] = model_rmd.addVar(vtype="B", name=f"x({i}, {j}, {r})")


    for i in N:
        u[i] = model_rmd.addVar(vtype="C", name=f"u({i})")
        z[i] = model_rmd.addVar(vtype="B", name=f"z({i})")
        #for r in R:
            #z[i, r] = model_rmd.addVar(vtype="B", name=f"z({i},{r})")
        for j in N:
            y[i, j] = model_rmd.addVar(vtype="B", name=f"y({i},{j})")

    model_rmd.update()

    # 順番制約
    for r in R:
        for v in N:
            for e in E:
                if v == e[0]:
                    if v == h:
                        model_rmd.addConstr(quicksum(x[i, j, r] for (i, j) in E if i == v) - quicksum(
                            x[j, i, r] for (j, i) in E if i == v) == 1)

                    else:
                        model_rmd.addConstr(quicksum(x[i, j, r] for (i, j) in E if i == v) - quicksum(
                            x[j, i, r] for (j, i) in E if i == v) <= 0)

    # xとzの関係 zはiに路線があるか
    for i in N:
        model_rmd.addConstr(quicksum(x[i, j, r] for r in R for j in N if (i, j) in E) >= z[i])
        model_rmd.addConstr(quicksum(x[j, i, r] for r in R for j in N if (i, j) in E) >= z[i])

    # yとzの関係　z_ijの反転がy_ij
    for i in N:
        for j in N:
            if i != j:
                #model_rmd.addConstr(quicksum(z[i, r] for r in R) * quicksum(z[j, r] for r in R) <= y[i, j])
                model_rmd.addConstr(z[i] * z[j] <= y[i, j])

    # 路線数
    model_rmd.addConstr(quicksum(x[h, j, r] for r in R for j in N if (h, j) in E) == len(R))

    # フロー捕捉量
    model_rmd.addConstr(quicksum(z[i] * z[j] * F[(i, j)] for i in N for j in N) +
                        quicksum(y[i, j]* F[(i, j)] for i in N for j in N) == quicksum(F[(i, j)] for i in N for j in N))


    for i in N:
        model_rmd.addConstr(u[i] >= 0)
    # 部分巡回路制約
    countN = len(N)
    for r in R:
        for (i, j) in E:
            if i != h and j != h:
                model_rmd.addConstr(u[i] - u[j] + countN * x[i, j, r] <= countN - 1)

    model_rmd.setObjective(quicksum(x[i, j, r] * D[(i, j)] for r in R for (i, j) in E)
                           + alfa * quicksum(y[i, j] * D[(i, j)] * F[(i, j)] for i in N for j in N), GRB.MINIMIZE)

    model_rmd.update()
    model_rmd.optimize()
    #model_rmf.write(mstFile_path)  # 解を出力
    is_timelimit=False
    if model_rmd.Status == GRB.INFEASIBLE:
        return None, None, None, None, None, is_timelimit

    model_rmd.__data = x, y
    select_routeEdge = [j for j in x if x[j].X ==1]
    select_directNodes = [j for j in y if y[j].X ==1]

    route_lengths = []
    direct_length = 0
    for r in R:
        length = 0
        for e in select_routeEdge:
            if e[2] == r:
                length += E[e[0], e[1]]
        route_lengths.append([r, length])

    for n in select_directNodes:
        direct_length += D[n]*F[n]

    # timelimtに到達
    if model_rmd.Status == GRB.TIME_LIMIT:
        is_timelimit=True


    return select_routeEdge, route_lengths, select_directNodes, direct_length, model_rmd, is_timelimit



def createRoute_loop_Flow(N, D, F, s, maxdist):
    model_loop = Model("route_loop")
    # model_loop.setParam("MIPFocus", 2)  # 上界を下げる形で実装

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
        model_loop.setParam("TimeLimit", 600)
        for k, v in sorted(nodes.items(), key=lambda x: x[1][2], reverse=True):
            if con <= maxNodeSize_createRoute_roop_Flow:
                con += 1
            else:
                nodes.pop(k)
    """
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

    model_loop.optimize()
    #model_loop.write(mstFile_roop)  # 解を出力
    model_loop.__data = x, y
    selectN = [j for j in y if y[j].X > EPS]
    selectE = [j for j in x if x[j].X > EPS]

    routeLength = 0
    for (i, j) in selectE:
        routeLength += D[(i, j)]

    return selectN, selectE, routeLength, model_loop





def createRouteList(N, E, D, F, G, E_id, listTerminal, minLength, output_path, maxLength=30000,
                    span=5000):
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime = {}, {}, {}, {}, {}, {}
    count = 0
    print("createRoute:" + str(listTerminal))
    for hub in listTerminal:
        shortestLength = 0
        model = None

        if hub[0] == hub[1]:
            for length in range(minLength, maxLength, span):
                print("---------------------------↓" + "solve:" + str(hub[0]) + ", " + str(hub[1]) + "," + str(
                    length) + "↓-------------------------")

                selectN, selectE, routeLength, model = createRoute_loop_Flow(N, D, F, hub[0], length)

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

        else:
            # ハブ間路線
            # 最短経路
            shortestLength = nx.dijkstra_path_length(G, hub[0], hub[1])
            weightShortestPath = 0
            sPath = nx.dijkstra_path(G, hub[0], hub[1])
            for i in range(len(sPath)-1):
                weightShortestPath += F[(sPath[i], sPath[i+1])]

            print("shortestLength_(" + str(hub[0]) + ", " + str(hub[1]) + "):" + str(shortestLength))
            routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str("spath")

            routes_node[count] = sPath
            select_edge_id = []
            for n in range(len(sPath)-1):
                select_edge_id.append(E_id[(sPath[n], sPath[n+1])])

            routes_edge[count] = select_edge_id
            routes_length[count] = shortestLength
            routes_objVal[count] = weightShortestPath
            routes_CalcTime[count] = 0
            count += 1
            timelimit=False

            # hub間路線網パターン作成

            for length in range(minLength, maxLength, span):
                #timelimitだったら次の組み合わせへ
                if timelimit:
                    continue

                model = None
                # 最大路線長を超えないパターンを作成
                if shortestLength < length:
                    print("solve:" + str(hub[0]) + ", " + str(hub[1]) + "," + str(length))
                    selectE, routeLength, model = createRoute_maxWeight(N, E, hub[0], hub[1], length, F)
                    routes_condition[count] = str(hub[0]) + "_" + str(hub[1]) + "_" + str(
                        length) + "_" + str(
                        "F")
                    if selectE is not None:
                        selectN = edgeConvertnodes(selectE)
                        select_edge_id=[]
                        for e in selectE:
                            select_edge_id.append(E_id[e])
                        routes_node[count] = selectN
                        routes_edge[count] = select_edge_id
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

                    outputRouteList(routes_condition, routes_node, routes_edge, routes_length, routes_objVal,
                                    routes_CalcTime, output_path)

                    if model.Status == GRB.TIME_LIMIT:
                        timelimit=True

                    """
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
                        """

    return routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime



def outputRouteList(routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime, output_path):
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

    outRouteList.to_csv(output_path, index=False)