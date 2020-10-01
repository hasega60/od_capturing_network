from gurobipy import *
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
EPS = 1.e-6

def load_data(nodeData, edgeData, distanceMatrixData, flowData, routeData):
    # 基準データを読み込み
    N, E, D, F, routes_node, routes_length = {}, {}, {}, {}, {}, {}

    df_node = pd.read_csv(nodeData)
    df_edge = pd.read_csv(edgeData)
    df_flow = pd.read_csv(flowData)
    df_distance = pd.read_csv(distanceMatrixData)
    df_route = pd.read_csv(routeData)

    # fid,lon,lat,name,flow_from,frow_to
    for v in df_node.values:
        N[v[0]] = (v[2], v[3], v[4])

    # source,target,weight,path
    for v in df_edge.values:
        E[(v[0], v[1])] = v[2]

    # from,to,value,origin_lon,origin_lat,destination_lon,destination_lat
    for v in df_flow.values:
        F[(v[0], v[1])] = v[2]

    for v in df_distance.values:
        D[(v[0], v[1])] = v[2]

    for v in df_route.values:
        routes_node[v[0]] = v[1]
        routes_length[v[0]] = v[3]

    G = nx.from_pandas_edgelist(df_edge, 'source', 'target', 'weight')

    return N, E, D, F, G, routes_node, routes_length

def calcBestRouteCombination(N, E, routes_node, routes_length, total_length, mainNode,
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
        H = mainNode
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
        lst = r.split("_")
        selectHubs.append((lst[0], lst[1]))

    print("case:" + str(total_length) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs



def calcRouteCombination_useFeeder(N, E, D, F, routes_node, routes_length, mainNode, alfa):
    model_brc = Model("routeCombination_feeder")
    R = routes_node.keys()

    x, y, z, u, v = {}, {}, {}, {}, {}

    # x_irは事前に作成
    for r in R:
        for i in N:
            # if i in routes_node[r][0]:
            if i in routes_node[r]:
                x[i, r] = 1
            else:
                x[i, r] = 0

    for i in N:
        # バス路線があるノード
        v[i] = model_brc.addVar(vtype="B", name="v(%s)" % i)
        for j in N:
            if i != j:
                # バス路線がないノード to あるノード
                y[i, j] = model_brc.addVar(vtype="B", name=f"y({i}, {j})")

                # バス路線があるノード　to ないノード
                z[i, j] = model_brc.addVar(vtype="B", name=f"z({i}, {j})")

    for r in R:
        u[r] = model_brc.addVar(vtype="B", name="u(%s)" % r)

    model_brc.update()
    for i in N:
        model_brc.addConstr(quicksum(x[i, r] * u[r] for r in R) >= v[i])
        # model_brc.addConstr(v[i] <= 1)

        # 起点・終点にバス路線がないとき
        model_brc.addConstr(quicksum(y[i, k] for k in N if i != k) <= v[i])
        model_brc.addConstr(quicksum(z[k, i] for k in N if i != k) <= v[i])

    for k in N:
        # kはバス路線に選択されているノード
        model_brc.addConstr(quicksum(y[i, k] for i in N if i != k) >= v[k])
        model_brc.addConstr(quicksum(z[k, i] for i in N if i != k) >= v[k])

    # フロー捕捉制約 バス直接と+feederの捕捉量は同じ
    model_brc.addConstr(quicksum(v[i] * v[j] * F[(i, j)] for i in N for j in N if i != j) +
                        quicksum(F[(i, j)] * y[i, k] + F[(i, j)] * z[l, j] - F[(i, j)] * y[i, k] * z[l, j] for i in N
                                 for k in N for l in N for j in N if i != j and i != k and k != l and l != j)
                        == quicksum(F[(i, j)] for i in N for j in N if i != j)
                        )

    if mainNode is not None:
        H = mainNode
        # mainNodeが選ばれない路線は選ばない
        for r in R:
            model_brc.addConstr(quicksum(x[h, r] for h in H) >= u[r])

    # 目的関数　ノードの重み取得最大化
    model_brc.setObjective(quicksum(routes_length[r] * u[r] for r in R)
                           + alfa * (quicksum(D[(i, k)]* F[(i, j)] * y[i, k] for i in N for k in N for j in N if i != j and i != k and k != j)
                                     + quicksum(D[(l, j)]* F[(i, j)] * z[l, j] for i in N for l in N for j in N if i != j and i != l and l != j)), GRB.MINIMIZE)

    model_brc.setObjective(quicksum(v[i] * v[j] * F[(i, j)] for i in N for j in N if i != j), GRB.MAXIMIZE)

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
        lst = r.split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print("case:" + str(alfa) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs

def outputRouteCombination(totalLength, routes_id, routes_hubs, routes_node, routes_length, routes_objVal,
                           routes_CalcTime, output_path):
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

    outRouteList.to_csv(output_path, index=False)

if __name__ == '__main__':
    alfa_list = [1, 2, 5, 10]
    totalLengthList, routes_hubs_b, routes_id_b, routes_node_b, routes_length_b, routes_objVal_b, routes_CalcTime_b = {}, {}, {}, {}, {}, {}, {}
    count = 0
    #TODO mainnode
    mainNode = ["N063"]

    base_dir = "../data/kawagoe_example"
    N, E, D, F, G, routes_node, routes_length = load_data(f"{base_dir}/node.csv",
                              f"{base_dir}/edge.csv",
                              f"{base_dir}/distance_matrix.csv",
                              f"{base_dir}/flow.csv", f"{base_dir}/route_list.csv")

    print("---------------------------↓3.route_Combination↓-------------------------")
    maxTotalLength = 30000  # 最大路線長
    minTotalLength = 5000  # 最小路線長
    totalLengthSpan = 1000  # 路線候補を作る間隔

    for length in range(minTotalLength, maxTotalLength + totalLengthSpan, totalLengthSpan):
        """        
        selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_condition,
                                                                                     routes_node, routes_edge,
                                                                                     routes_length, length, mainNode, F)
        """
        selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_node, routes_length, length,
                                                                                     mainNode, F)

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
                                   routes_objVal_b, routes_CalcTime_b, f"{base_dir}/route_combination.csv")

    #for alfa in alfa_list:
    #    selectNode, selectRoute, length, model_brc, selectHubs = calcRouteCombination_useFeeder(N, E, D, F, routes_node, routes_length, mainNode, alfa)

