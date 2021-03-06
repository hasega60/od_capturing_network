from gurobipy import *
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import select_hub as hub
import select_feeder_depot as feeder_depot

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

def calcBestRouteCombination(N, routes_node, routes_length, total_length, F, mainNode=None):
    model_brc = Model("bestRouteCombination")
    R = routes_node.keys()

    A, y, z, u, x = {}, {}, {}, {}, {}

    # x_irは事前に作成
    for r in R:
        for i in N:
            if f"{i}" in routes_node[r]:
                A[i, r] = 1
            else:
                A[i, r] = 0

    for i in N:
        y[i] = model_brc.addVar(vtype="B", name="y(%s)" % i)
        for j in N:
            if i != j:
                # i_jが路線で移動可能かどうか
                z[i, j] = model_brc.addVar(vtype="B", name="z(%s, %s)" % (i,j))

    for r in R:
        u[r] = model_brc.addVar(vtype="B", name="u(%s)" % r)

    model_brc.update()
    for i in N:
        model_brc.addConstr(quicksum(A[i, r] * u[r] for r in R) >= y[i])
        model_brc.addConstr(y[i] <= 1)

    for r in R:
        # 路線長制約
        model_brc.addConstr(sum(routes_length[r] * u[r] for r in R) <= total_length)

    for i in N:
        for j in N:
            if i != j:
                # どちらも路線でカヴァーできるなら路線利用
                model_brc.addConstr(z[i, j] >= y[i] + y[j] - 1)
                model_brc.addConstr(z[i, j] <= y[i])
                model_brc.addConstr(z[i, j] <= y[j])

    if mainNode is not None:
        H = mainNode
        # mainNodeが選ばれない路線は選ばない
        for r in R:
            model_brc.addConstr(quicksum(A[h, r] for h in H) >= u[r])

    # 目的関数　ノードの重み取得最大化
    model_brc.setObjective(quicksum(z[i, j] * F[(i, j)] for i in N for j in N if i != j), GRB.MAXIMIZE)
    #model_brc.setObjective(quicksum(y[i] * y[j] * F[(i, j)] for i in N for j in N if i != j), GRB.MAXIMIZE)
    model_brc.update()
    model_brc.optimize()
    model_brc.__data = y, u
    selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in u if u[j].X > EPS]
    captured_flow = [j for j in z if z[j].X > EPS]

    selectNode = []
    for r in selectRoute:
        if isinstance(routes_node[r], str):
            routes_node[r] = routes_node[r].replace('[', '').replace(']', '').split(",")
        selectNode.extend(routes_node[r])
    length = 0
    selectHubs = []
    for r in selectRoute:
        length += routes_length[r]
        lst = r.split("_")
        selectHubs.append((lst[0], lst[1]))

    flow = 0
    for i in N:
        for j in N:
            if (i, j) in captured_flow:
                flow += F[(i, j)]


    print("case:" + str(total_length) +"flow:"+str(flow) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs, flow


# 直達，フィーダー，路線の組み合わせ　実装完了せず
def calcRouteCombination_useFeeder2(N, E, D, F, routes_node, routes_length, alfa=2, beta=4):
    model_brc = Model("routeCombination_feeder")
    R = routes_node.keys()

    A, x, y, z, u, v = {}, {}, {}, {}, {}, {}
    N_count=len(N)

    for r in R:
        # 路線選択の変数
        x[r] = model_brc.addVar(vtype="B", name=f"x({r})")
        for i in N:
            # 路線rでカヴァーされるノードi　A_irは事前に作成
            if i in routes_node[r]:
                A[i, r] = 1
            else:
                A[i, r] = 0

    for i in N:
        # バス路線があるノード
        z[i] = model_brc.addVar(vtype="B", name="z(%s)" % i)
        for j in N:
            if i != j:
                # フィーダー路線選択ノード
                y[i, j] = model_brc.addVar(vtype="B", name=f"y({i}, {j})")

                # i_jが路線で移動可能かどうか
                u[i, j] = model_brc.addVar(vtype="B", name=f"u({i}, {j})")

                # i_jを直接移動する必要があるか
                v[i, j] = model_brc.addVar(vtype="B", name=f"v({i}, {j})")


    model_brc.update()
    # 路線によるノードの捕捉
    for i in N:
        model_brc.addConstr(quicksum(A[i, r] * x[r] for r in R) >= z[i])
        model_brc.addConstr(z[i] <= 1)

    for i in N:
        for j in N:
            if i != j:
                # どちらも路線でカヴァーできるなら路線利用
                model_brc.addConstr(u[i, j] >= z[i] + z[j] - 1)
                model_brc.addConstr(u[i, j] <= z[i])
                model_brc.addConstr(u[i, j] <= z[j])
                # 最低でも路線を一つ選ぶ
                model_brc.addConstr(v[i, j] >= 1 -(z[i] + z[j]))
                #model_brc.addConstr(v[i, j] <= z[i])
                #model_brc.addConstr(v[i, j] <= z[j])

    for i in N:
        for j in N:
            if i != j:
                # それ以外はフィーダー移動
                model_brc.addConstr(v[i, j] + u[i, j] + y[i, j] == 1)

    # 最低でも路線を一つ選ぶ
    model_brc.addConstr(quicksum(x[r] for r in R) >= 1)

    # 目的関数　距離
    model_brc.setObjective(quicksum(routes_length[r] * x[r] for r in R)
                           + (quicksum(F[(i, j)]* alfa * D[(i, j)]*y[i, j] for i in N for j in N if i != j))
                           + (quicksum(F[(i, j)]* beta * D[(i, j)]*v[i, j] for i in N for j in N if i != j))
                           , GRB.MINIMIZE)

    model_brc.update()
    model_brc.optimize()
    #debug
    #model_brc.computeIIS()
    #model_brc.write("debug.ilp")
    model_brc.__data = x, y, u, v
    selectRoute = [j for j in x if x[j].X > EPS]
    selectfeeder = [j for j in y if y[j].X > EPS]
    selectdirect= [j for j in v if v[j].X > EPS]
    selectNode = []
    for r in selectRoute:
        li = list(routes_node[r].split("-"))
        for n in li:
            selectNode.append(n)
    length = 0
    selectHubs = []
    """
    for r in selectRoute:
        length += routes_length[r]
        lst = r.split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))
"""
    print("case:" + str(alfa) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs



def outputRouteCombination(totalLength, routes_id, routes_hubs, routes_node, routes_length, routes_objVal,
                           routes_CalcTime,feeder_hubs, feeder_hub_count, select_feeder_hubs,
                           total_feeder_distance, flow_feeders, depot_capacities, max_capacities, avg_capacities,
                           output_path, output_path_summary, output_path_node, output_path_feeder_flow):
    outRouteList = pd.DataFrame(index=[],
                                columns=["condition",
                                         "routeID",
                                         "hubs",
                                         "nodes",
                                         "routeLength",
                                         "objVal", "calcTime", "feeder_hub", "feeder_hub_count",
                                         "select_feeder_hub", "flow_feeder","total_feeder_distance","depot_capacities",
                                         "max_capacities", "avg_capacities"])

    outRouteList_summary = pd.DataFrame(index=[],
                                columns=["condition", "flow","routes_length","feeder_hub_count","flow_feeder",
                                         "total_feeder_distance","max_capacities", "avg_capacities"])

    for key, routeId in routes_id.items():
        series = pd.Series(
            [totalLength[key], routeId, routes_hubs[key], routes_node[key], routes_length[key], routes_objVal[key],
             routes_CalcTime[key],feeder_hubs[key],feeder_hub_count[key], select_feeder_hubs[key],flow_feeders[key],
             total_feeder_distance[key], depot_capacities[key],
             max_capacities[key], avg_capacities[key]],
            index=outRouteList.columns)

        series_summary=pd.Series(
            [totalLength[key], routes_objVal[key],routes_length[key], feeder_hub_count[key],flow_feeders[key],
             total_feeder_distance[key],max_capacities[key], avg_capacities[key]],
            index=outRouteList_summary.columns)

        outRouteList = outRouteList.append(series,
                                           ignore_index=True)
        outRouteList_summary = outRouteList_summary.append(series_summary,
                                           ignore_index=True)

    outRouteList.to_csv(output_path, index=False)
    outRouteList_summary.to_csv(output_path_summary, index=False)

    # 選択されたノード，エッジ，フローのリストを出力する
    outRouteList_export = outRouteList[["condition", "routeID", "nodes", "select_feeder_hub"]]
    out_node, out_feeder = [], []
    for v in outRouteList_export.values:
        condition = v[0]
        for n in v[2]:
            out_node.append([condition, n])
        for f in v[3]:
            out_feeder.append([condition, f[0], f[1]])

    outRouteNodes=pd.DataFrame(out_node, columns=["condition", "node"])
    outFeederFlow=pd.DataFrame(out_feeder, columns=["condition", "from_node", "hub_node"])

    outRouteNodes.to_csv(output_path_node, index=False)
    outFeederFlow.to_csv(output_path_feeder_flow, index=False)





if __name__ == '__main__':
    totalLengthList, routes_hubs_b, routes_id_b, routes_node_b, \
    routes_length_b, routes_objVal_b, routes_CalcTime_b = {}, {}, {}, {}, {}, {}, {}
    feeder_hubs,feeder_hub_count , select_feeder_hubs, \
    total_feeder_distance, depot_capacities, max_capacities, avg_capacities, flow_feeders = {}, {}, {}, {}, {}, {}, {}, {}
    count = 0
    base_dir = "../data/moriya"
    N, E, D, F, G, routes_node, routes_length = load_data(f"{base_dir}/node.csv",
                              f"{base_dir}/edge.csv",
                              f"{base_dir}/distance_matrix.csv",
                              f"{base_dir}/flow.csv", f"{base_dir}/route_list.csv")

    p = 1
    q = 7

    # string to list
    for r in routes_node.keys():
        if isinstance(routes_node[r], str):
            routes_node[r] = routes_node[r].replace('[', '').replace(']', '').replace(' ', '').split(",")

    print("---------------------------↓1.create_Hub↓-------------------------")
    main_hub = hub.pmedian(N, D, p)
    sub_hub_c, _ = hub.pcenter_existing(N, D, main_hub, q)
    sub_hub_m, _ = hub.pmedian_existing(N, D, main_hub, q)
    hubs = main_hub.copy()
    for h in sub_hub_c:
        hubs.append(h)
    for h in sub_hub_m:
        hubs.append(h)

    print("main_hub:" + str(main_hub))
    print("sub_hub_center:" + str(sub_hub_c))
    print("sub_hub_median:" + str(sub_hub_m))

    print("---------------------------↓3.route_Combination and feeder_port↓-------------------------")
    maxTotalLength = 50000  # 最大路線長
    minTotalLength = 5000  # 最小路線長
    totalLengthSpan = 1000  # 路線候補を作る間隔
    distance_limit = 5000 # feederまでのアクセス距離
    max_port_capacity = 99999 # feederポートの容量
    max_port_count = 5 #feederポート数

    for length in range(minTotalLength, maxTotalLength + totalLengthSpan, totalLengthSpan):
        selectNode, selectRoute, total_length, model, selectHubs, flow = calcBestRouteCombination(N, routes_node, routes_length, length, F, main_hub)
        selectNode_int=[]
        for i in selectNode:
            selectNode_int.append(int(i))

        hubs, select_node_hubs, flow_feeder, flow_dist, depot_capacity, max_capacity, avg_capacity = feeder_depot.select_feeder_hub_max_flow(N, D, F, selectNode_int,
                                                                                                                                             max_port_capacity=max_port_capacity,
                                                                                                                                             max_port_count=max_port_count,
                                                                                                                                             distance_limit=distance_limit)

        if id is not None:
            totalLengthList[count] = length
            routes_id_b[count] = selectRoute
            routes_hubs_b[count] = selectHubs
            routes_node_b[count] = selectNode
            routes_length_b[count] = total_length
            routes_objVal_b[count] = flow
            routes_CalcTime_b[count] = model.Runtime
            feeder_hubs[count] = hubs
            feeder_hub_count[count] = len(hubs)
            select_feeder_hubs[count] = select_node_hubs
            flow_feeders[count] = flow_feeder
            total_feeder_distance[count] = flow_dist
            depot_capacities[count] = depot_capacity
            max_capacities[count] = max_capacity
            avg_capacities[count] = avg_capacity
            count += 1

    outputRouteCombination(totalLengthList, routes_id_b, routes_hubs_b, routes_node_b, routes_length_b,
                           routes_objVal_b, routes_CalcTime_b, feeder_hubs, feeder_hub_count,select_feeder_hubs,
                           total_feeder_distance, flow_feeders, depot_capacities, max_capacities, avg_capacities,
                           f"{base_dir}/route_combination.csv", f"{base_dir}/route_combination_summary.csv",
                           f"{base_dir}/route_node.csv", f"{base_dir}/feeder_hub_select.csv"
                           )

    #for alfa in alfa_list:
    #    selectNode, selectRoute, length, model_brc, selectHubs = calcRouteCombination_useFeeder(N, E, D, F, routes_node, routes_length, mainNode, alfa)

