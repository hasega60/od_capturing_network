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

def calcBestRouteCombination(N, E, routes_node, routes_length, total_length, mainNode, F=None):
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



def calcRouteCombination_useFeeder(N, E, D, F, routes_node, routes_length, alfa=1):
    model_brc = Model("routeCombination_feeder")
    R = routes_node.keys()

    A, x, y1, y2, z, u = {}, {}, {}, {}, {}, {}
    N_count=len(N)

    for r in R:
        # 路線選択の変数
        x[r] = model_brc.addVar(vtype="B", name=f"x({r})")
        for i in N:
            # A_irは事前に作成
            if i in routes_node[r]:
                A[i, r] = 1
            else:
                A[i, r] = 0

    for i in N:
        # バス路線があるノード
        z[i] = model_brc.addVar(vtype="B", name="z(%s)" % i)
        for j in N:
            if i != j:
                # フィーダー路線選択ノード iが路線でない
                y1[i, j] = model_brc.addVar(vtype="B", name=f"y1({i}, {j})")
                # jが路線でない
                y2[i, j] = model_brc.addVar(vtype="B", name=f"y2({i}, {j})")
                # i_jが路線で移動可能かどうか
                u[i, j] = model_brc.addVar(vtype="B", name=f"u({i}, {j})")


    model_brc.update()
    # 路線補足
    for i in N:
        model_brc.addConstr(quicksum(A[i, r] * x[r] for r in R) >= z[i])

    # 路線同士で結ばれた地点は移動可能
    for i in N:
        for j in N:
            if i != j:
                model_brc.addConstr(u[i, j] >= z[i] + z[j] - 1)
                model_brc.addConstr(u[i, j] <= z[i])
                model_brc.addConstr(u[i, j] <= z[j])

    # 片方だけ路線で結ばれた地点はフィーダー移動可能
    for i in N:
        for j in N:
            if i != j:
                model_brc.addConstr(y1[i, j] >= z[i])
                model_brc.addConstr(y1[i, j] <= z[j])

                model_brc.addConstr(y2[i, j] <= z[i])
                model_brc.addConstr(y2[i, j] >= z[j])

                model_brc.addConstr(y1[i, j] <= u[i, j])
                model_brc.addConstr(y2[i, j] <= u[i, j])


    # すべてのノードはyかuで捕捉される

    #model_brc.addConstr(quicksum(y1[i, j] + y2[i, j] for i in N for j in N if i != j) >= N_count)

    # 最低でも路線を一つ選ぶ

    model_brc.addConstr(quicksum(x[r] for r in R) >= 1)

    # 目的関数　距離

    model_brc.setObjective(quicksum(routes_length[r] * x[r] for r in R)
                           + alfa * (quicksum(F[(i, k)]*D[(i, j)]*y1[i, j] + F[(i, k)]*D[(j, k)]*y2[j, k]
                                              for i in N for j in N for k in N if i != k and i != j and j != k)), GRB.MINIMIZE)

    model_brc.update()
    model_brc.optimize()
    model_brc.__data = x, y1, y2, u
    # selectNode = [j for j in y if y[j].X > EPS]
    selectRoute = [j for j in x if x[j].X > EPS]
    selectNode = []
    for r in selectRoute:
        for n in routes_node[r]:
            selectNode.append(n)
    length = 0
    selectHubs = []
    for r in selectRoute:
        length += routes_length[r]
        lst = r.split("_")
        selectHubs.append((int(lst[0]), int(lst[1])))

    print("case:" + str(alfa) + " hubs:" + str(selectHubs) + " node:" + str(selectNode) + "_route:" + str(
        selectRoute) + "_length:" + str(length))
    return selectNode, selectRoute, length, model_brc, selectHubs



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

    alfa_list = [0.1,1, 2, 3]
    beta_ratio = 4

    # フローを標準化
    f_max=max(F.values())
    f_min=min(F.values())
    F_n={}
    for k, v in F.items():
        F_n[k]=(v-f_min)/(f_max-f_min)



    for alfa in alfa_list:
        selectNodes, id, routeLength, model_b, selectHubs = calcRouteCombination_useFeeder2(N, E, D, F_n,routes_node,
                                                                                            routes_length,
                                                                                            alfa, alfa*beta_ratio)
        continue

        if id is not None:
            totalLengthList[count] = alfa
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


    exit()
    maxTotalLength = 30000  # 最大路線長
    minTotalLength = 5000  # 最小路線長
    totalLengthSpan = 1000  # 路線候補を作る間隔

    for length in range(minTotalLength, maxTotalLength + totalLengthSpan, totalLengthSpan):
        """        
        selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_condition,
                                                                                     routes_node, routes_edge,
                                                                                     routes_length, length, mainNode, F)
        """
        #selectNodes, id, routeLength, model_b, selectHubs = calcBestRouteCombination(N, E, routes_node, routes_length, length, mainNode, F)

        selectNodes, id, routeLength, model_b, selectHubs = calcRouteCombination_useFeeder2(N, E, D, F,routes_node,routes_length,alfa, beta)

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

