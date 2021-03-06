from gurobipy import *
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx
import math
import select_hub as hub
import select_routes_from_edge as route
import select_route_combination as combination

EPS = 1.e-6
semiMajorAxis = 6378137.0  # 赤道半径
flattening = 1 / 298.257223563  # 扁平率
e_2 = flattening * (2 - flattening)
degree = math.pi / 180


def distance_points(point1, point2):
    # euclid distance
    return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)*1000

def distance_point_latlon(point1, point2):
    x1, y1, x2, y2=point1[0], point1[1],point2[0], point2[1]
    coslat = math.cos((y1 + y2) / 2 * degree)
    w2 = 1 / (1 - e_2 * (1 - coslat * coslat))
    dx = (x1 - x2) * coslat
    dy = (y1 - y2) * w2 * (1 - e_2)
    return math.sqrt((dx * dx + dy * dy) * w2) * semiMajorAxis * degree


def distance(id1, id2, nodes, is_latlon=True):
    if is_latlon:
        return distance_point_latlon(nodes[id1], nodes[id2])

    return distance_points(nodes[id1], nodes[id2])


def load_data(nodeData, edgeData, distanceMatrixData, flowData):
    # 基準データを読み込み
    N, E, E_id, D, F = {}, {}, {}, {},{}

    df_node = pd.read_csv(nodeData)
    df_edge = pd.read_csv(edgeData)
    df_flow = pd.read_csv(flowData)
    df_distance = pd.read_csv(distanceMatrixData)

    # fid,lon,lat,name,flow_from,frow_to
    for v in df_node.values:
        N[v[0]] = (v[2], v[3], v[4])

    # edge_id, source,target,weight,path
    for v in df_edge.values:
        E[(v[1], v[2])] = v[3]
        E_id[(v[1], v[2])] = v[0]

    # from,to,value,origin_lon,origin_lat,destination_lon,destination_lat
    for v in df_flow.values:
        F[(v[0], v[1])] = v[2]

    for v in df_distance.values:
        D[(v[0], v[1])] = v[2]

    G = nx.from_pandas_edgelist(df_edge, 'source', 'target', 'weight')

    return N, E, D, F, G, E_id




def create_nodes(num, r=1, seed=32):
    np.random.seed(seed=seed)
    points = {}
    # 最初の点は原点
    points[0] = [0, 0]
    for i in range(num-1):
        x, y = np.random.rand()*2*r - r, np.random.rand()*2*r - r
        points[i+1] = [x, y]
    return points

def create_distance_matrix(points:dict):
    matrix = {}
    for i in points.keys():
        for j in points.keys():
            if i != j:
                matrix[(i, j)] = distance(i, j, points)
            else:
                matrix[(i, j)] = 0
    return matrix

def create_od_flow(points:dict, pattern="many2one", h=0):
    od_flow = {}
    if pattern == "many2one": # 最初の点が原点なので，そこに集中するように
        for i in points.keys():
            for j in points.keys():
                if i != j:
                    if i == h:
                        od_flow[(i, j)] = 1
                    else:
                        od_flow[(i, j)] = 0
                else:
                    od_flow[(i, j)] = 0

    elif pattern == "many2many":
        for i in points.keys():
            for j in points.keys():
                if i != j:
                    od_flow[(i, j)] = 1
                else:
                    od_flow[(i, j)] = 0

    return od_flow

def outIIS(model):
    model.computeIIS()
    model.write("outputIISLog.ilp")


if __name__ == '__main__':

    base_dir = "../data/moriya"
    N, E, D, F, G, E_id = load_data(f"{base_dir}/node.csv",
                              f"{base_dir}/edge.csv",
                              f"{base_dir}/distance_matrix.csv",
                              f"{base_dir}/flow.csv")

    p = 1
    q = 4

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

    hubs = list(set(hubs))
    print("---------------------------↓2.create_Route↓-------------------------")
    minLength = 2500  # 最小路線長
    maxLength = 15000
    span = 2500
    alfa = 1
    max_detour_ratio = 1.5
    ratio_span=0.05


    # 路線パターン作成
    listTerminal = hub.createTerminalCombinationNoLoop(hubs, D, minLength)
    output_list=f"{base_dir}/route_list.csv"
    routes_condition, routes_node, routes_edge, routes_length, routes_objVal, routes_CalcTime = route.createRouteList(N, E, D, F, G, E_id,
                                                                                               listTerminal, minLength, output_list,maxLength, span, max_detour_ratio, ratio_span)

