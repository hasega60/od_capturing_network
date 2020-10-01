from gurobipy import *
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx




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


if __name__ == '__main__':

    base_dir = "../data/kawagoe_example"
    N, E, D, F, G, E_id = load_data(f"{base_dir}/node.csv",
                              f"{base_dir}/edge.csv",
                              f"{base_dir}/distance_matrix.csv",
                              f"{base_dir}/flow.csv")

    df_node_selected = pd.read_csv(f"{base_dir}/node_selected.csv")
    df_node_selected_1 = df_node_selected[df_node_selected["8000"] > 0]
    df_node_selected_2= df_node_selected[df_node_selected["18000"] > 0]
    out=[]
    for n in df_node_selected.values:
        nearest_1, nearest_2=None,None
        if n[7] == 0:
            length = 99999

            for v in df_node_selected_1.values:
                d = nx.shortest_path_length(G, n[0], v[0], weight='weight')
                if d < length:
                    length = d
                    nearest_1 = v[0]
        if n[8] == 0:
            for v in df_node_selected_2.values:
                d = nx.shortest_path_length(G, n[0], v[0], weight='weight')
                if d < length:
                    length = d
                    nearest_2 = v[0]

        n = np.append(n, nearest_1)
        n = np.append(n, nearest_2)
        out.append(n)

    cols = list(df_node_selected.columns)
    cols.append("n_8000")
    cols.append("n_18000")
    df_out = pd.DataFrame(out, columns=cols)

    df_out.to_csv(f"{base_dir}/node_selected_v.csv", index=False)
