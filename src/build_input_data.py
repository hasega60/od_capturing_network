import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import math

semiMajorAxis = 6378137.0  #赤道半径
flattening = 1/298.257223563  # 扁平率
e_2 = flattening * (2 - flattening)
degree = math.pi / 180


def distance_latlon(p, q):
    x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
    coslat = math.cos((y1 + y2) / 2 * degree)
    w2 = 1 / (1 - e_2 * (1 - coslat * coslat))
    dx = (x1 - x2) * coslat
    dy = (y1 - y2) * w2 * (1 - e_2)
    return math.sqrt((dx * dx + dy * dy) * w2) * semiMajorAxis * degree



def create_edges(nodes, nodes_index, G):
    edges = {}
    tri = Delaunay(list(nodes.values()))
    for t in tri.simplices:
        t = np.sort(t)
        t_edge1 = (t[0], t[1])
        t_edge2 = (t[1], t[2])
        t_edge3 = (t[0], t[2])

        if t_edge1 not in edges:
            d = nx.shortest_path_length(G, nodes_index[t_edge1[0]], nodes_index[t_edge1[1]], weight='weight')
            edges[t_edge1] = d
            edges[(t_edge1[1], t_edge1[0])] = d
        if t_edge2 not in edges:
            d = nx.shortest_path_length(G, nodes_index[t_edge2[0]], nodes_index[t_edge2[1]], weight='weight')
            edges[t_edge2] = d
            edges[(t_edge2[1], t_edge2[0])] = d
        if t_edge3 not in edges:
            d = nx.shortest_path_length(G, nodes_index[t_edge3[0]], nodes_index[t_edge3[1]], weight='weight')
            edges[t_edge3] = d
            edges[(t_edge3[1], t_edge3[0])] = d

    return edges

if __name__ == '__main__':
    base_dir="../data/kawagoe_example"


    df_node = pd.read_csv(f"{base_dir}/node.csv")
    df_edge_all = pd.read_csv(f"{base_dir}/edge_all.csv")

    G = nx.from_pandas_edgelist(df_edge_all, 'source', 'target', 'weight')
    out, out_path = {}, {}
    for i in df_node["id"].values:
        for j in df_node["id"].values:
            if i == j:
                out[(i, j)] = 0
                out_path[(i, j)] = None
            else:
                out[(i, j)] = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                path=[]
                for p in nx.shortest_path(G, source=i, target=j, weight='weight'):
                    if p in df_node["id"].values:
                        path.append(p)

                out_path[(i, j)] = path


    df_out = pd.DataFrame(out.keys(), columns=["from", "to"])
    df_out["length"] = out.values()
    df_out["path"] = out_path.values()


df_out.to_csv(f"{base_dir}/distance_matrix.csv", index=False)

nodes ,nodes_index={}, {}
count=0
for v in df_node.values:
    nodes_index[count] = v[0]
    nodes[count]=(v[3], v[4])
    count +=1

#　ノード間のドローネもうをedgeにする
edges = create_edges(nodes, nodes_index, G)

out_e =[]
count=0
for e in edges:
    out_e.append([count, nodes_index[e[0]], nodes_index[e[1]], edges[e]])
    count += 1

df_edge = pd.DataFrame(out_e, columns=["edge_id","source", "target", "weight"])

df_edge.to_csv(f"{base_dir}/edge.csv", index=False)