import pandas as pd
import networkx as nx
import numpy as np
base_dir="../data/moriya"


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
out_e=[]
for v in df_out.values:
    if v[3] is not None and len(v[3])==2:
        out_e.append(v)

df_edge = pd.DataFrame(out_e, columns=["source", "target", "weight", "path"])

df_edge.to_csv(f"{base_dir}/edge.csv", index=False)