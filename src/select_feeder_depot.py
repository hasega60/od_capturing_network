from gurobipy import *
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

EPS = 1.e-6

def select_feeder_hub(N, D, F, select_nodes, capa):
    # 容量制約付きpmedian
    #発生・集中量
    F_o, F_d, N_s = {}, {}, {}

    for i in N:
        o, d=0,0
        for j in N:
            if i != j:
                o += F[(i, j)]
                d += F[(j, i)]

        F_o[i]=o
        F_d[i]=d
        if i not in select_nodes:
            N_s[i] = N[i]

    model = Model("feeder_hub")
    # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y = {}, {}
    for j in select_nodes:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in N_s:
            if (i, j) not in x.keys():
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N_s:
        model.addConstr(quicksum(x[i, j] for j in select_nodes) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in select_nodes:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    #model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個

    #容量制約
    for j in select_nodes:
        model.addConstr(quicksum(x[i, j] * (F_o[i]+F_d[i]) for i in N_s) <= capa)

    model.setObjective(quicksum(D[i, j] * x[i, j] * (F_o[i]+F_d[i]) for i in N_s for j in select_nodes), GRB.MINIMIZE)

    model.update()
    model.optimize()

    model.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    select_node_hubs = [j for j in x if x[j].X > EPS]
    dist=0
    for i in select_node_hubs:
        dist += D[(i[0], i[1])]

    depot_capacity={}
    for t in select_node_hubs:
        if t[1] in depot_capacity.keys():
            depot_capacity[t[1]] += F_o[t[0]]+F_d[t[0]]
        else:
            depot_capacity[t[1]] = F_o[t[0]] + F_d[t[0]]

    print(f"feeder_hubs:{hubs}, dist:{dist}")
    print(f"capacity:{depot_capacity}")
    return hubs, select_node_hubs, dist

def select_feeder_hub2(N, D, F, select_nodes, distance_limit, capacity_cost):
    # 配置コスト+容量コストの最小化
    model = Model("feeder_hub")
    # 発生・集中量
    F_o, F_d, N_s = {}, {}, {}
    for i in N:
        if i not in select_nodes:
            N_s[i] = N[i]
        f_o, f_d = 0,0
        for j in N:
            if i != j:
                f_o += F[(i, j)]
                f_d += F[(j, i)]

        F_o[i] = f_o
        F_d[i] = f_d

        # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y = {}, {}
    for j in select_nodes:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in N_s:
            if (i, j) not in x.keys():
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N_s:
        model.addConstr(quicksum(x[i, j] for j in select_nodes) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in select_nodes:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    #距離制約
    for j in select_nodes:
        for i in N_s:
            model.addConstr(D[(i, j)] * x[i, j] <= distance_limit)

    model.setObjective(quicksum(y[j] + capacity_cost * x[i, j] * (F_o[i]+F_d[i]) for i in N_s for j in select_nodes), GRB.MINIMIZE)

    model.update()
    model.optimize()

    model.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    select_node_hubs = [j for j in x if x[j].X > EPS]
    dist = 0

    for i in select_node_hubs:
        dist += D[(i[0], i[1])]

    depot_capacity={}

    for t in select_node_hubs:
        if t[1] in depot_capacity.keys():
            depot_capacity[t[1]] += F_o[t[0]]+F_d[t[0]]
        else:
            depot_capacity[t[1]] = F_o[t[0]] + F_d[t[0]]

    print(f"feeder_hubs:{hubs}, dist:{dist}")
    print(f"capacity:{depot_capacity}")
    return hubs, select_node_hubs, dist, depot_capacity
