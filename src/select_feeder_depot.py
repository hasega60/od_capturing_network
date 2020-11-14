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
        model.addConstr(quicksum(x[i, j] * (F_o[i]+F_d[i]) for i in N_s for j in select_nodes) <= capa)

    model.setObjective(quicksum(D[i, j] * x[i, j] * (F_o[i]+F_d[i])for i in N_s for j in select_nodes), GRB.MINIMIZE)

    model.update()
    model.optimize()

    model.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    print("hubs:" + str(hubs))
    return hubs

def select_feeder_hub2(N, D, F, select_nodes, alfa=1, beta=1):
    # 距離コスト＋配置コストの最小化
    model = Model("feeder_hub")
    # 発生・集中量
    F_o, F_d, N_s = {}, {}, {}
    for i in N:
        F_o[i] = quicksum(F[(i, j)] for j in N)
        F_d[i] = quicksum(F[(j, i)] for j in N)
        if i not in select_nodes:
            N_s[i] = N[i]
    
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

    # model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個

    model.setObjective(quicksum(alfa * D[i, j] * x[i, j] * (F_o[i]+F_d[i]) + beta * x[i, j] * (F_o[i]+F_d[i])
                                for i in N_s for j in select_nodes), GRB.MINIMIZE)

    model.update()
    model.optimize()

    model.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    print("hubs:" + str(hubs))
    return hubs
