from gurobipy import *
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

EPS = 1.e-6



def pmedian(N, D, p, bWeight=True):
    model = Model("p-median")
    # binary xij 顧客iの需要が施設jによって満たされる時　　yj施設jが開設するとき
    x, y = {}, {}
    for j in N:
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        for i in N:
            if (i, j) not in x.keys():
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model.update()

    for i in N:
        model.addConstr(quicksum(x[i, j] for j in N) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in N:
            model.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
    model.addConstr(quicksum(y[j] for j in N) == p, "Facilities")  # 施設数はp個

    if bWeight:
        model.setObjective(quicksum(D[i, j] * x[i, j] * N[i][2] for i in N for j in N), GRB.MINIMIZE)
    else:
        model.setObjective(quicksum(D[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

    model.update()
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("pmedian:実行不可能で終了_" + str(p))
        # outIIS(model_rmf)
        return None

    model.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    print("hubs:" + str(hubs))
    return hubs


def pmedian_existing(N, D, yExt, q, bWeight=True):
    model2 = Model("p-median-ext")
    I = N
    J = N
    r = 0  # 廃止施設は考えない
    x, y, ye, z = {}, {}, {}, {}

    for j in J:
        y[j] = model2.addVar(vtype="B", name="y(%s)" % j)
        ye[j] = model2.addVar(vtype="B", name="ye(%s)" % j)
        z[j] = model2.addVar(vtype="B", name="z(%s)" % j)
        for i in I:
            x[i, j] = model2.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    model2.update()

    for i in I:
        model2.addConstr(quicksum(x[i, j] for j in J) == 1, "Assign(%s)" % i)  # 顧客iが何れかの施設に割り当てられる
        for j in J:
            model2.addConstr(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))

    model2.addConstr(quicksum(y[j] for j in J) == -r + q, "Facilities_New")
    model2.addConstr(quicksum(z[j] for j in J) == r + q, "Changed")

    for j in J:
        for e in yExt:
            if j == e:
                model2.addConstr(-1 * z[j] <= y[j] - 1)
                model2.addConstr(1 - ye[j] <= z[j])
            else:
                model2.addConstr(-1 * z[j] <= y[j])
                model2.addConstr(ye[j] <= z[j])

    # 距離行列の更新
    cNew = updateDistance(I, J, D, yExt)
    if bWeight:
        # 重みを考慮するか
        model2.setObjective(quicksum(cNew[i, j] * x[i, j] * N[i][2] for i in I for j in J), GRB.MINIMIZE)
    else:
        model2.setObjective(quicksum(cNew[i, j] * x[i, j] for i in I for j in J), GRB.MINIMIZE)

    model2.update()
    model2.optimize()
    if model2.Status == GRB.INFEASIBLE:
        print("pmedian_ext:実行不可能で終了_" + "_" + str(q))
        # outIIS(model_rmf)
        return None, None

    model2.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    select = [j for j in x if x[j].X > EPS]

    model2.write("pmedian_ext_sol_out.mst")

    return hubs, select


def pcenter_existing(N, D, yExt, q):
    r = 0
    model3 = Model("p-center-ext")
    a = model3.addVar(vtype="C", name="a")
    I = N
    J = N
    x, y, ye, z = {}, {}, {}, {}
    for j in J:
        y[j] = model3.addVar(vtype="B",
                             name="y(%s)" % j)
        ye[j] = model3.addVar(vtype="B",
                              name="ye(%s)" % j)
        z[j] = model3.addVar(vtype="B",
                             name="z(%s)" % j)
        for i in I:
            x[i, j] = model3.addVar(vtype="B",
                                    name="x(%s,%s)" % (
                                        i, j))
    model3.update()

    # 距離行列の更新
    cNew = updateDistance(I, J, D, yExt)

    for i in I:
        model3.addConstr(
            quicksum(x[i, j] for j in J) == 1,
            "Assign(%s)" % i)
        model3.addConstr(quicksum(
            cNew[i, j] * x[i, j] for j in J) <= a,
                         "Max_x(%s)" % (i))

        for j in J:
            model3.addConstr(x[i, j] <= y[j],
                             "Strong(%s,%s)" % (
                                 i, j))

    model3.addConstr(
        quicksum(y[j] for j in J) == -r + q,
        "Facilities")

    for j in J:
        for e in yExt:
            if j == e:
                model3.addConstr(
                    -1 * z[j] <= y[j] - 1)
                model3.addConstr(
                    1 - ye[j] <= z[j])
            else:
                model3.addConstr(
                    -1 * z[j] <= y[j])
                model3.addConstr(ye[j] <= z[j])

    model3.setObjective(a, GRB.MINIMIZE)
    model3.update()

    model3.optimize()

    model3.__data = x, y
    hubs = [j for j in y if y[j].X > EPS]
    select = [j for j in x if x[j].X > EPS]
    print("hubs:" + str(hubs))

    return hubs, select



def updateDistance(I, J, c, yExt):
    ce = {}
    for i in I:
        for j in J:
            dist = c[i, j]
            for k in yExt:
                distNew = c[i, k]
                if distNew < dist:
                    dist = distNew

                ce[i, j] = dist

    return ce

def createTerminalCombination(hubs, D, minLength):
    list = []
    for hub1 in hubs:
        for hub2 in hubs:
            if list not in [(hub1, hub2)]:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 == hub2 or D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list


def createTerminalCombinationNoLoop(hubs, D, minLength):
    list = []
    for hub1 in hubs:
        for hub2 in hubs:
            if list not in [(hub1, hub2)]:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 != hub2 and D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list

def createTerminalCombination_mainsub(main_hubs, sub_hubs, D, minLength):
    list = []
    for hub1 in main_hubs:
        for hub2 in sub_hubs:
            if list not in [(hub1, hub2)]:
                # 同一ハブ or ハブ間距離が一定長以上
                if hub1 != hub2 and D[hub1, hub2] > minLength:
                    list.append([hub1, hub2])
    return list