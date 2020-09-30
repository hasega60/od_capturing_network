from gurobipy import *
import math


# Add variables
def pmedian(facilities, clients, charge):
    numFacilities = len(facilities)
    numClients = len(clients)

    m = Model()
    x = {}
    y = {}
    d = {}  # Distance matrix (not a variable)
    alpha = 1

    for j in range(numFacilities):
        x[j] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)

    for i in range(numClients):
        for j in range(numFacilities):
            y[(i, j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t%d,%d" % (i, j))
            d[(i, j)] = distance(clients[i], facilities[j])

    m.update()

    # Add constraints
    for i in range(numClients):
        for j in range(numFacilities):
            m.addConstr(y[(i, j)] <= x[j])

    for i in range(numClients):
        m.addConstr(quicksum(y[(i, j)] for j in range(numFacilities)) == 1)

    m.setObjective(quicksum(charge[j] * x[j] + quicksum(alpha * d[(i, j)] * y[(i, j)]
                                                        for i in range(numClients)) for j in range(numFacilities)))

    m.optimize()


def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


# Problem data
if __name__ == "__main__":
    ary_clients = [[[0, 1.5], [2.5, 1.2]],[[1, 1.5], [1.5, 1.2]],[[2, 1.5], [0.5, 1.2]]] #ŒÚ‹q‚Ìƒpƒ^[ƒ“
    facilities = [[0, 0], [0, 1], [0, 1],
                  [1, 0], [1, 1], [1, 2],
                  [2, 0], [2, 1], [2, 2]]
    charge = [3, 2, 3, 1, 3, 3, 4, 3, 2]

    for clients in ary_clients:
    
        pmedian(facilities, clients, charge)
