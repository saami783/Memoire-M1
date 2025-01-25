from pulp import LpProblem, LpMinimize, LpVariable, lpSum

def lp_rounding_vertex_cover(graph):
    """
    Linear Programming Rounding (LP-Rounding) for Vertex Cover.
    Solves the LP relaxation and rounds fractional values.
    """
    prob = LpProblem("VertexCover", LpMinimize)
    x = {v: LpVariable(f"x_{v}", lowBound=0, upBound=1) for v in graph.nodes()}
    for u, v in graph.edges():
        prob += x[u] + x[v] >= 1
    prob += lpSum(x.values())
    prob.solve()
    C = {v for v in graph.nodes() if x[v].value() >= 0.5}
    return list(C)