import random
from collections import deque


def bfs(graph):

    vertex_cover = set()
    visited = set()

    nodes = list(graph.nodes())
    random.shuffle(nodes)

    for start in nodes:
        if start not in visited:
            queue = deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                for v in graph.neighbors(u):
                    if v not in visited:
                        if graph.degree(u) > graph.degree(v):
                            vertex_cover.add(u)
                        else:
                            vertex_cover.add(v)
                        visited.add(v)
                        queue.append(v)

    return list(vertex_cover)