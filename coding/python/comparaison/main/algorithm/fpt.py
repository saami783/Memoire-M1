def fpt_vertex_cover(graph, k):
    """
    Renvoie un vertex cover de taille <= k si possible, sinon None.
    Complexité en O(2^k * poly(n)).
    """
    # 1) Si le graphe n'a plus d'arêtes, on peut renvoyer l'ensemble vide
    if graph.number_of_edges() == 0:
        return set()

    # 2) Si un sommet a un degré > k, il doit être dans le cover
    degrees = dict(graph.degree())
    for node, deg in degrees.items():
        if deg > k:
            # On doit inclure ce node
            if k < 1:
                return None  # impossible
            # On réduit k et on retire ce sommet du graphe
            subgraph = graph.copy()
            subgraph.remove_node(node)
            rec_cover = fpt_vertex_cover(subgraph, k - 1)
            if rec_cover is None:
                return None
            else:
                return rec_cover | {node}

    # 3) Sinon, on prend une arête (u, v) quelconque
    (u, v) = list(graph.edges())[0]

    # 3a) On branche en mettant u dans la cover
    if k < 1:
        return None
    subgraph_u = graph.copy()
    subgraph_u.remove_node(u)
    cover_u = fpt_vertex_cover(subgraph_u, k - 1)

    if cover_u is not None:
        return cover_u | {u}

    # 3b) Sinon on branche en mettant v dans la cover
    subgraph_v = graph.copy()
    subgraph_v.remove_node(v)
    cover_v = fpt_vertex_cover(subgraph_v, k - 1)

    if cover_v is not None:
        return cover_v | {v}

    # Si aucune branche ne fonctionne, c'est impossible
    return None
