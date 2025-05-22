import networkx as nx
from itertools import combinations, product
from scipy.optimize import linprog
import sqlite3
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

# Règle R.1 – suppression des sommets isolés
def reduction_R1(G, k):
    G = G.copy()
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return G, k

# Règle R.2 – sommet de degré > k
def reduction_R2(G, k):
    for v in G.nodes:
        if G.degree[v] > k:
            G = G.copy()
            G.remove_node(v)
            return G, k - 1
    return G, k


def find_crown_decomposition(G):
    """
    Recherche une décomposition en couronne (C, H, B) dans G.
    Retourne (C, H) si trouvée, sinon None.
    """
    # Calcul d’un matching maximum sur le graphe biparti G'
    matching = nx.algorithms.bipartite.hopcroft_karp_matching(G)

    # Filtrage : on ne garde que les correspondances C-H
    matched_C = {u for u in matching if u in G and matching[u] in G}
    matched_H = {matching[u] for u in matched_C}
    C = set()
    H = set()

    for u in matched_C:
        if u not in matched_H and all(v not in matched_C for v in G.neighbors(u)):
            C.add(u)
            H.add(matching[u])

    # Vérifications formelles
    if not C:
        return None

    if not nx.is_independent_set(G, C):
        return None

    for c in C:
        for b in G.nodes:
            if b != c and G.has_edge(c, b) and b not in H:
                return None

    # On a bien une couronne (C, H)
    return C, H


def reduction_R3(G, k):
    """
    Applique la réduction R.3 si une crown decomposition est trouvée.
    """
    # Construire le graphe biparti G[C,H]
    for component in nx.connected_components(G):
        subG = G.subgraph(component)
        try:
            C, H = find_crown_decomposition(subG)
            if C and H:
                G_reduced = G.copy()
                G_reduced.remove_nodes_from(C.union(H))
                return G_reduced, k - len(H)
        except:
            continue

    return G, k

def reduction_R4(G, k):
    """
    LP-based kernelisation (R.4) via relaxation linéaire de Vertex Cover.
    """
    nodelist = list(G.nodes)
    idx_map = {v: i for i, v in enumerate(nodelist)}
    num_vars = len(nodelist)

    # Objectif : min sum(x_v)
    c = [1.0] * num_vars

    # Contraintes : x_u + x_v >= 1 pour chaque arête
    A = []
    b = []

    for u, v in G.edges:
        row = [0.0] * num_vars
        row[idx_map[u]] = -1.0
        row[idx_map[v]] = -1.0
        A.append(row)
        b.append(-1.0)

    # bornes : x_v ∈ [0, 1]
    bounds = [(0, 1) for _ in range(num_vars)]

    # Résolution LP
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if not res.success:
        return G, k  # Échec du solveur, on ne fait rien

    x = res.x
    x_sum = sum(x)

    if x_sum > k:
        # L'instance est un non : il n'existe pas de VC de taille ≤ k
        return None, None

    V0 = set()
    V1 = set()
    for v, val in zip(nodelist, x):
        if val < 0.5:
            V0.add(v)
        elif val > 0.5:
            V1.add(v)
        # les autres sont dans V_{1/2}

    G_reduced = G.copy()
    G_reduced.remove_nodes_from(V0.union(V1))
    return G_reduced, k - len(V1)

# Règle R.5 – arête pendante
def reduction_R5(G, k):
    for u in G.nodes:
        if G.degree[u] == 1:
            v = next(iter(G.neighbors(u)))
            G = G.copy()
            G.remove_nodes_from([u, v])
            return G, k - 1
    return G, k

def reduction_R6(G, k):
    """
    Si un sommet u a un voisin v tel que N(v) ⊆ N[u], alors u est dans une VC de taille optimale.
    On le retire et on décrémente k.
    """
    for u in G.nodes:
        for v in G.neighbors(u):
            if set(G.neighbors(v)).issubset(set(G.neighbors(u)).union({u})):
                G_reduced = G.copy()
                G_reduced.remove_node(u)
                return G_reduced, k - 1
    return G, k

def reduction_R7(G, k):
    """
    Pour un sommet v de degré 2 dont les voisins ne vérifient pas R6, contracte N[v] en un sommet z.
    Décrémente k.
    """
    for v in G.nodes:
        if G.degree[v] == 2:
            a, b = list(G.neighbors(v))

            # Vérifie si R6 s'applique à (a,v) ou (b,v) — si oui, on ne fait pas R7
            if set(G.neighbors(v)).issubset(set(G.neighbors(a)).union({a})):
                continue
            if set(G.neighbors(v)).issubset(set(G.neighbors(b)).union({b})):
                continue

            # R6 ne s'applique pas, on applique R7
            G_reduced = G.copy()
            neighbors_a = set(G.neighbors(a)) - {v, b}
            neighbors_b = set(G.neighbors(b)) - {v, a}
            neighbors_v = set(G.neighbors(v)) - {a, b}
            new_neighbors = neighbors_a.union(neighbors_b).union(neighbors_v)

            # Supprimer a, b, v
            G_reduced.remove_nodes_from([a, b, v])

            # Ajouter un nouveau sommet z et ses arêtes
            z = f"z_{v}"
            G_reduced.add_node(z)
            for u in new_neighbors:
                G_reduced.add_edge(z, u)

            return G_reduced, k - 1

    return G, k

def is_clique(G, nodes):
    return all(G.has_edge(u, v) for u, v in combinations(nodes, 2))

def reduction_R8(G, k, alpha=4):
    """
    Règle R.8 : clique-clique avec non-arêtes bien structurées entre C1 et C2.
    """
    for v in G.nodes:
        if G.degree[v] > alpha:
            continue

        neighbors = list(G.neighbors(v))

        # Toutes les partitions possibles du voisinage
        for i in range(1, len(neighbors)):
            C1 = neighbors[:i]
            C2 = neighbors[i:]
            if not C1 or not C2:
                continue

            if not is_clique(G, C1) or not is_clique(G, C2):
                continue

            # Identifier les non-arêtes entre C1 et C2
            M = [(c1, c2) for c1, c2 in product(C1, C2) if not G.has_edge(c1, c2)]

            # Vérifier la condition clé : chaque c1 a exactement un non-voisin dans C2
            count_non_edges = {c1: 0 for c1 in C1}
            for c1, c2 in M:
                count_non_edges[c1] += 1
            if not all(count == 1 for count in count_non_edges.values()):
                continue

            # Appliquer la réduction
            G_reduced = G.copy()
            G_reduced.remove_nodes_from(C2 + [v])

            for c1, c2 in M:
                for u in G.neighbors(c2):
                    if u in G_reduced and u != c1:
                        G_reduced.add_edge(c1, u)

            return G_reduced, k - len(C2)

    return G, k

def reduction_R9(G, k):
    """
    Réduction R.9 : v de degré 3 avec voisins indépendants (a,b,c).
    Ajout de certaines arêtes, suppression de v.
    """
    for v in G.nodes:
        if G.degree[v] != 3:
            continue
        a, b, c = list(G.neighbors(v))

        # Vérifier si R.6 s'applique
        def r6_applies(u, w):
            return set(G.neighbors(w)).issubset(set(G.neighbors(u)).union({u}))
        if any(r6_applies(x, y) for x, y in [(a, v), (b, v), (c, v)]):
            continue

        # Vérifier si a, b, c sont indépendants
        if G.has_edge(a, b) or G.has_edge(b, c) or G.has_edge(a, c):
            continue

        # R.9 applicable
        G_reduced = G.copy()
        G_reduced.remove_node(v)

        # Arêtes à ajouter
        new_edges = set()
        new_edges.add((a, b))
        new_edges.add((b, c))
        for x in G.neighbors(b):
            if x != a:
                new_edges.add((a, x))
        for y in G.neighbors(c):
            if y != b:
                new_edges.add((b, y))
        for z in G.neighbors(a):
            if z != c:
                new_edges.add((c, z))

        # Ajouter les arêtes sans dupliquer
        for u, w in new_edges:
            if G_reduced.has_node(u) and G_reduced.has_node(w):
                G_reduced.add_edge(u, w)

        return G_reduced, k

    return G, k

def reduction_R10(G, k):
    for v in G.nodes:
        if G.degree[v] != 4:
            continue

        N_v = list(G.neighbors(v))

        # Vérifier si R6 ou R8 s’applique à v (simplifié ici)
        def r6_applies(u, w):
            return set(G.neighbors(w)).issubset(set(G.neighbors(u)).union({u}))
        if any(r6_applies(x, v) for x in N_v):
            continue

        # Compter les arêtes dans le sous-graphe induit G[N(v)]
        subgraph = G.subgraph(N_v)
        if subgraph.number_of_edges() < 3:
            continue  # Seulement si G[N(v)] a ≥ 3 arêtes

        # Appliquer la réduction
        G_reduced = G.copy()
        G_reduced.remove_node(v)

        # Ajouter toutes les arêtes manquantes entre voisins de v
        for u, w in combinations(N_v, 2):
            G_reduced.add_edge(u, w)

        # Ajouter les arêtes croisées :
        # - {a, b} avec a, b ∈ N(v)
        # - pour chaque x dans N(d), ajouter {a,x}, {b,x}
        # - pour chaque y dans N(a), ajouter {c,y}, {d,y}
        # Choix arbitraire pour nommer les sommets : a-b-c-d selon l’ordre de N(v)
        a, b, c, d = N_v

        for x in G.neighbors(d):
            if x not in N_v:
                G_reduced.add_edge(a, x)
                G_reduced.add_edge(b, x)
        for y in G.neighbors(a):
            if y not in N_v:
                G_reduced.add_edge(c, y)
                G_reduced.add_edge(d, y)

        return G_reduced, k

    return G, k


def apply_all_reductions(G, k):
    """
    Applique toutes les règles R1 à R10 de manière itérative jusqu’à stabilisation.
    """
    reduction_functions = [
        reduction_R1,
        reduction_R2,
        reduction_R3,
        reduction_R4,
        reduction_R5,
        reduction_R6,
        reduction_R7,
        reduction_R8,
        reduction_R9,
        reduction_R10,
    ]

    changed = True
    while changed:
        changed = False
        for reduce_fn in reduction_functions:
            result = reduce_fn(G, k)
            if result == (None, None):
                return None, None  # instance infaisable détectée (R4)

            G_new, k_new = result
            if G_new.number_of_nodes() < G.number_of_nodes() or G_new.number_of_edges() < G.number_of_edges() or k_new != k:
                G, k = G_new, k_new
                changed = True
                break  # recommence depuis la première règle

    return G, k


def canonical_form(G):
    return weisfeiler_lehman_graph_hash(G)

def kernelize_graphs_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Lire les graphes avec leur taille de couverture optimale
    cursor.execute("SELECT id, g6_string, Cover_Size FROM original_graphs")
    rows = cursor.fetchall()

    # Créer la table de sortie si elle n'existe pas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kernelized_graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id INTEGER,
            reduced_canonical_form TEXT,
            reduced_g6 TEXT,
            reduced_k INTEGER,
            nb_nodes INTEGER,
            nb_edges INTEGER,
            FOREIGN KEY(original_id) REFERENCES original_graphs(id)
        )
    """)

    for original_id, g6_str, cover_size in rows:
        try:
            G = nx.from_graph6_bytes(g6_str.encode("utf-8"))
            G_kernel, k_kernel = apply_all_reductions(G, cover_size)

            if G_kernel is None:
                continue  # instance infaisable

            reduced_g6 = nx.to_graph6_bytes(G_kernel).decode("utf-8").strip()
            reduced_form = canonical_form(G_kernel)
            nb_nodes = G_kernel.number_of_nodes()
            nb_edges = G_kernel.number_of_edges()

            cursor.execute("""
                INSERT INTO kernelized_graphs (
                    original_id,
                    reduced_canonical_form,
                    reduced_g6,
                    reduced_k,
                    nb_nodes,
                    nb_edges
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (original_id, reduced_form, reduced_g6, k_kernel, nb_nodes, nb_edges))

        except Exception as e:
            print(f"Erreur sur le graphe ID {original_id} : {e}")
            continue

    conn.commit()
    conn.close()
    print("Tous les graphes ont été kernelisés et sauvegardés.")


# Exemple d'utilisation :
kernelize_graphs_from_db("graphe_vertexcover.sqlite")
