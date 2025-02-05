import random
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx


def load_dimacs_graph(filename):
    """
    Charge un graphe non orienté au format DIMACS à partir d'un fichier.

    Format attendu :
      c <commentaires...>
      p edge <nb_sommets> <nb_aretes>
      e <u> <v>
      e <x> <y>
      ...
    Les sommets peuvent être indexés à partir de 1.
    """
    G = nx.Graph()

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignorer les lignes vides et les commentaires
            if not line or line.startswith('c'):
                continue
            parts = line.split()

            # p edge <nb_sommets> <nb_aretes> (vous pouvez utiliser ces infos si besoin)
            if parts[0] == 'p':
                # Exemple : p edge 100 200
                # num_vertices = int(parts[2])
                # num_edges = int(parts[3])
                pass

            # e <u> <v>
            elif parts[0] == 'e':
                # Les sommets sont souvent 1-based en DIMACS, on peut les laisser tels quels
                # ou les passer en 0-based (à vous de voir).
                u = int(parts[1])
                v = int(parts[2])
                G.add_edge(u, v)
    return G


def plot_choice_distribution(counts, nodes, num_trials, title):
    """
    Affiche un histogramme des sélections pour chaque nœud.
      - Utilise un axe catégoriel (positions 0, 1, 2, …)
      - Adapte la largeur des barres et/ou la taille de la figure en fonction
        du nombre de nœuds afin d'éviter que les barres paraissent trop larges.
      - Aucun titre n'est affiché.
    """
    # Trie les nœuds pour un affichage cohérent
    sorted_nodes = sorted(nodes)
    frequencies = [counts[n] for n in sorted_nodes]
    x_positions = list(range(len(sorted_nodes)))

    num_bars = len(sorted_nodes)

    # Ajuster la taille de la figure et la largeur des barres selon le nombre de catégories
    if num_bars < 10:
        # Pour peu de barres, réduire la largeur des barres et la figure
        bar_width = 0.3
        fig_width = max(4, num_bars * 1.5)
        # On ajoutera aussi une marge sur l'axe X
        x_margin = 0.5
    else:
        bar_width = 0.8
        fig_width = 10
        x_margin = 0.0  # pas de marge particulière

    plt.figure(figsize=(fig_width, 5))
    plt.bar(x_positions, frequencies, width=bar_width, color='skyblue', edgecolor="black", alpha=0.7)

    # Définir les labels sur l'axe X
    # Si trop de barres, on n'affiche qu'une étiquette sur n
    if num_bars > 20:
        step = max(1, num_bars // 20)
        displayed_positions = x_positions[::step]
        displayed_labels = [str(sorted_nodes[i]) for i in displayed_positions]
        plt.xticks(displayed_positions, displayed_labels, rotation=90, fontsize=10)
    else:
        plt.xticks(x_positions, [str(n) for n in sorted_nodes], rotation=0, fontsize=10)

    # Suppression du titre du graphique
    # plt.title(f"{title} ({num_trials:,} tirages)", fontsize=14)

    plt.xlabel("Nœuds", fontsize=12)
    plt.ylabel("Nombre de sélections", fontsize=12)

    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajuster les limites de l'axe X si peu de barres
    if x_margin > 0:
        plt.xlim(-x_margin, num_bars - 1 + x_margin)

    plt.tight_layout()
    plt.show()


def analyze_and_plot(G, selection_type="max", num_trials=10_000_000):
    """
    Sélectionne uniformément un nœud de degré max ou min dans le graphe G,
    répété num_trials fois, puis affiche la distribution.

    G : réseau (networkx.Graph)
    selection_type : "max" ou "min" pour spécifier le degré cible
    num_trials : nombre de tirages pour l'expérience
    """
    # Récupération des degrés de chaque nœud
    degrees = dict(G.degree())

    # Degré cible : max ou min
    if selection_type == "max":
        target_degree = max(degrees.values())
    else:
        target_degree = min(degrees.values())

    # On rassemble les nœuds ayant le degré cible
    target_nodes = [node for node, deg in degrees.items() if deg == target_degree]

    # Titre pour le graphique (sera passé à la fonction mais non affiché)
    title = f"Répartition des choix (degré {selection_type}imum)"

    # Comptage des sélections
    counts = Counter(random.choice(target_nodes) for _ in range(num_trials))

    # Affiche la distribution obtenue
    plot_choice_distribution(counts, target_nodes, num_trials, title)


if __name__ == "__main__":
    # Chargez votre fichier DIMACS
    file_dimacs = "main/test/test_graph_tree.dimacs"  # À adapter si nécessaire
    G = load_dimacs_graph(file_dimacs)

    # Nombre de répétitions pour l'expérience
    num_trials = 10_000_000

    # Analyse sur les nœuds de degré minimum
    print("=== Test pour les nœuds de degré minimum ===")
    analyze_and_plot(G, selection_type="min", num_trials=num_trials)

    # Analyse sur les nœuds de degré maximum
    print("\n=== Test pour les nœuds de degré maximum ===")
    analyze_and_plot(G, selection_type="max", num_trials=num_trials)
