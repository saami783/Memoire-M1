import random
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chisquare
import networkx as nx

SEED = 42
random.seed(SEED)


def plot_choice_distribution(counts, nodes, num_trials, title):
    """
    Trace un histogramme pour visualiser la répartition des choix des nœuds.
    Utilise le même style graphique que pour la distribution des intervalles.
    """
    values = sorted(nodes)
    frequencies = [counts[v] for v in values]

    plt.figure(figsize=(12, 6))
    plt.bar(values, frequencies, width=0.8, edgecolor="black", alpha=0.7)

    # Titre et axes
    plt.title(f"{title} ({num_trials:,} tirages)", fontsize=14)
    plt.xlabel("Nœuds", fontsize=12)
    plt.ylabel("Nombre de fois que ce nœud a été choisi", fontsize=12)

    plt.ticklabel_format(style='plain', axis='y')
    max_freq = max(frequencies)
    step = max_freq // 10 if max_freq // 10 > 0 else 1
    y_ticks = range(0, max_freq + step, step)
    plt.yticks(y_ticks, [f"{y:,}" for y in y_ticks], fontsize=10)

    # Gestion des ticks en X
    if len(values) > 20:
        plt.xticks(values[::max(len(values) // 20, 1)], fontsize=10)
    else:
        plt.xticks(values, fontsize=10)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def analyze_and_plot(G, selection_type="max", num_trials=10_000_000):
    """
    Analyse et affiche la distribution de choix entre les nœuds de degré min ou max d'un graphe G.

    Args:
    - G (networkx.Graph): le graphe à analyser
    - selection_type (str): "max" ou "min" pour sélectionner les nœuds de degré maximal ou minimal
    - num_trials (int): nombre de tirages pour la simulation

    Renvoie : None (affiche les résultats et le graphique)
    """
    degrees = dict(G.degree())
    if selection_type == "max":
        target_degree = max(degrees.values())
        title = "Répartition des choix parmi les nœuds de degré maximum"
    else:
        target_degree = min(degrees.values())
        title = "Répartition des choix parmi les nœuds de degré minimum"

    target_nodes = [node for node, deg in degrees.items() if deg == target_degree]

    print(f"Nœuds et leurs degrés : {degrees}")
    print(f"Degré {'maximum' if selection_type == 'max' else 'minimum'} : {target_degree}")
    print(f"Nœuds de degré {selection_type}imum : {target_nodes}")

    counts = Counter()

    for _ in range(num_trials):
        chosen = random.choice(target_nodes)
        counts[chosen] += 1

    expected = num_trials / len(target_nodes)
    observed_frequencies = [counts[node] for node in target_nodes]
    expected_frequencies = [expected] * len(target_nodes)

    chi2_stat, p_value = chisquare(observed_frequencies, expected_frequencies)

    print("\nRésultats du test du Chi² :")
    print(f"  Chi² Statistic : {chi2_stat:.2f}")
    print(f"  p-value : {p_value:.4f}")

    if p_value > 0.05:
        print(
            f"Aucune preuve de non-uniformité. Les nœuds de degré {selection_type}imum sont sélectionnés de manière équiprobable.")
    else:
        print("Distribution non uniforme détectée (p-value ≤ 0.05).")

    # Affichage du graphique
    plot_choice_distribution(counts, target_nodes, num_trials, title)


if __name__ == "__main__":
    num_trials = 100_000_000

    # Graphe pour le degré minimum
    G_min = nx.Graph()
    G_min.add_node(0)
    for i in range(1, 6):
        # Chaque noeud 1 à 5 est connecté uniquement à 0, degré = 1
        G_min.add_edge(0, i)

    print("=== Test pour les nœuds de degré minimum ===")
    analyze_and_plot(G_min, selection_type="min", num_trials=num_trials)

    # Graphe pour le degré maximum
    G_max = nx.Graph()
    G_max.add_node(0)
    for i in range(1, 6):
        G_max.add_edge(0, i)
    for i in range(1, 6):
        for j in range(i + 1, 6):
            G_max.add_edge(i, j)

    print("\n=== Test pour les nœuds de degré maximum ===")
    analyze_and_plot(G_max, selection_type="max", num_trials=num_trials)
