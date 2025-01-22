import random
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chisquare
SEED = 42
random.seed(SEED)

def generate_random_distribution(interval, num_trials):
    """
    Génère des valeurs pseudo-aléatoires dans un intervalle donné et enregistre leur fréquence.

    Args:
    - interval (tuple): Intervalle (min, max) pour les valeurs générées.
    - num_trials (int): Nombre de tirages à effectuer.

    Returns:
    - Counter: Fréquences des valeurs générées.
    """
    min_val, max_val = interval
    counts = Counter()

    for _ in range(num_trials):
        value = random.randint(min_val, max_val)
        counts[value] += 1

    return counts


def plot_random_distribution(counts, interval, num_trials):
    """
    Trace un histogramme pour visualiser la répartition des valeurs générées.

    Args:
    - counts (Counter): Fréquences des valeurs générées.
    - interval (tuple): Intervalle des valeurs possibles.
    - num_trials (int): Nombre total de tirages.
    """
    values = list(range(interval[0], interval[1] + 1))
    frequencies = [counts.get(value, 0) for value in values]

    plt.figure(figsize=(12, 6))
    plt.bar(values, frequencies, width=0.8, edgecolor="black", alpha=0.7)

    # Titre et axes
    plt.title(f"Répartition des tirages pseudo-aléatoires ({num_trials:,} tirages) pour l'intervalle {interval}",
              fontsize=14)
    plt.xlabel("Valeurs générées", fontsize=12)
    plt.ylabel("Nombre de fois que cette valeur a été générée", fontsize=12)

    # Ajuster les ticks de l'axe Y avec une notation scientifique si nécessaire
    plt.ticklabel_format(style='plain', axis='y')
    y_ticks = range(0, max(frequencies) + int(max(frequencies) / 10), int(max(frequencies) / 10))
    plt.yticks(y_ticks, [f"{y:,}" for y in y_ticks], fontsize=10)

    if len(values) > 20:
        plt.xticks(values[::max(len(values) // 20, 1)], fontsize=10)
    else:
        plt.xticks(values, fontsize=10)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_uniformity(counts, interval, num_trials):
    """
    Vérifie l'uniformité des tirages via un test du Chi².

    Args:
    - counts (Counter): Fréquences des valeurs générées.
    - interval (tuple): Intervalle des valeurs possibles.
    - num_trials (int): Nombre total de tirages.

    Returns:
    - bool: True si les tirages sont uniformes, False sinon.
    """
    expected_frequency = num_trials / (interval[1] - interval[0] + 1)
    observed_frequencies = [counts[value] for value in range(interval[0], interval[1] + 1)]
    expected_frequencies = [expected_frequency] * len(observed_frequencies)

    chi2_stat, p_value = chisquare(observed_frequencies, expected_frequencies)
    print(f"Résultat du test du Chi² pour l'intervalle {interval} :")
    print(f"  Chi² Statistic : {chi2_stat:.2f}")
    print(f"  p-value : {p_value:.4f}")

    return p_value > 0.05


if __name__ == "__main__":
    # Configuration des intervalles et des tirages
    intervals = [ (0, 100000)]  # Différents intervalles à tester
    num_trials = 10_000_000  # Nombre de tirages par intervalle

    for interval in intervals:
        print(f"\nTest d'équiprobabilité pour l'intervalle {interval} avec {num_trials:,} tirages...")

        # Générer les données
        counts = generate_random_distribution(interval, num_trials)

        # Tracer le graphique
        plot_random_distribution(counts, interval, num_trials)

        # Évaluer l'uniformité des tirages
        is_uniform = evaluate_uniformity(counts, interval, num_trials)
        if is_uniform:
            print(f"L'intervalle {interval} montre une bonne uniformité des tirages (p-value > 0.05).\n")
        else:
            print(f"L'intervalle {interval} montre une non-uniformité des tirages (p-value ≤ 0.05).\n")
