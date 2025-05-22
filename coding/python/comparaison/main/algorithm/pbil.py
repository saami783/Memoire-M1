import random
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def get_node_features(graph):
    clustering = nx.clustering(graph)
    core_number = nx.core_number(graph)
    features = {}
    for node in graph.nodes:
        features[node] = [
            graph.degree[node],
            clustering[node],
            core_number[node]
        ]
    return features

def evaluate_solution(graph, solution):
    cover = set(i for i, bit in enumerate(solution) if bit == 1)
    uncovered_edges = [e for e in graph.edges if e[0] not in cover and e[1] not in cover]

    penalty = len(uncovered_edges) * len(graph.nodes)
    return len(cover) + penalty

def generate_individual(prob_vector):
    return [1 if random.random() < p else 0 for p in prob_vector]

def train_ml_model(graph, valid_solutions):
    X, y = [], []
    features = get_node_features(graph)
    for sol in valid_solutions:
        for i, bit in enumerate(sol):
            X.append(features[i])
            y.append(bit)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf

def pbil_mvc_hybrid(graph, num_generations=500, population_size=50, learning_rate=0.1,
                    mutation_prob=0.02, mutation_shift=0.05, ml_update_every=50, ml_min_samples=100):
    n = graph.number_of_nodes()
    prob_vector = [0.5] * n
    best_solution = None
    best_score = float('inf')
    valid_solutions = []

    for generation in range(num_generations):
        population = [generate_individual(prob_vector) for _ in range(population_size)]
        scored_population = [(ind, evaluate_solution(graph, ind)) for ind in population]
        scored_population.sort(key=lambda x: x[1])

        best_ind, best_ind_score = scored_population[0]

        if evaluate_solution(graph, best_ind) == len([i for i in best_ind if i == 1]):
            valid_solutions.append(best_ind)

        # Mise à jour du vecteur de probabilité
        for i in range(n):
            prob_vector[i] = (1 - learning_rate) * prob_vector[i] + learning_rate * best_ind[i]

        # Hybridation ML
        if generation % ml_update_every == 0 and len(valid_solutions) >= ml_min_samples:
            clf = train_ml_model(graph, valid_solutions)
            features = get_node_features(graph)
            for i in range(n):
                pred = clf.predict_proba([features[i]])[0][1]  # proba d'appartenance au VC
                mix = 0.5 * best_ind[i] + 0.5 * pred
                prob_vector[i] = (1 - learning_rate) * prob_vector[i] + learning_rate * mix

        # Mutation
        for i in range(n):
            if random.random() < mutation_prob:
                shift = mutation_shift if random.random() < 0.5 else -mutation_shift
                prob_vector[i] = min(1.0, max(0.0, prob_vector[i] + shift))

        # Mémorisation du meilleur
        if best_ind_score < best_score:
            best_solution = best_ind
            best_score = best_ind_score

        if generation % 50 == 0:
            print(f"Gen {generation} — Best Score: {best_score} — Prob avg: {np.mean(prob_vector):.3f}")

    return best_solution
