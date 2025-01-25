import os
import random
import pandas as pd
import logging
from tqdm import tqdm
from utils.file_handler import get_dimacs_files
from graphs.load_graph import load_graph_from_dimacs
from algorithm.evaluation import process_graph

SEED = 42
INPUT_DIR = "dimacs_files/trees"
OUTPUT_FILE = "out/test/tree_results.csv"

# Configuration du logger
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialiser la graine pour la reproductibilité
random.seed(SEED)

def main():
    try:
        file_list = get_dimacs_files(INPUT_DIR)
    except FileNotFoundError as e:
        logger.error(e)
        exit(1)
    except ValueError as e:
        logger.warning(e)
        exit(1)

    results = []

    for filename in tqdm(file_list, desc="Traitement des fichiers"):
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            graph = load_graph_from_dimacs(filepath)
            graph_name, num_nodes, opt_size, _ = filename.replace(".dimacs", "").split("-")
            results.extend(process_graph(graph_name, graph, int(opt_size), verbose=False))
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {filename} : {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    logger.warning(f"\nRésultats sauvegardés dans le fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Une erreur critique est survenue : {e}")
