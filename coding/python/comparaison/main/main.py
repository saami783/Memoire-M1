import logging
from tqdm import tqdm
import utils.graph as graph_utils
from algorithm.evaluation import process_graph
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
# excepted ed max sum dg
OUTPUT_FILE = "out/ed_max_sum_dg.csv"

def main():

    try:
        graphs = graph_utils.get_graphs_from_db()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des graphes depuis la base de données : {e}")
        exit(1)

    results = []

    for id, graph_name, graph_class, canonical_form, cover_size, instance_number, num_nodes, num_edges in tqdm(graphs, desc="Traitement des graphes"):
        try:
            graph = graph_utils.load_graph_from_db(canonical_form)

            result = process_graph(id, instance_number, graph_class, graph_name, graph, cover_size, num_nodes, num_edges, verbose=False)

            results.extend(result)

        except Exception as e:
            logger.error(f"Erreur lors du traitement du graphe {graph_name} : {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    logger.warning(f"\nRésultats sauvegardés dans le fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Une erreur critique est survenue : {e}")
