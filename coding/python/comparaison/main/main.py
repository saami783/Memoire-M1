import logging
from tqdm import tqdm
import utils.graph as graph_utils
from algorithm.evaluation import process_graph_from_db, process_graph_from_hog
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
# excepted ed max sum dg
OUTPUT_FILE = "out/algo20_a_300.csv"
GRAPH_SOURCE = "hog"

def main():

    try:
        if GRAPH_SOURCE == "hog":
            graphs = graph_utils.get_graphs_from_hog()
        else:
            graphs = graph_utils.get_graphs_from_db()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des graphes depuis la base de données : {e}")
        exit(1)

    results = []

    for graph_data in tqdm(graphs, desc="Traitement des graphes"):
        try:
            if GRAPH_SOURCE == "hog":
                id, canonical_form, graph_name, num_nodes, num_edges, cover_size = graph_data
                graph = graph_utils.load_graph_from_db(canonical_form)
                result = process_graph_from_hog(id, graph_class="hog", filename=graph_name,
                                       graph=graph, opt_size=cover_size, num_nodes=num_nodes, num_edges=num_edges,
                                       verbose=False)
            else:
                id, graph_name, graph_class, canonical_form, cover_size, instance_number, num_nodes, num_edges = graph_data
                graph = graph_utils.load_graph_from_db(canonical_form)
                result = process_graph_from_db(id, instance_number, graph_class, graph_name, graph, cover_size, num_nodes,
                                       num_edges, verbose=False)

            results.extend(result)

        except Exception as e:
            logger.error(f"Erreur lors du traitement du graphe {graph_data[1]} : {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    logger.warning(f"\nRésultats sauvegardés dans le fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Une erreur critique est survenue : {e}")
