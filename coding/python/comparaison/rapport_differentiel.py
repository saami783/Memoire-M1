import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_histograms(file_path, sheet_name='result_trees', output_dir='histograms'):
    # Load the data from the specified sheet
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Filter relevant columns
    diff_column = 'Rapport d\'approximation différentiel en %'
    algo_column = 'Heuristic'

    if diff_column not in data.columns or algo_column not in data.columns:
        raise ValueError(f"Expected columns '{diff_column}' and '{algo_column}' not found in sheet {sheet_name}.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique algorithms
    algorithms = data[algo_column].dropna().unique()

    for algo in algorithms:
        # Filter data for the algorithm
        algo_data = data[data[algo_column] == algo]

        # Extract differential approximation values
        diff_values = algo_data[diff_column].dropna()

        # Round the values to the nearest 10 for better grouping (e.g., 0, 10, 20...)
        rounded_values = (diff_values / 10).round() * 10

        # Count occurrences of each value
        counts = rounded_values.value_counts().sort_index()

        # Handle case where counts might be empty
        if counts.empty:
            print(f"No data available for algorithm '{algo}', skipping.")
            continue

        # Create the histogram
        plt.figure(figsize=(8, 6))
        plt.bar(counts.index, counts.values, width=8, color='skyblue', edgecolor='black', align='center')
        plt.xlabel('Pourcentage d\'erreur')
        plt.ylabel('Nombre de solutions')
        plt.xticks(ticks=counts.index, labels=[f"{int(x)}%" for x in counts.index])

        # Adjust y-axis to the nearest multiple of 5 above the max count
        y_max = int((max(counts.max(), 5) // 5 + 1) * 5)
        plt.yticks(range(0, y_max + 1, 5))
        plt.ylim(0, y_max)

        # Save the plot
        output_path = os.path.join(output_dir, f"{algo.replace(' ', '_')}_histogram.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Histogram for '{algo}' saved to {output_path}.")

# Example usage
plot_histograms('rapport approximation differentiel.xlsx')
