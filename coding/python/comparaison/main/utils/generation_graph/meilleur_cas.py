from .plot_rapport import plot_like_rapport
import sys
sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    file_name = "rapport approximation differentiel.xlsx"
    col_name = "Meilleur cas"

    # Liste des feuilles à traiter.
    sheet_names = ["erdos_renyi", "bhoslib", "regular", "tree"]

    for sheet in sheet_names:

        output_folder = f"graphics/meilleur_cas/{sheet}"
        plot_like_rapport(
            file_name=file_name,
            sheet_name=sheet,
            original_col=col_name,
            output_folder=output_folder
        )
