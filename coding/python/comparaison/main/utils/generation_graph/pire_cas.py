# pire_cas.py
from .plot_rapport import plot_like_rapport

if __name__ == "__main__":
    file_name = "rapport approximation differentiel.xlsx"
    # sheet_name = "tree"
    # sheet_name = "barabasi_albert"
    sheet_name = "erdos_renyi"
    # sheet_name = "bhoslib"
    # sheet_name = "regular"
    col_name = "Pire cas"

    # Parallèle à "graphics/differentiel/{sheet_name}"
    output_folder = f"graphics/pire_cas/{sheet_name}"

    # Appel de la fonction
    plot_like_rapport(
        file_name=file_name,
        sheet_name=sheet_name,
        original_col=col_name,
        output_folder=output_folder
    )
