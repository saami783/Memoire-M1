from .plot_rapport import plot_like_rapport

if __name__ == "__main__":
    file_name = "rapport approximation differentiel.xlsx"
    sheet_name = "bhoslib"
    col_name = "Moyen cas"

    output_folder = f"graphics/moyen_cas/{sheet_name}"

    plot_like_rapport(
        file_name=file_name,
        sheet_name=sheet_name,
        original_col=col_name,
        output_folder=output_folder
    )
