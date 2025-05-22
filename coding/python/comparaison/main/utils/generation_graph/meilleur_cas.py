from .plot_rapport import plot_like_rapport

if __name__ == "__main__":
    file_name = "performances.xlsx"

    sheets = [
        "tree",
        "barabasi_albert",
        "erdos_renyi",
        "bhoslib",
        "HoG",
        "regular",
        "kernels_hog"
    ]

    col_name = "Meilleur cas"

    for sheet in sheets:
        output_folder = f"out/graphics/{col_name}/{sheet}"

        plot_like_rapport(
            file_name=file_name,
            sheet_name=sheet,
            original_col=col_name,
            output_folder=output_folder
        )
