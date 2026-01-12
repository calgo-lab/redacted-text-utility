from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

class PlotUtils:
    """
    Utility class for plotting figures and charts.
    """

    @staticmethod
    def plot_num_tokens_distribution(data_handler: Any, 
                                     filename: str, 
                                     output_path: Path, 
                                     show_logs: bool = True) -> None:
        """
        Plots the num_tokens distribution using data_handler.get_num_tokens_df() function.

        :param data_handler: An instance of the data handler with get_num_tokens_df method.
        :param filename: The filename to load the data from.
        :param output_path: The path to save the output plot.

        :return: None
        """

        num_tokens_df = data_handler.get_num_tokens_df(filename)
        if filename.endswith('.parquet'):
            filename = filename[:-8]
        figure_file_path = output_path / f"{filename}_num_tokens_distribution.jpg"
        
        plt.hist(num_tokens_df['num_tokens'])
        plt.xlabel("Token count")
        plt.ylabel("Frequency")
        plt.title("Distribution of Token Counts")
        plt.savefig(figure_file_path)
        plt.close()
        
        if show_logs:
            print(f"figure_path: {figure_file_path}")