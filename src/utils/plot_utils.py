from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

class PlotUtils:
    """
    Utility class for plotting figures and charts.
    """

    @staticmethod
    def plot_num_tokens_distribution(num_tokens_df: pd.DataFrame, 
                                     figure_file_path: Path, 
                                     show_logs: bool = True, 
                                     plot_config: Dict[str, Any] = None) -> None:
        """
        Plots the num_tokens distribution using the provided num_tokens_df DataFrame.

        :param num_tokens_df: A DataFrame containing num_tokens column.
        :param figure_file_path: The path to save the output plot.
        :param show_logs: Whether to print log messages.
        :param plot_config: Additional configuration for the plot.

        :return: None
        """

        bins: int = plot_config.get('bins') if plot_config and 'bins' in plot_config else None
        show_grid: bool = plot_config.get('show_grid') if plot_config and 'show_grid' in plot_config else False

        plt.hist(num_tokens_df['num_tokens'], bins=bins)
        plt.grid(show_grid)
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.title("Distribution of Token Counts")
        plt.savefig(figure_file_path)
        plt.close()
        
        if show_logs:
            print(f"figure_file_path: {figure_file_path}")