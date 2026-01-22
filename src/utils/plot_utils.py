from pathlib import Path
from typing import Any, Dict

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

import re

import matplotlib.pyplot as plt
import numpy as np
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

    @staticmethod
    def parse_mean_std(cell: str):
        """
        Parse the string with mean and std and return as separate values.
        Example: '0.8900 ± 0.0123' -> (0.89, 0.0123)

        :param cell: string in the format 'mean ± std'
        :return: tuple of (mean, std) as floats
        """
        mean, std = re.split(r"\s*±\s*", cell)
        return float(mean), float(std)
    
    @staticmethod
    def expand_mean_std_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts columns like '0-100' with 'mean ± std'
        into numeric columns: '0-100_mean', '0-100_std'

        :param df: DataFrame with columns containing 'mean ± std' strings
        :return: DataFrame with expanded mean and std columns
        """
        out = dict()
        for col in df.columns:
            means, stds = list(), list()
            for cell in df[col]:
                mean, std = PlotUtils.parse_mean_std(cell)
                means.append(mean)
                stds.append(std)

            out[f"{col}_mean"] = means
            out[f"{col}_std"] = stds

        return pd.DataFrame(out, index=df.index)
    
    @staticmethod
    def plot_echr_tc_classifier_performance_type_one(model_dfs,
                                                     model_names,
                                                     percentile_labels,
                                                     entity_counts,
                                                     redaction_strategies,
                                                     model_bg_colors,
                                                     strategy_styles,
                                                     figsize=(20, 5)) -> plt.Figure:
        n_strategies = len(redaction_strategies)
        strategy_offsets = np.linspace(-0.2, 0.2, n_strategies)

        fig, ax = plt.subplots(figsize=figsize)

        x = 0
        region_boundaries = list()
        for p_idx, p_label in enumerate(percentile_labels):
            if p_idx > 0:
                ax.axvline(x - 0.5, 
                           linestyle="--", 
                           linewidth=1, 
                           color="gray")

            region_start = x
            for m_idx, model_name in enumerate(model_names):
                ax.axvspan(
                    x - 0.5,
                    x + 0.5,
                    color=model_bg_colors[model_name],
                    alpha=0.18,
                    zorder=0
                )

                model_df = PlotUtils.expand_mean_std_columns(model_dfs[m_idx])

                for strategy in redaction_strategies:
                    style = strategy_styles[strategy]

                    mean = model_df.loc[strategy, f"{p_label}_mean"]
                    std = model_df.loc[strategy, f"{p_label}_std"]

                    x_offset = strategy_offsets[redaction_strategies.index(strategy)]

                    ax.errorbar(
                        x + x_offset,
                        mean,
                        yerr=std,
                        fmt=style["marker"],
                        color=style["color"],
                        ecolor=style["color"],
                        elinewidth=1.2,
                        capsize=3,
                        markersize=5,
                        linewidth=1,
                        zorder=3
                    )

                x += 1

            region_end = x - 1
            region_boundaries.append((region_start + region_end) / 2)


        ax.set_xticks(region_boundaries)
        ax.set_xticklabels([
            f"{p}\n(n={entity_counts[p]})" for p in percentile_labels
        ])

        ax.set_ylim(0.70, 0.95)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.set_yticks(np.arange(0.70, 0.96, 0.05))
        ax.set_ylabel("Macro F1-score")

        ax.set_xlim(-0.5, x - 0.5)

        ax.set_title("Macro F1-score vs. Entity Count Percentiles")

        ax.grid(
            axis="y",
            which="major",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7
        )

        ax.grid(
            axis="y",
            which="minor",
            linestyle=":",
            linewidth=0.6,
            alpha=0.35
        )

        ax.tick_params(axis="y", which="minor", length=0)

        legend_elements = [
            Line2D(
                [0], [0],
                marker=style["marker"],
                color=style["color"],
                label=strategy,
                linestyle="-"
            )
            for strategy, style in strategy_styles.items()
        ]

        strategy_legend = ax.legend(
            handles=legend_elements,
            title="Redaction Strategy",
            loc="upper right",
            frameon=True,
            ncol=1
        )

        model_legend_elements = [
            Patch(
                facecolor=model_bg_colors[model],
                edgecolor="none",
                alpha=0.4,
                label=model
            )
            for model in model_names
        ]

        model_legend = ax.legend(
            handles=model_legend_elements,
            title="Model",
            loc="lower left",
            frameon=True
        )

        ax.add_artist(strategy_legend)

        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_echr_tc_classifier_performance_type_two(model_df,
                                                     model_name: str,
                                                     strategy_a: str,
                                                     strategy_b: str,
                                                     percentile_labels,
                                                     colors,
                                                     figsize=(5, 3)) -> plt.Figure:
        """
        Plot performance comparison of a model for two specified strategies.

        :param model_df: performance dataframe of the model with strategies
                         as rows and percentiles as columns for performance 
                         values in "mean ± std" format.
        :param model_name: model name alias for the title.
        :param strategy_a: usually "No Redaction"
        :param strategy_b: one of the redaction strategies 
                           [
                               "Semantic Label Masking", 
                               "Random Masking", 
                               "Generic Masking"
                           ]
        :param percentile_labels: list of percentile labels
        :param colors: dict mapping strategy names to colors
        :param figsize: figure size
        """

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(percentile_labels))
        x0 = x[0]
        x_rest = x[1:]

        offset = 0.12
        x_offsets = {
            strategy_a: x0 - offset,
            strategy_b: x0 + offset,
        }

        model_df = PlotUtils.expand_mean_std_columns(model_df)
        for strategy in [strategy_a, strategy_b]:
            ax.errorbar(
                x_offsets[strategy],
                model_df.loc[strategy, "0-100_mean"],
                yerr=model_df.loc[strategy, "0-100_std"],
                fmt="o",
                color=colors[strategy],
                capsize=2,
                markersize=3,
                linewidth=1,
                zorder=3
            )

        for strategy in [strategy_a, strategy_b]:
            means = [
                model_df.loc[strategy, f"{p}_mean"]
                for p in percentile_labels[1:]
            ]
            stds = [
                model_df.loc[strategy, f"{p}_std"]
                for p in percentile_labels[1:]
            ]

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(
                x_rest,
                means,
                marker="o",
                markersize=2.5,
                linewidth=1,
                color=colors[strategy],
                label=strategy
            )

            ax.fill_between(
                x_rest,
                means - stds,
                means + stds,
                color=colors[strategy],
                alpha=0.30,
                linewidth=0
            )

        ax.set_xlim(-0.4, len(percentile_labels) - 0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(percentile_labels, fontsize=7)
        ax.set_ylabel("Macro F1-score", fontsize=7)
        ax.set_title(f"{model_name}: {strategy_a} vs {strategy_b}", fontsize=7)

        y_text = -0.15

        ax.text(
            x[0],
            y_text,
            "unsegmented",
            ha="center",
            va="top",
            fontsize=7,
            transform=ax.get_xaxis_transform()
        )

        x_center_segmented = (x[1] + x[-1]) / 2

        ax.text(
            x_center_segmented,
            y_text,
            "test samples segmented by entity count percentile",
            ha="center",
            va="top",
            fontsize=7,
            transform=ax.get_xaxis_transform()
        )

        divider_x = (x[0] + x[1]) / 2

        ax.vlines(
            divider_x,
            ymin=-0.19,
            ymax=1.0,
            colors="gray",
            linestyles=(0, (4, 3)),
            linewidth=1,
            alpha=0.8,
            transform=ax.get_xaxis_transform(),
            clip_on=False
        )

        ax.set_ylim(0.70, 0.95)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))

        ax.grid(axis="y", which="major", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.35)
        ax.tick_params(axis="y", which="minor", length=0, labelsize=7)

        ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=7
        )

        plt.tight_layout()
        return fig, ax
