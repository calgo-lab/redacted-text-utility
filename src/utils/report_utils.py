from pathlib import Path
from typing import Any, Dict, List, Union

from pandas import DataFrame

import re

import numpy as np
import pandas as pd


class ReportUtils:
    """
    Utility class for parsing and preparing report stats based on 
    classification reports text files.
    """

    @staticmethod
    def get_classification_report_dict(report_file_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parses a classification report from a text file and returns a 
        dictionary with metrics for each class or statistic.
        
        :param report_file_path: Path to the classification report text file.
        :return: Dictionary with class or statistic name as keys and their 
                 metrics as values.
        """
        
        metrics_dict: Dict[str, Dict[str, Any]] = dict()
        lines: List[str] = list()
        with (report_file_path.open('r', encoding='utf-8')) as file_reader:
            lines = file_reader.readlines()
        start_processing = False
        for line in lines:
            line = line.strip()
            if "precision" in line and "recall" in line and "f1-score" in line and "support" in line:
                start_processing = True
                continue
            if start_processing:
                if line.startswith("accuracy"):
                    match = re.match(r"accuracy\s+(\d+\.\d+)\s+(\d+)$", line)
                    if match:
                        accuracy_score, support = match.groups()
                        metrics_dict["micro avg"] = {
                            "precision": float(accuracy_score),
                            "recall": float(accuracy_score),
                            "f1-score": float(accuracy_score),
                            "support": int(support)
                        }
                else:
                    match = re.match(r"(\S+(?:\s+\S+)*)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)$", line)
                    if match:
                        class_name, precision, recall, f1_score, support = match.groups()
                        metrics_dict[class_name] = {
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1-score": float(f1_score),
                            "support": int(support)
                        }
        return metrics_dict
    
    @staticmethod
    def get_metrics_with_hierarchy(metrics_dir: Path) -> Dict[str, Any]:
        """
        Parses classification report text files in a hierarchical dictionary 
        structure.
        
        :param metrics_dir: Path to the root directory containing 
                            classification report text files.
        :return: Nested dictionary with hierarchy based on directory 
                 structure.
        """
        
        metrics_files: List[Path] = list(metrics_dir.glob("**/*.txt"))
        
        def build_nested_dict(files: List[Path], 
                              metrics_dir: Path, 
                              current_depth: int = 0) -> Dict[str, Any]:
            """
            Recursively build nested dictionary based on path depth of files.
            
            For each file, calculate its depth relative to metrics_dir.
            If all files at this level have the same depth to leaf, parse them.
            Otherwise, group by next path component and recurse.

            :param files: List of file paths to process.
            :param metrics_dir: Root metrics directory path.
            :param current_depth: Current depth in the recursion.
            :return: Nested dictionary of parsed classification reports.
            """
            if not files:
                return dict()
            
            file_depths = [
                len(file_path.relative_to(metrics_dir).parts) - current_depth 
                for file_path in files
            ]
            max_depth = max(file_depths)
            
            if max_depth == 1:
                result_dict = dict()
                for file_path in files:
                    fold: str = file_path.name.split(".")[0]
                    result_dict[fold] = ReportUtils.get_classification_report_dict(file_path)
                return dict(sorted(result_dict.items()))
            
            grouped_files: Dict[str, List[Path]] = dict()
            for file_path in files:
                relative_parts = file_path.relative_to(metrics_dir).parts
                key = relative_parts[current_depth]
                if key not in grouped_files:
                    grouped_files[key] = list()
                grouped_files[key].append(file_path)
            
            result_dict = dict()
            for key in sorted(grouped_files.keys()):
                result_dict[key] = build_nested_dict(
                    grouped_files[key], 
                    metrics_dir, 
                    current_depth + 1
                )
            return result_dict
        
        return build_nested_dict(metrics_files, metrics_dir)
    
    @staticmethod
    def get_performance_metrics_with_hierarchy(metrics_dir: Path) -> Dict[str, Any]:
        """
        Collect all metrics (prec./rec./f1./support) for all folds, grouped by 
        class or statistic.
        
        :param metrics_dir: Path to the root directory containing 
                            classification report text files.
        :return: Nested dictionary with hierarchy based on directory structure.
        """
        
        metrics_with_hierarchy = ReportUtils.get_metrics_with_hierarchy(metrics_dir)
        
        def build_class_or_stat_metrics_dict(fold_wise_dict) -> Dict[str, List[Any]]:
            """
            Convert fold-wise classification reports into class/stat metrics lists.

            :param fold_wise_dict: Dictionary with fold names as keys and 
                                   classification report dicts as values.
            :return: Dictionary with class/stat names as keys and lists of 
                     metrics lists as values.
            """
            if not fold_wise_dict or not any(isinstance(v, dict) for v in fold_wise_dict.values()):
                return dict()
            
            first_fold = next(iter(fold_wise_dict.values()))
            class_or_stat_names = list(first_fold.keys())
            
            class_metrics_dict: Dict[str, List[List[Union[float, int]]]] = dict()
            
            for class_or_stat in class_or_stat_names:
                metrics_list: List[List[Union[float, int]]] = list()
                
                for fold_key in sorted(fold_wise_dict.keys()):
                    fold_data = fold_wise_dict[fold_key]
                    if class_or_stat in fold_data:
                        metrics = fold_data[class_or_stat]
                        metrics_list.append([
                            round(metrics['precision'], 4),
                            round(metrics['recall'], 4),
                            round(metrics['f1-score'], 4),
                            metrics['support']
                        ])
                
                class_metrics_dict[class_or_stat] = metrics_list
            
            return class_metrics_dict
        
        def build_hierarchy(data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
            """
            Recursively build the nested structure based on hierarchy depth.
            Stops when reaching fold-level data.

            :param data: Current level dictionary.
            :param depth: Current depth in the hierarchy.
            :return: Nested dictionary or class metrics dictionary.
            """
            if not data:
                return dict()
            
            first_value = next(iter(data.values()), None)
            if first_value is None:
                return dict()
            
            if isinstance(first_value, dict) and 'precision' in first_value:
                return dict()
            
            if isinstance(first_value, dict):
                inner_first = next(iter(first_value.values()), None) if first_value else None
                if isinstance(inner_first, dict) and 'precision' in inner_first:
                    return build_class_or_stat_metrics_dict(data)
            
            result_dict = dict()
            for key in sorted(data.keys()):
                result_dict[key] = build_hierarchy(data[key], depth + 1)
            
            return result_dict
        
        final_result = dict()
        for outer_key in sorted(metrics_with_hierarchy.keys()):
            final_result[outer_key] = build_hierarchy(metrics_with_hierarchy[outer_key])
        
        return final_result
    
    @staticmethod
    def get_performance_metrics_summary_table(metrics_dir: Path,
                                              row_dimension: str,
                                              row_values_order: List[str],
                                              column_dimension: str,
                                              column_values_order: List[str],
                                              class_or_stat: str,
                                              hierarchy: List[str],
                                              fixed_dimensions: Dict[str, str],
                                              model_name_aliases: Dict[str, str],
                                              redaction_strategy_aliases: Dict[str, str]) -> DataFrame:
        """
        Prepare a summary table with flexible dimensions for performance metrics.
        
        :param metrics_dir: Path to the root directory containing classification report text files.
        :param row_dimension: Dimension for rows (must be in hierarchy).
        :param row_values_order: Specific order for row values.
        :param column_dimension: Dimension for columns (must be in hierarchy).
        :param column_values_order: Specific order for column values.
        :param class_or_stat: The class name or statistic (e.g., 'macro avg', 'weighted avg', or specific class label).
        :param hierarchy: List of dimensions in order (e.g., 
               ['model_name', 'redaction_strategy', 'percentile', 'metric_type']).
        :param fixed_dimensions: Dict of dimensions to hold fixed (e.g., 
               {'percentile': '0-100', 'metric_type': 'f1-score'}).
        :param model_name_aliases: Dict mapping model names to their aliases.
        :param redaction_strategy_aliases: Dict mapping redaction strategies to their aliases.
        :return: Pandas DataFrame with the specified dimensions showing mean ± std.
        """
        if hierarchy is None:
            raise ValueError("hierarchy parameter must be provided.")
        
        if fixed_dimensions is None:
            raise ValueError("fixed_dimensions parameter must be provided.")
        
        if row_dimension not in hierarchy or column_dimension not in hierarchy:
            raise ValueError(f"row_dimension and column_dimension must be in hierarchy: {hierarchy}")
        
        metrics_hierarchy = ReportUtils.get_performance_metrics_with_hierarchy(metrics_dir)
        
        def extract_all_keys_at_level(data: Dict[str, Any], 
                                      level_index: int, 
                                      current_level: int = 0) -> List[str]:
            
            """
            Extract all unique keys at a specific hierarchy level.
            
            :param data: Current level dictionary.
            :param level_index: Target level index to extract keys from.
            :param current_level: Current level index during recursion.
            :return: Sorted list of unique keys at the specified level.
            """

            if current_level == level_index:
                return sorted(list(data.keys()))
            
            all_keys = set()
            for value in data.values():
                if isinstance(value, dict):
                    all_keys.update(extract_all_keys_at_level(value, level_index, current_level + 1))
            
            return sorted(list(all_keys))
        
        def get_metric_value(class_stat_metrics: List[List[Union[float, int]]], metric_type: str) -> str:
            """
            Extract mean ± std from metrics list for a specific metric type.
            
            :param class_stat_metrics: List of metrics lists for each fold.
            :param metric_type: Metric type to extract ('precision', 'recall', 'f1-score', 'support').
            :return: Formatted string of mean ± std.
            """
            if not class_stat_metrics or len(class_stat_metrics) == 0:
                return "N/A"
            
            metric_index = {
                'precision': 0, 
                'recall': 1, 
                'f1-score': 2, 
                'support': 3
            }[metric_type]
            
            values = [fold[metric_index] for fold in class_stat_metrics]
            
            if metric_type == 'support':
                print(values)
                mean_val = int(np.mean(values))
                std_val = int(np.std(values))
                return f"{mean_val} ± {std_val}"
            else:
                mean_val = round(np.mean(values), 4)
                std_val = round(np.std(values), 4)
                return f"{mean_val:.4f} ± {std_val:.4f}"
        
        structural_dims = [d for d in hierarchy if d != 'metric_type']
        
        dimension_values = dict()
        for dim in hierarchy:
            if dim in fixed_dimensions:
                dimension_values[dim] = [fixed_dimensions[dim]]
            elif dim == 'metric_type':
                dimension_values[dim] = ['precision', 'recall', 'f1-score', 'support']
            else:
                dim_index = structural_dims.index(dim)
                dimension_values[dim] = extract_all_keys_at_level(metrics_hierarchy, dim_index)
        
        row_values = dimension_values.get(row_dimension, list())
        if row_values_order:
            row_values = [val for val in row_values_order if val in row_values]
        col_values = dimension_values.get(column_dimension, list())
        if column_values_order:
            col_values = [val for val in column_values_order if val in col_values]
        
        table_data = list()
        for row_val in row_values:
            
            if row_dimension == 'model_name':
                row_entry = {row_dimension: model_name_aliases.get(row_val, row_val)}
            elif row_dimension == 'redaction_strategy':
                row_entry = {row_dimension: redaction_strategy_aliases.get(row_val, row_val)}
            else:
                row_entry = {row_dimension: row_val}
            
            for col_val in col_values:
                dim_values = dict()
                for dim in hierarchy:
                    if dim in fixed_dimensions:
                        dim_values[dim] = fixed_dimensions[dim]
                    elif dim == row_dimension:
                        dim_values[dim] = row_val
                    elif dim == column_dimension:
                        dim_values[dim] = col_val
                    else:
                        dim_values[dim] = dimension_values[dim][0] if dimension_values[dim] else None
                
                metric_type = dim_values['metric_type']
                
                nav_path = {dim: dim_values[dim] for dim in structural_dims if dim_values[dim] is not None}
                
                current_data = metrics_hierarchy
                valid_path = True
                
                for dim in structural_dims:
                    if dim in nav_path and nav_path[dim] in current_data:
                        current_data = current_data[nav_path[dim]]
                    elif dim in nav_path:
                        valid_path = False
                        break
                
                if valid_path and class_or_stat in current_data:
                    metrics_list = current_data[class_or_stat]
                    cell_value = get_metric_value(metrics_list, metric_type)
                else:
                    cell_value = "N/A"
                
                if column_dimension == 'model_name':
                    row_entry[model_name_aliases.get(col_val, col_val)] = cell_value
                elif column_dimension == 'redaction_strategy':
                    row_entry[redaction_strategy_aliases.get(col_val, col_val)] = cell_value
                else:
                    row_entry[col_val] = cell_value
            
            table_data.append(row_entry)
        
        df = pd.DataFrame(table_data)
        df = df.set_index(row_dimension)
        return df