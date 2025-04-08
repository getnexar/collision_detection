import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc
import seaborn as sns
import pandas as pd

def plot_organized_evaluation(y_true, y_prob, threshold=0.5, population_config=None, title=None):
    """
    Displays organized plots for model performance evaluation:
    Row 1: Confusion matrices (standard and normalized)
    Row 2: Precision-recall curve and precision/recall vs confidence threshold
    Row 3: Comparison of metrics before and after population adjustment and ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities (output from predict_proba)
    threshold : float
        Classification threshold (default: 0.5)
    population_config : dict, optional
        Dictionary with 'true_pos_population' and 'true_neg_population'
    title : str, optional
        Main title for the plot
    """
    # Ensure y_true is a numpy array
    y_true = np.array(y_true)
    
    # Handle different formats of prediction probabilities
    if len(y_prob.shape) == 1:
        # If y_prob is a vector of probabilities for the positive class only
        probs_positive = y_prob
        probs_negative = 1 - y_prob
        y_prob_matrix = np.column_stack((probs_negative, probs_positive))
    else:
        # If y_prob is a matrix of probabilities for both classes
        y_prob_matrix = y_prob
        probs_positive = y_prob_matrix[:, 1]
    
    # Create binary predictions based on threshold
    y_pred = (probs_positive >= threshold).astype(int)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 15))
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    
    # ------------------------ Row 1: Confusion Matrices -------------------------
    # Standard confusion matrix
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Class 0', 'Class 1']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_title(f'Confusion Matrix (threshold={threshold})')
    
    # Normalized confusion matrix
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=plt.cm.Blues, 
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    ax2.set_title(f'Normalized Confusion Matrix (threshold={threshold})')
    
    # Calculate performance metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"
    
    # Add performance metrics as text
    ax1.text(0.5, -0.2, metrics_text, transform=ax1.transAxes, 
             horizontalalignment='center', verticalalignment='center', fontsize=10)
    
    # ------------------------ Row 2: PR Curves and Threshold Graph -------------------------
    # Precision-Recall Curve
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    
    # For class 1 (positive class)
    precision_1, recall_1, thresholds_1 = precision_recall_curve(y_true, probs_positive)
    avg_precision_1 = average_precision_score(y_true, probs_positive)
    ax3.plot(recall_1, precision_1, lw=2, label=f'Class 1 (AP = {avg_precision_1:.4f})')
    
    # For class 0 (negative class - need to invert labels)
    precision_0, recall_0, thresholds_0 = precision_recall_curve(1 - y_true, 1 - probs_positive)
    avg_precision_0 = average_precision_score(1 - y_true, 1 - probs_positive)
    ax3.plot(recall_0, precision_0, lw=2, label=f'Class 0 (AP = {avg_precision_0:.4f})')
    
    # Add iso-f1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax3.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax3.annotate(f'f1={f_score:.1f}', xy=(0.8, y[45] + 0.02))
    
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    # Precision & Recall vs Confidence Threshold graph
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    
    # Display options for metrics
    show_metrics = "recall"  # Options: "both", "precision", "recall"
    
    # Create test threshold values
    thresholds_test = np.linspace(0.1, 0.9, 9)
    precisions = []
    recalls = []
    
    # Calculate precision and recall for each threshold
    for thresh in thresholds_test:
        y_pred_t = (probs_positive >= thresh).astype(int)
        
        # Calculate metrics
        prec = precision_score(y_true, y_pred_t, zero_division=0)
        rec = recall_score(y_true, y_pred_t, zero_division=0)
        
        precisions.append(prec)
        recalls.append(rec)
    
    # Plot values against threshold based on show_metrics option
    if show_metrics == "both" or show_metrics == "precision":
        ax4.plot(thresholds_test, precisions, 'b-', label='Precision')
        
    if show_metrics == "both" or show_metrics == "recall":
        ax4.plot(thresholds_test, recalls, 'r-', label='Recall')
    
    # Mark current threshold
    ax4.axvline(x=threshold, color='g', linestyle='--', label=f'Current Threshold ({threshold})')
    
    # Update title based on what's shown
    if show_metrics == "both":
        title = 'Precision & Recall vs. Confidence Threshold'
    elif show_metrics == "precision":
        title = 'Precision vs. Confidence Threshold'
    else:  # show_metrics == "recall"
        title = 'Recall vs. Confidence Threshold'
    
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('Confidence Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title(title)
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    # ------------------------ Row 3: Split into two graphs -------------------------
    # Create two subplots in the third row
    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))
    
    # Left graph: Bar charts for adjusted metrics
    if population_config is not None:
        true_pos_population = population_config.get('true_pos_population', 2000)
        true_neg_population = population_config.get('true_neg_population', 5000)
        
        # Calculate adjusted metrics
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Scale FP to population size
        fp_population = fp_rate * true_neg_population
        
        # Calculate adjusted precision
        tp_population = recall * true_pos_population
        precision_adj = tp_population / (tp_population + fp_population) if (tp_population + fp_population) > 0 else 0
        
        # Calculate adjusted F1
        f1_adj = 2 * (precision_adj * recall) / (precision_adj + recall) if (precision_adj + recall) > 0 else 0
        
        # Visualization
        labels = ['Recall', 'Precision', 'F1 Score']
        before_adjustment = [recall, precision, f1]
        after_adjustment = [recall, precision_adj, f1_adj]  # recall remains the same
        
        x_pos = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax5.bar(x_pos - width/2, before_adjustment, width, label='Before Adjustment')
        bars2 = ax5.bar(x_pos + width/2, after_adjustment, width, label='After Adjustment')
        
        # Add value labels
        for bars in [bars1, bars2]:
            ax5.bar_label(bars, padding=3, fmt='%.2f')
        
        ax5.set_ylabel('Scores')
        ax5.set_title('Metrics Before and After Adjustment')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(labels)
        ax5.legend()
        ax5.set_ylim(0, 1)
        
        # Add population information
        population_text = (f"Population Info:\n"
                          f"True Positive Population: {true_pos_population}\n"
                          f"True Negative Population: {true_neg_population}")
        ax5.text(0.02, 0.05, population_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', alpha=0.1))
    else:
        ax5.text(0.5, 0.5, "Population configuration not provided\nCannot compute adjusted metrics",
                transform=ax5.transAxes, horizontalalignment='center',
                verticalalignment='center', fontsize=14)
    
    # Right graph: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs_positive)
    roc_auc = auc(fpr, tpr)
    
    ax6.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax6.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark current threshold on ROC curve if possible
    if len(fpr) > 1 and len(tpr) > 1:
        # Find closest point to current threshold's FPR and TPR
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        current_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax6.plot(current_fpr, current_tpr, 'go', markersize=8, label=f'Threshold = {threshold}')
    
    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax6.legend(loc='lower right')
    ax6.grid(alpha=0.3)
    
    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, top=0.92 if title else 0.95)
    
    return fig

class SegmentAnalyzer:
    """
    Class for analyzing model performance across multiple segments
    with integrated configuration for thresholds and population data.
    """
    
    def __init__(self, df, segment_column):
        """
        Initialize the segment analyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing prediction data
        segment_column : str
            Column name to use for segmentation
        """
        self.df = df
        self.segment_column = segment_column
        self.segments_config = []
        self.results = {}
    
    def add_segment(self, min_value=None, max_value=None, name=None, 
                   true_pos_population=2000, true_neg_population=5000):
        """
        Add a segment configuration.
        
        Parameters:
        -----------
        min_value : float, optional
            Minimum value for the segment (inclusive)
        max_value : float, optional
            Maximum value for the segment (exclusive)
        name : str, optional
            Custom name for the segment (generated automatically if not provided)
        true_pos_population : int
            True positive population size for adjustments
        true_neg_population : int
            True negative population size for adjustments
        """
        segment_config = {
            'min_value': min_value,
            'max_value': max_value,
            'name': name,
            'population': {
                'true_pos_population': true_pos_population,
                'true_neg_population': true_neg_population
            }
        }
        
        # Generate name if not provided
        if name is None:
            if min_value is None and max_value is not None:
                segment_config['name'] = f"{self.segment_column} < {max_value}"
            elif min_value is not None and max_value is None:
                segment_config['name'] = f"{self.segment_column} ≥ {min_value}"
            elif min_value is not None and max_value is not None:
                segment_config['name'] = f"{min_value} ≤ {self.segment_column} < {max_value}"
            else:
                segment_config['name'] = "All Data"
        
        self.segments_config.append(segment_config)
        return self
    
    def add_segments_from_thresholds(self, thresholds, population_values=None):
        """
        Add multiple segments from a list of thresholds.
        
        Parameters:
        -----------
        thresholds : list
            List of threshold values to create segments
        population_values : list of tuples, optional
            List of (true_pos_population, true_neg_population) tuples for each segment
            If not provided, defaults to (2000, 5000) for all segments
        """
        if population_values is None:
            population_values = [(2000, 5000)] * (len(thresholds) + 1)
        
        # Ensure we have enough population values
        if len(population_values) < len(thresholds) + 1:
            # Extend with default values if needed
            population_values.extend([(2000, 5000)] * (len(thresholds) + 1 - len(population_values)))
        
        # Sort thresholds
        sorted_thresholds = sorted(thresholds)
        
        # First segment: values less than first threshold
        self.add_segment(
            min_value=None, 
            max_value=sorted_thresholds[0],
            true_pos_population=population_values[0][0],
            true_neg_population=population_values[0][1]
        )
        
        # Middle segments
        for i in range(len(sorted_thresholds) - 1):
            self.add_segment(
                min_value=sorted_thresholds[i],
                max_value=sorted_thresholds[i + 1],
                true_pos_population=population_values[i + 1][0],
                true_neg_population=population_values[i + 1][1]
            )
        
        # Last segment: values greater than last threshold
        self.add_segment(
            min_value=sorted_thresholds[-1],
            max_value=None,
            true_pos_population=population_values[-1][0],
            true_neg_population=population_values[-1][1]
        )
        
        return self
    
    def analyze(self, y_true_col='y_true', y_prob_cols=None, threshold=0.5, show_plots=True):
        """
        Analyze all configured segments.
        
        Parameters:
        -----------
        y_true_col : str
            Column name for true labels
        y_prob_cols : str or list
            Column name(s) for prediction probabilities. 
            Can be a single column name (for positive class probs) or 
            a list of two column names [negative_class, positive_class]
        threshold : float
            Classification threshold
        show_plots : bool
            Whether to display plots during analysis
            
        Returns:
        --------
        dict
            Dictionary containing segment DataFrames and their metrics
        """
        if y_prob_cols is None:
            # Try to guess the probability columns
            if 'y_prob_1' in self.df.columns and 'y_prob_0' in self.df.columns:
                y_prob_cols = ['y_prob_0', 'y_prob_1']
            elif 'y_prob' in self.df.columns:
                y_prob_cols = 'y_prob'
            else:
                raise ValueError("Could not determine probability columns. Please specify y_prob_cols.")
        
        # Process each segment
        for segment_config in self.segments_config:
            min_value = segment_config['min_value']
            max_value = segment_config['max_value']
            name = segment_config['name']
            
            # Filter data for this segment
            if min_value is None and max_value is None:
                segment_df = self.df.copy()  # All data
            elif min_value is None:
                segment_df = self.df.loc[self.df[self.segment_column] < max_value]
            elif max_value is None:
                segment_df = self.df.loc[self.df[self.segment_column] >= min_value]
            else:
                segment_df = self.df.loc[(self.df[self.segment_column] >= min_value) & 
                                       (self.df[self.segment_column] < max_value)]
            
            # Store segment data
            self.results[name] = {
                'data': segment_df,
                'config': segment_config,
                'metrics': None
                
            }
            
            # Only analyze if segment has data
            if len(segment_df) > 0 and y_true_col in segment_df.columns:
                print(f"\n=== Analyzing Segment: {name} ===")
                print(f"Segment size: {len(segment_df)} samples")
                
                # Prepare prediction data
                y_true = segment_df[y_true_col].values
                
                if isinstance(y_prob_cols, list) and len(y_prob_cols) == 2:
                    y_probs = segment_df[y_prob_cols].values
                else:
                    y_probs = segment_df[y_prob_cols].values
                
                # Create binary predictions if needed
                if isinstance(y_prob_cols, list) and len(y_prob_cols) == 2:
                    y_pred = (y_probs[:, 1] >= threshold).astype(int)
                else:
                    y_pred = (y_probs >= threshold).astype(int)
                
                # Create and show plots
                if show_plots:
                    plot_organized_evaluation(
                        y_true,
                        y_probs,
                        threshold=threshold,
                        population_config=segment_config['population'],
                        title=f'Analysis for Segment: {name}'
                    )
                    plt.show()
                
                # Calculate metrics
                cm = confusion_matrix(y_true, y_pred)
                try:
                    tn, fp, fn, tp = cm.ravel()
                    
                    recall = recall_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    # Calculate ROC AUC
                    if isinstance(y_prob_cols, list) and len(y_prob_cols) == 2:
                        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
                        roc_auc = auc(fpr, tpr)
                    else:
                        fpr, tpr, _ = roc_curve(y_true, y_probs)
                        roc_auc = auc(fpr, tpr)
                    
                    # Calculate adjusted metrics
                    true_pos_population = segment_config['population']['true_pos_population']
                    true_neg_population = segment_config['population']['true_neg_population']
                    
                    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fp_population = fp_rate * true_neg_population
                    
                    tp_population = recall * true_pos_population
                    precision_adj = tp_population / (tp_population + fp_population) if (tp_population + fp_population) > 0 else 0
                    
                    f1_adj = 2 * (precision_adj * recall) / (precision_adj + recall) if (precision_adj + recall) > 0 else 0
                    
                    metrics = {
                        "Samples": len(segment_df),
                        "TN": tn,
                        "FP": fp,
                        "FN": fn, 
                        "TP": tp,
                        "Recall": recall,
                        "Precision": precision,
                        "F1": f1,
                        "ROC_AUC": roc_auc,
                        "Precision_adj": precision_adj,
                        "F1_adj": f1_adj,
                        "Pop_TP": true_pos_population,
                        "Pop_TN": true_neg_population
                    }
                    
                    self.results[name]['metrics'] = metrics
                    
                    print(f"\nSegment: {name}")
                    print(f"True Positive Population: {true_pos_population}")
                    print(f"True Negative Population: {true_neg_population}")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            print(f"{k}: {v:.4f}")
                        else:
                            print(f"{k}: {v}")
                except Exception as e:
                    print(f"Error calculating metrics for segment {name}: {e}")
            else:
                print(f"Segment '{name}' has no data or missing true labels column.")
        
        return self.results
    
    def summary(self):
        """
        Create a summary DataFrame of all segment metrics.
        
        Returns:
        --------
        pandas.DataFrame
            Summary of metrics across all segments
        """
        metrics_list = []
        
        for name, result in self.results.items():
            if result['metrics'] is not None:
                metrics = result['metrics'].copy()
                metrics['Segment'] = name
                metrics_list.append(metrics)
        
        if not metrics_list:
            return pd.DataFrame()
        
        summary_df = pd.DataFrame(metrics_list)
        
        # Reorder columns to put Segment first
        cols = ['Segment'] + [col for col in summary_df.columns if col != 'Segment']
        summary_df = summary_df[cols]
        
        return summary_df

    def plot_segment_comparisons(self, y_true_col='y_true', y_prob_cols=None, metric_type='both', figsize=(12, 8)):
        """
        Generate comparative plots across all segments:
        1. Precision & Recall vs Confidence Threshold
        2. ROC curves for all segments
        
        Parameters:
        -----------
        y_true_col : str
            Column name for true labels
        y_prob_cols : str or list
            Column name(s) for prediction probabilities
        metric_type : str
            'precision', 'recall', or 'both' to determine which metrics to plot
        figsize : tuple
            Figure size for plots
        
        Returns:
        --------
        Tuple containing:
            - metrics_fig: Figure object for precision/recall vs threshold
            - roc_fig: Figure object for ROC curves
            - points_data: Dictionary with complete point data for all curves in metrics_fig
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import precision_recall_curve, roc_curve, auc
        
        # Create figure for metrics vs confidence plot
        metrics_fig, ax1 = plt.subplots(figsize=figsize)
        
        # Create figure for ROC curves
        roc_fig, ax_roc = plt.subplots(figsize=figsize)
        
        # Add diagonal reference line for ROC plot
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Color map for multiple segments
        colors = plt.cm.tab10.colors
        
        # Dictionary to store all points data
        points_data = {}
        
        # Process each segment for plots
        for i, (name, result) in enumerate(self.results.items()):
            segment_df = result['data']
            color = colors[i % len(colors)]
            
            # Skip if no data or metrics
            if len(segment_df) == 0 or result['metrics'] is None:
                continue
                
            # Prepare prediction data
            y_true = segment_df[y_true_col].values
            
            if isinstance(y_prob_cols, list) and len(y_prob_cols) == 2:
                y_probs = segment_df[y_prob_cols[1]].values  # Positive class probabilities
            else:
                y_probs = segment_df[y_prob_cols].values
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
            
            # Add precision curve if requested
            if metric_type in ['precision', 'both']:
                # Extend thresholds array to match precision length
                extended_thresholds = np.append(thresholds, 1.0)
                ax1.plot(extended_thresholds, precision, '-', color=color, 
                        label=f'{name} - Precision', linewidth=2)
                
                # Store points data
                points_data[f'{name}_precision'] = {
                    'thresholds': extended_thresholds.tolist(),
                    'values': precision.tolist()
                }
            
            # Add recall curve if requested
            if metric_type in ['recall', 'both']:
                # Extend thresholds array to match recall length
                extended_thresholds = np.append(thresholds, 1.0)
                ax1.plot(extended_thresholds, recall, '--', color=color, 
                        label=f'{name} - Recall', linewidth=2)
                
                # Store points data
                points_data[f'{name}_recall'] = {
                    'thresholds': extended_thresholds.tolist(),
                    'values': recall.tolist()
                }
            
            # Calculate and plot ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{name} (AUC = {roc_auc:.3f})')
            
            # Store ROC points data
            points_data[f'{name}_roc'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist() if len(roc_thresholds) > 0 else []
            }
        
        # Finalize metrics vs threshold plot
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Recall vs. Confidence Threshold Across Segments')
        ax1.set_xlim([0.1, 0.9])
        ax1.set_ylim([0.0, 1.05])
        ax1.grid(alpha=0.3)
        ax1.legend(loc='best')
        
        # Finalize ROC curve plot
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves Across Segments')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.grid(alpha=0.3)
        ax_roc.legend(loc='lower right')
        
        # Return both figures and the points data
        return metrics_fig, roc_fig, points_data

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any

def plot_metrics(data_points_dict: Dict[str, Dict[str, Dict[str, Any]]], 
                 output_filename: str = None,
                 figsize: Tuple[int, int] = (14, 8),
                 dpi: int = 300,
                 plot_background: bool = True,
                 first_model_dashed: bool = True):
    """
    Plot recall and ROC curves for multiple data dictionaries, with colors based on g_max categories.
    
    Args:
        data_points_dict: Dictionary where keys are model names and values are data dictionaries
        output_filename: If provided, save the figure to this filename
        figsize: Size of the figure (width, height)
        dpi: Resolution of the figure
        plot_background: Whether to plot light gray background
    
    Returns:
        Tuple containing:
            - recall_fig: Figure object for recall vs threshold
            - roc_fig: Figure object for ROC curves
    """
    # Define g_max categories and their labels
    g_max_categories = [
        "g_max < 1.0",
        "1.0 ≤ g_max < 1.4",
        "g_max ≥ 1.4",
        #"All Data"
    ]
    
    # Define colors for g_max categories
    colors = {
        "g_max < 1.0": "blue",
        "1.0 ≤ g_max < 1.4": "orange",
        "g_max ≥ 1.4": "green",
        #"All Data": "red"
    }
    
    # Create separate figures with gray background if requested
    recall_fig, ax_recall = plt.subplots(figsize=figsize)
    roc_fig, ax_roc = plt.subplots(figsize=figsize)
    
    if plot_background:
        ax_recall.set_facecolor('#f0f0f0')
        ax_roc.set_facecolor('#f0f0f0')
    
    # Add diagonal reference line for ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.7)
    
    # Line styles for different models
    model_line_styles = {
        0: '--' if first_model_dashed else '-',  # First model line style
        1: '-' if first_model_dashed else '--',  # Second model line style
    }
    
    # Process each g_max category to match the image style
    for g_max_cat in g_max_categories:
        # Process for the recall plot
        recall_key = f"{g_max_cat}_recall"
        category_color = colors[g_max_cat]
        
        for dict_idx, (dict_name, data_dict) in enumerate(data_points_dict.items()):
            if recall_key in data_dict:
                if "thresholds" not in data_dict[recall_key] or "values" not in data_dict[recall_key]:
                    continue
                
                thresholds = data_dict[recall_key]['thresholds']
                recall_values = data_dict[recall_key]['values']
                
                # Sort thresholds and values together
                points = sorted(zip(thresholds, recall_values))
                if points:
                    thresholds_sorted, recall_values_sorted = zip(*points)
                    
                    # Use appropriate line style based on model index
                    line_style = model_line_styles.get(dict_idx, '--' if dict_idx % 2 == 0 else '-')
                    label = f"{dict_name} - {g_max_cat} - Recall"
                    
                    ax_recall.plot(thresholds_sorted, recall_values_sorted, 
                                  linestyle=line_style,
                                  color=category_color,
                                  label=label,
                                  linewidth=1.5,
                                  alpha=0.9)
        
        # Process for the ROC plot
        roc_key = f"{g_max_cat}_roc"
        
        for dict_idx, (dict_name, data_dict) in enumerate(data_points_dict.items()):
            if roc_key in data_dict:
                # Check if ROC data is in fpr/tpr format
                if "fpr" in data_dict[roc_key] and "tpr" in data_dict[roc_key]:
                    fpr = data_dict[roc_key]['fpr']
                    tpr = data_dict[roc_key]['tpr']
                    
                    # Calculate AUC if there are enough points
                    if len(fpr) > 1 and len(tpr) > 1:
                        try:
                            from sklearn.metrics import auc
                            roc_auc = auc(fpr, tpr)
                            auc_text = f" (AUC = {roc_auc:.3f})"
                        except:
                            auc_text = ""
                    else:
                        auc_text = ""
                    
                    # Sort by fpr to ensure proper line drawing
                    points = sorted(zip(fpr, tpr))
                    if points:
                        fpr_sorted, tpr_sorted = zip(*points)
                        
                        # Use appropriate line style based on model index
                        line_style = model_line_styles.get(dict_idx, '--' if dict_idx % 2 == 0 else '-')
                        label = f"{dict_name} - {g_max_cat}{auc_text}"
                        
                        ax_roc.plot(fpr_sorted, tpr_sorted, 
                                    linestyle=line_style,
                                    color=category_color,
                                    label=label,
                                    linewidth=1.5,
                                    alpha=0.9)
                
                # Check if ROC data is in thresholds/values format
                elif "thresholds" in data_dict[roc_key] and "values" in data_dict[roc_key]:
                    thresholds = data_dict[roc_key]['thresholds']
                    values = data_dict[roc_key]['values']
                    
                    # Sort thresholds and values together
                    points = sorted(zip(thresholds, values))
                    if points:
                        thresholds_sorted, values_sorted = zip(*points)
                        
                                                        # Use appropriate line style based on model index
                        line_style = model_line_styles.get(dict_idx, '--' if dict_idx % 2 == 0 else '-')
                        label = f"{dict_name} - {g_max_cat}"
                            
                        ax_roc.plot(thresholds_sorted, values_sorted, 
                                    linestyle=line_style,
                                    color=category_color,
                                    label=label,
                                    linewidth=1.5,
                                    alpha=0.9)
    
    # Finalize recall plot
    ax_recall.set_xlabel('Confidence Threshold', fontsize=12)
    ax_recall.set_ylabel('Score', fontsize=12)
    ax_recall.set_title('Precision & Recall vs. Confidence Threshold Across Segments', fontsize=14)
    ax_recall.set_xlim([0.1, 0.9])
    ax_recall.set_ylim([0.0, 1.05])
    ax_recall.grid(alpha=0.3)
    ax_recall.legend(loc='lower left', fontsize=9, framealpha=0.9)
    
    # Finalize ROC plot
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves Across Segments', fontsize=14)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Apply tight layout
    recall_fig.tight_layout()
    roc_fig.tight_layout()
    
    # Save figures if output_filename is provided
    if output_filename:
        recall_fig.savefig(f"{output_filename}_recall.png", dpi=dpi, bbox_inches='tight')
        roc_fig.savefig(f"{output_filename}_roc.png", dpi=dpi, bbox_inches='tight')
    
    return recall_fig, roc_fig

import matplotlib.pyplot as plt

def plot_time_series(df):
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract time and sensor data
    time = df.iloc[:, 0]  # First column is timestamp
    # x_data = df.iloc[:, 1]  # Second column is x-axis
    # y_data = df.iloc[:, 2]  # Third column is y-axis
    # z_data = df.iloc[:, 3]  # Fourth column is z-axis
    g_data = df.iloc[:, 4]  # Fourth column is z-axis

    # Plot the three axes
    # ax.plot(time, x_data, label='X-axis', alpha=0.8)
    # ax.plot(time, y_data, label='Y-axis', alpha=0.8)
    # ax.plot(time, z_data, label='Z-axis', alpha=0.8)
    ax.plot(time, g_data, label='G', alpha=0.8)

    # Add labels and legend
    ax.set_title('Accelerometer Data')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Acceleration (G force)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

from simpml.tabular.all import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def train_binary_tabular_model(df, target_col, train_ids, test_ids):
    splitter = IndexSplitter(split_sets={Dataset.Train: train_ids, Dataset.Valid: test_ids})

    data_manager = SupervisedTabularDataManager(
        data=df,
        target=target_col,
        prediction_type=PredictionType.BinaryClassification,
        splitter=splitter,
    )

    data_manager.build_pipeline(
        remove_special_json_characters=False,
        step_params={'encoding_dict': {'normal': 0, 'collision': 1}},
        smote=False
    )

    exp_mang = ExperimentManager(data_manager, optimize_metric=MetricName.AUC)
    exp_mang.remove_models([
        'LightGBM', 'Support Vector Classifier',
        'AdaBoost Classifier', 'Decision Tree', 'Gradient Boosting'
    ])
    exp_mang.run_experiment(metrics_kwargs={'pos_label': 1})

    return exp_mang, data_manager


def analyze_by_segment(preds, meta_df, g_max_col='g_max'):
    preds = preds.merge(meta_df[['id', g_max_col]], on='id', how='left')

    analyzer = SegmentAnalyzer(preds, g_max_col)
    analyzer.add_segment(None, 1.0, true_pos_population=400, true_neg_population=500000)
    analyzer.add_segment(1.0, 1.4, true_pos_population=800, true_neg_population=50000)
    analyzer.add_segment(1.4, None, true_pos_population=1200, true_neg_population=5000)

    _ = analyzer.analyze(y_true_col='y_true', y_prob_cols=['y_prob_0', 'y_prob_1'])

    return analyzer.summary(), analyzer.plot_segment_comparisons(
        y_true_col='y_true',
        y_prob_cols=['y_prob_0', 'y_prob_1'],
        metric_type='recall'
    )


def plot_top_features(fi, df_features, meta_df, n=10):
    # Normalize video_type labels
    meta_df['video_type'] = meta_df['video_type'].replace({
        'nv_collision_reviewed': 'collision',
        'nv_collision_full_annotation': 'collision'
    })
    meta_df['video_type'] = meta_df['video_type'].apply(lambda x: 'normal' if x != 'collision' else x)

    meta_df['id'] = meta_df['id'].astype(str)
    df_features.index = df_features.index.astype(str)

    # Top N features
    top_features = fi.sort_values('feature_importance_vals', ascending=False).head(n)

    # Horizontal bar plot of top features
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='feature_importance_vals', 
        y='col_name', 
        data=top_features,
        orient='h'
    )
    plt.title('Top Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Create subplots for distributions: 2 plots per feature
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    for i, (_, row) in enumerate(top_features.iterrows()):
        feature = row['col_name']
        importance = row['feature_importance_vals']
        merged = meta_df.merge(df_features[[feature]], left_on='id', right_index=True)

        # Violin plot
        sns.violinplot(ax=axes[i, 0], x='video_type', y=feature, data=merged, inner='quartile')
        axes[i, 0].set_title(f"{feature} - Violin\nImportance: {importance:.4f}")
        axes[i, 0].tick_params(axis='x', rotation=15)

        # Histogram
        sns.histplot(ax=axes[i, 1], data=merged, x=feature, hue='video_type', element='step', stat='density', common_norm=False)
        axes[i, 1].set_title(f"{feature} - Histogram\nImportance: {importance:.4f}")
        axes[i, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.show()