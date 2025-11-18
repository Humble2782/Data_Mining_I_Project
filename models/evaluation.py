"""
Standardized Model Evaluation Module
==================================

This module provides a standardized set of functions for evaluating
classification and clustering models based on the project outline.

It includes:
- Baseline "Most Frequent Class" classifier performance.
- Classification metrics for imbalanced and ordinal data:
  - Weighted F1-Score
  - Weighted Cohen's Kappa (with quadratic weighting)
  - Full classification_report
- Visualization functions:
  - Confusion Matrix (raw or normalized)
  - Multi-class Precision-Recall Curves
- Clustering metrics:
  - Silhouette Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional

# Scikit-learn metrics and tools
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    precision_recall_curve,
    average_precision_score,
    silhouette_score
)
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyClassifier


def get_baseline_performance(y_train: pd.Series,
                             y_test: pd.Series,
                             labels: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Calculates the performance of a "Most Frequent Class" baseline classifier.

    This serves as the minimum threshold all other models must outperform.

    Args:
        y_train: Training target values (used to find the most frequent class).
        y_test: Test target values (used to evaluate the baseline).
        labels: The unique, sorted list of class labels (e.g., [0, 1, 2]).

    Returns:
        A dictionary containing the baseline's weighted F1 and Kappa scores.
    """
    print("Calculating baseline (Most Frequent Class) performance...")
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(y_train, y_train)  # Fit on training data
    y_pred_baseline = baseline_model.predict(y_test)

    if labels is None:
        labels = np.unique(y_test)

    # Calculate key baseline metrics
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted', labels=labels)
    # Use quadratic weighting for Kappa to penalize distant errors
    baseline_kappa = cohen_kappa_score(y_test, y_pred_baseline, weights='quadratic')

    print("\n--- Baseline (Most Frequent Class) Report ---")
    print(classification_report(y_test, y_pred_baseline, labels=labels, zero_division=0))
    print(f"Baseline Weighted F1-Score: {baseline_f1:.4f}")
    print(f"Baseline Weighted Cohen's Kappa: {baseline_kappa:.4f}")
    print("--------------------------------------------------")

    return {
        "baseline_weighted_f1": baseline_f1,
        "baseline_weighted_kappa": baseline_kappa
    }


def print_classification_report(y_true: pd.Series,
                                y_pred: np.ndarray,
                                target_names: Optional[List[str]] = None,
                                labels: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Prints a comprehensive classification report including weighted F1
    and quadratically weighted Cohen's Kappa.

    Args:
        y_true: True target values.
        y_pred: Predicted target values from the model.
        target_names: String names for the labels (e.g., ["Uninjured", "Light", "Severe"]).
        labels: The unique, sorted list of class labels (e.g., [0, 1, 2]).

    Returns:
        A dictionary containing the model's weighted F1 and Kappa scores.
    """
    if labels is None:
        labels = np.unique(y_true)
    if target_names is None:
        target_names = [str(label) for label in labels]

    # 1. Weighted F1-Score (balances precision/recall for imbalanced data)
    model_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels)

    # 2. Weighted Cohen's Kappa (penalizes distant ordinal errors)
    #    'quadratic' weighting is specified for ordinal metrics.
    model_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print("\n--- Model Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=target_names, labels=labels, zero_division=0))
    print("\n--- Ordinal & Imbalanced Metrics ---")
    print(f"Model Weighted F1-Score: {model_f1:.4f}")
    print(f"Model Weighted Cohen's Kappa (Quadratic): {model_kappa:.4f}")
    print("--------------------------------------------------")

    return {
        "model_weighted_f1": model_f1,
        "model_weighted_kappa": model_kappa
    }


def plot_confusion_matrix(y_true: pd.Series,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          ax: Optional[plt.Axes] = None,
                          normalize: Optional[str] = None,
                          title: str = 'Confusion Matrix'):
    """
    Plots a confusion matrix on a given matplotlib Axes.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        ax: The matplotlib Axes object to plot on. If None, a new figure is created.
        class_names: String names for the labels.
        normalize: 'true', 'pred', or None.
                  'true' normalizes over the true labels (rows).
                  'pred' normalizes over the predicted labels (columns).
        title: The title for the plot.
    """
    # --- NEW: Standard behavior if ax is not provided ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))  # Create a new figure and axis
        show_plot = True  # Flag to call plt.show() later
    # --- END NEW ---

    cm = confusion_matrix(y_true, y_pred)

    fmt = 'd'  # Format for annotations (integer)

    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # Switch to float format
        title += ' (Normalized by True Class)'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized by Predicted Class)'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    # --- NEW: Show plot if created internally ---
    if show_plot:
        plt.tight_layout()
        plt.show()
    # --- END NEW ---


def plot_precision_recall_curve(y_true: pd.Series,
                                y_prob: np.ndarray,
                                class_names: List[str],
                                ax: Optional[plt.Axes] = None):
    """
    Plots a multi-class Precision-Recall curve (One-vs-Rest) on a
    given matplotlib Axes.

    Args:
        y_true: True target values.
        y_prob: Predicted probabilities (from model.predict_proba()).
        class_names: String names for the labels.
        ax: The matplotlib Axes object to plot on. If None, a new figure is created.
    """
    # --- NEW: Standard behavior if ax is not provided ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))  # Create a new figure and axis
        show_plot = True  # Flag to call plt.show() later
    # --- END NEW ---

    # Binarize the output
    labels = np.unique(y_true)
    y_true_binarized = label_binarize(y_true, classes=labels)
    n_classes = len(labels)

    # Plot PR curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])
        avg_precision = average_precision_score(y_true_binarized[:, i], y_prob[:, i])

        ax.plot(recall, precision, lw=2,
                label=f'Class {class_names[i]} (AP = {avg_precision:0.2f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Multi-class Precision-Recall Curve', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.4)

    # --- NEW: Show plot if created internally ---
    if show_plot:
        plt.tight_layout()
        plt.show()
    # --- END NEW ---


def run_classification_evaluation(model: Any,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series,
                                  y_train: pd.Series,
                                  class_names: List[str],
                                  labels: List[int]):
    """
    A full-service wrapper function that runs the complete
    classification evaluation as specified in the project outline.

    Args:
        model: The trained classifier model (must have .predict() and .predict_proba()).
        X_test: Test features.
        y_test: Test target.
        y_train: Training target (for baseline calculation).
        class_names: List of string names for the classes (e.g., ["Uninjured", "Light", "Severe"]).
        labels: List of integer labels (e.g., [0, 1, 2]).

    Returns:
        A dictionary with key baseline and model scores.
    """

    # --- 1. Get Baseline Performance ---
    baseline_scores = get_baseline_performance(y_train, y_test, labels=labels)

    # --- 2. Get Model Predictions ---
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # --- 3. Get Model Metrics ---
    model_scores = print_classification_report(y_test, y_pred,
                                               target_names=class_names,
                                               labels=labels)

    # --- 4. Create Visualizations ---
    print("Generating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Model Evaluation: {type(model).__name__}', fontsize=16)

    # Plot Raw Confusion Matrix
    # plot_confusion_matrix(y_test, y_pred, ax=ax1,
    #                       class_names=class_names,
    #                       normalize=None,
    #                       title='Confusion Matrix (Raw Counts)')

    # Plot Normalized Confusion Matrix (often more insightful)
    plot_confusion_matrix(y_test, y_pred, ax=ax1,
                          class_names=class_names,
                          normalize='true',  # Normalize by true class (rows)
                          title='Confusion Matrix (Normalized by True Class)')

    # Plot PR Curve
    plot_precision_recall_curve(y_test, y_prob, ax=ax2,
                                class_names=class_names)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return {**baseline_scores, **model_scores}


def evaluate_clustering(X_data: pd.DataFrame, cluster_labels: np.ndarray):
    """
    Evaluates a clustering model using the Silhouette Score.

    Args:
        X_data: The feature data used for clustering.
        cluster_labels: The labels assigned by the clustering algorithm.
    """
    print("\n--- Clustering Evaluation ---")

    # 1. Silhouette Score
    try:
        score = silhouette_score(X_data, cluster_labels, metric='euclidean')
        print(f"Silhouette Score: {score:.4f}")
    except ValueError as e:
        print(f"Could not calculate Silhouette Score. (Error: {e})")
        print("This often happens if only one cluster is found.")

    # 2. Qualitative Review Reminder
    print("\nReminder: Qualitative review is required.")
    print("Please manually inspect the centroids or exemplars of each cluster")
    print("to assess the interpretability of the 'accident scenarios.'")
    print("--------------------------------")