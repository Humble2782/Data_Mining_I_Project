import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np


def plot_confusion_matrices_row(models_dict, y_test, class_names=['Uninjured', 'Injured', 'Severe']):
    """
    Plots confusion matrices in a single row (1x4) layout.
    """
    # Set font scale for readability
    sns.set_context("notebook", font_scale=1.5)

    # 1 row, 4 columns. Adjusted figure size (Wide and short)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes = axes.flatten()

    for i, (name, y_pred) in enumerate(models_dict.items()):
        if i >= 4: break  # Safety break

        ax = axes[i]

        # Calculate normalized confusion matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true')

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    annot_kws={"size": 16, "weight": "bold"})

        ax.set_title(name, fontsize=18, fontweight='bold', pad=15)

        # Y-Label only on the first plot to avoid clutter
        if i == 0:
            ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
        else:
            ax.set_ylabel('')

        ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')

        # Customize Ticks (Fat and Big)
        # labelsize sets font size, width makes the tick marks themselves thicker
        ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=7)

        # Make the tick label text bold ("fat")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        if i > 0:
            ax.set_yticks([])  # Optional: Hide y-ticks for subsequent plots for cleaner look

    plt.tight_layout()
    plt.savefig('images/confusion_matrices_row.pdf', bbox_inches='tight')
    plt.show()



def plot_precision_recall_row(models_proba_dict, y_test, class_names=['Uninjured', 'Injured', 'Severe']):
    """
    Plots Precision-Recall curves for multiple models in a single row.

    Args:
        models_proba_dict: Dictionary {ModelName: y_score}
                           where y_score is usually model.predict_proba(X_test)
                           or model.decision_function(X_test).
                           Shape must be (n_samples, n_classes).
        y_test: True labels (integers 0, 1, 2)
        class_names: List of class names
    """
    # Increased font_scale significantly
    sns.set_context("notebook", font_scale=1.5)
    n_classes = len(class_names)

    # Binarize y_test for multi-class PR calculation (One-vs-Rest)
    # Assumes classes are mapped to 0, 1, 2...
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    n_models = len(models_proba_dict)
    # Create subplots
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    # Handle single model case
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = sns.color_palette("deep", n_classes)  # distinct colors for classes

    for i, (name, y_score) in enumerate(models_proba_dict.items()):
        ax = axes[i]

        lines = []
        labels = []

        for j in range(n_classes):
            # Calculate Precision and Recall
            precision, recall, _ = precision_recall_curve(y_test_bin[:, j], y_score[:, j])
            # Calculate Average Precision (Area under the curve)
            avg_precision = average_precision_score(y_test_bin[:, j], y_score[:, j])

            # Increased line width (lw=3) for "fatter" look
            l, = ax.plot(recall, precision, color=colors[j], lw=3)
            lines.append(l)
            labels.append(f'{class_names[j]} (AP={avg_precision:.2f})')

        # Title (Bold and Big)
        ax.set_title(name, fontsize=18, fontweight='bold', pad=15)

        # Axis Labels (Bold and Big)
        ax.set_xlabel('Recall', fontsize=16, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Precision', fontsize=16, fontweight='bold')
        else:
            ax.set_ylabel('')
            ax.set_yticks([])

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Legend (Bold and Big)
        # prop={'weight': 'bold', 'size': 13} makes the text fat and large
        ax.legend(lines, labels, loc='lower left', prop={'weight': 'bold', 'size': 13})

        # Ticks (Fat and Big)
        # labelsize sets font size
        # width=2.5 makes the tick marks themselves thicker
        ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=7)

        # Make the tick label text bold ("fat")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.grid(True, alpha=0.3, linewidth=1.5)

    plt.tight_layout()
    plt.savefig('images/precision_recall_row.pdf', bbox_inches='tight')
    plt.show()