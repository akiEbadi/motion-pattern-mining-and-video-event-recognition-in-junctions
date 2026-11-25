"""Naming, matching and scoring of event classes, plus the plots.

- describe_profile / describe_clips: give a learned class a readable name (its main directions)
- map_states_to_labels: match learned states to label names (Hungarian algorithm)
- evaluate_test: accuracy, per-class report and confusion matrix on the test clips
- plot_class_motion: arrow map of each class's main direction per grid cell
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, accuracy_score,
                             balanced_accuracy_score, f1_score, classification_report, confusion_matrix)

# Image coordinates: y grows downward, so 90 degrees points down the frame.
COMPASS = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']


def describe_profile(profile):
    """Dominant direction and spread of one event class.

    profile: (cells, bins) average word counts of the clips in the class.
    Returns a short name and the resultant length R (1 = one direction, 0 = uniform).
    """
    bins = profile.shape[1]
    weights = profile.sum(axis=0)
    total = weights.sum()
    if total <= 0:
        return 'no motion', 0.0
    centers = np.arange(bins) * 2 * np.pi / bins   # bins are centred on E, SE, S, ...
    z = (weights * np.exp(1j * centers)).sum() / total
    resultant = float(abs(z))
    angle = np.angle(z) % (2 * np.pi)
    if bins == 8:
        name = COMPASS[int(round(angle / (2 * np.pi / 8))) % 8]
    else:
        name = f'{np.degrees(angle):.0f} deg'
    if resultant < 0.6:
        return f'mixed (turning or multi-direction), mainly {name}', resultant
    return f'{name} flow', resultant


def dir_name(d, bins):
    """Compass name of direction bin d."""
    return COMPASS[d] if bins == 8 else f'{d * 360 / bins:.0f} deg'


def describe_clips(clip_dirs, bins, min_share=0.15):
    """Name a class from the dominant direction of each of its clips.

    clip_dirs: dominant direction of each clip in the class (-1 = no motion).
    Returns e.g. 'N 41% / S 22%' (share of the class's clips with that dominant direction).
    """
    valid = clip_dirs[clip_dirs >= 0]
    if len(clip_dirs) == 0:
        return 'empty class'
    if len(valid) == 0:
        return 'no motion'
    shares = np.bincount(valid, minlength=bins) / len(valid)
    order = np.argsort(-shares)[:3]
    parts = [f'{dir_name(d, bins)} {shares[d]:.0%}' for d in order if shares[d] >= min_share]
    return ' / '.join(parts) if parts else 'mixed directions'


def map_states_to_labels(states, labels):
    """Match learned states to ground-truth labels with the Hungarian algorithm (training clips only)."""
    classes = sorted(set(labels))
    cm = np.zeros((int(states.max()) + 1, len(classes)), dtype=int)
    for s, lab in zip(states, labels):
        cm[s, classes.index(lab)] += 1
    rows, cols = linear_sum_assignment(-cm)
    mapping = {int(r): classes[c] for r, c in zip(rows, cols)}
    return mapping


def evaluate_test(pred_states, true_labels, mapping, out_dir, log=print):
    """Accuracy of the mapped test predictions against the supplied labels."""
    named = np.array([mapping.get(int(s), 'unmatched') for s in pred_states])
    y_true = np.asarray(true_labels)
    classes = sorted(set(y_true) | set(named))
    text = []
    text.append(f'Test clips with labels: {len(y_true)}')
    text.append(f'Accuracy: {accuracy_score(y_true, named):.4f}')
    text.append(f'Balanced accuracy: {balanced_accuracy_score(y_true, named):.4f}')
    text.append(f'Macro F1: {f1_score(y_true, named, average="macro", labels=classes, zero_division=0):.4f}')
    text.append(classification_report(y_true, named, labels=classes, zero_division=0))
    text = '\n'.join(text)
    log(text)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(text + '\n')

    cm = confusion_matrix(y_true, named, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200)
    plt.close(fig)
    return named


def cluster_agreement(pred_states, true_labels):
    enc = {lab: i for i, lab in enumerate(sorted(set(true_labels)))}
    y = np.array([enc[lab] for lab in true_labels])
    return (adjusted_rand_score(y, pred_states[:len(y)]), normalized_mutual_info_score(y, pred_states[:len(y)]))


def plot_class_motion(profiles, names, rows, cols, out_path):
    """Quiver plot of each event class: arrow = dominant direction of each grid cell."""
    k = len(profiles)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), squeeze=False)
    bins = profiles[0].shape[1]
    ys, xs = np.mgrid[0:rows, 0:cols]
    for i, (profile, name) in enumerate(zip(profiles, names)):
        ax = axes[0, i]
        weight = profile.sum(axis=1)
        direction = profile.argmax(axis=1)
        angle = direction * 2 * np.pi / bins
        scale = weight / max(weight.max(), 1e-12)
        ax.quiver(xs.ravel(), ys.ravel(), np.cos(angle) * scale, np.sin(angle) * scale,
                  angles='xy', scale_units='xy', scale=0.8, width=0.006)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
