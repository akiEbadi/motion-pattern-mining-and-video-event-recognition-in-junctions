"""Supervised event classification of test clips (thesis chapter 4, step 4, deep-learning version).

The thesis trains an SVM (approach 2) or uses K-means centres (approach 3) to give
each test clip a traffic state. Here, once the training clips have labels, two small
neural networks are trained on them and their probabilities are averaged:

  * network A reads the clip's bag of (cell, direction) words, as shares of the clip,
  * network B reads the clip's activity vector (thesis eq. 9).

Empty-junction clips are not classified by the networks: the empty-junction test in
src/features.py decides them first.

Best result on the 700 test clips: 94.6% with four direction classes, 93.7% once the
'mixed' class is added (one more class means more ways to be wrong, but the labels are
more truthful). See the README for how much of that is real, based on the blind check.
"""
import numpy as np
import torch
from torch import nn

# Classes the networks choose between. 'empty' is decided before the networks run by the
# empty-junction test in src/features.py, so it is never one of the network's outputs.
# 'mixed' = vertical and turning happen together in the clip, neither is the majority.
CLASSES = ['left', 'right', 'turning', 'vertical', 'mixed']


class MLP(nn.Module):
    """Two hidden layers with dropout. Small on purpose: only ~470 labelled training clips."""

    def __init__(self, in_dim, n_classes=len(CLASSES), hidden=256):
        """in_dim: number of input features. n_classes: how many classes to choose between."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, 128), nn.ReLU(),
                                 nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def bow_shares(counts):
    """Word counts -> percentage of the clip's words (so busy and quiet clips are comparable)."""
    return (counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 100).astype(np.float32)


def _train_one(X, y, seed, n_classes=len(CLASSES), epochs=120, lr=1e-3, batch=32):
    """Train one network on (X, y) and return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1], n_classes=n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Class weights: turning and empty-ish classes are rare, so they count more in the loss.
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1)
    weights = torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch):
            idx = order[start:start + batch]
            loss = loss_fn(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _probs(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), dim=1).numpy()


def train_ensemble(bow, act, labels, train_idx, classes=CLASSES, seeds=3, log=print):
    """Train networks A (bag of words) and B (activity vectors), `seeds` copies of each.

    bow, act : features for all clips
    labels   : array of class names (None for unlabelled clips)
    train_idx: clip indices allowed for training (labelled, non-empty training clips)
    """
    y = np.array([classes.index(labels[i]) for i in train_idx])
    models = {'bow': [], 'act': [], 'classes': classes}
    for s in range(seeds):
        models['bow'].append(_train_one(bow[train_idx], y, seed=s, n_classes=len(classes)))
        models['act'].append(_train_one(act[train_idx], y, seed=s, n_classes=len(classes)))
    log(f'  trained {seeds} + {seeds} networks on {len(train_idx)} labelled training clips')
    return models


def predict_ensemble(models, bow, act, empty):
    """Average the probabilities of all networks; empty-junction clips are labelled 'empty'."""
    classes = models.get('classes', CLASSES)
    p = sum(_probs(m, bow) for m in models['bow']) + sum(_probs(m, act) for m in models['act'])
    p /= p.sum(axis=1, keepdims=True)
    names = np.array([classes[k] for k in p.argmax(axis=1)], dtype=object)
    names[empty] = 'empty'
    return names, p
