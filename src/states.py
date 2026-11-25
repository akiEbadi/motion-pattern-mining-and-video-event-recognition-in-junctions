"""Event-class (traffic-state) learning, thesis chapter 4, step 3, approach 2.

The thesis combines HMM-LDA with a classifier. Here a hybrid neural HMM does
both jobs: a feed-forward network gives per-clip state posteriors, a learned
transition matrix links consecutive clips, and EM trains the two together.
Test clips are classified online with the causal forward filter.
"""
import numpy as np
import torch
from torch import nn

from src.kmeans import kmeans_numpy


class StateNet(nn.Module):
    def __init__(self, in_dim, hidden, n_states):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_states))

    def forward(self, x):
        return self.net(x)


def forward_backward(likelihood, A, pi):
    """Scaled forward-backward. likelihood: (T, K), A: (K, K), pi: (K,)."""
    T, K = likelihood.shape
    alpha = np.zeros((T, K))
    scale = np.zeros(T)
    a = pi * likelihood[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * likelihood[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]

    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (likelihood[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(T - 1):
        m = alpha[t][:, None] * A * (likelihood[t + 1] * beta[t + 1])[None, :]
        xi += m / m.sum()
    return gamma, xi, np.log(scale).sum()


def _network_posteriors(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).numpy()


def train_state_model(features, n_states, hidden, em_iters, mlp_steps, lr, seed, log=print):
    """Train the hybrid HMM on the training clips.

    Returns the network, transition matrix, initial distribution, state
    priors, smoothed training posteriors (gamma) and the feature scaler.
    """
    torch.manual_seed(seed)
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-6
    scaler = (mean, std)
    features_std = (features - mean) / std
    X = torch.tensor(features_std, dtype=torch.float32)
    T = len(features)

    # Start from K-means labels (thesis approach 3) so the states begin well separated.
    init_labels, _ = kmeans_numpy(features_std, k=n_states, random_state=seed)
    gamma = np.full((T, n_states), 0.1 / (n_states - 1))
    gamma[np.arange(T), init_labels] = 0.9

    A = np.ones((n_states, n_states))
    for t in range(1, T):
        A[init_labels[t - 1], init_labels[t]] += 1
    A /= A.sum(axis=1, keepdims=True)
    pi = gamma[0].copy()

    model = StateNet(features_std.shape[1], hidden, n_states)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prior = np.full(n_states, 1.0 / n_states)

    for it in range(em_iters):
        # M-step for the emission network: soft-target cross-entropy on gamma.
        targets = torch.tensor(gamma, dtype=torch.float32)
        model.train()
        for _ in range(mlp_steps):
            log_probs = torch.log_softmax(model(X), dim=1)
            loss = -(targets * log_probs).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # E-step: scaled likelihoods p(state | clip) / p(state) plus the HMM recursion.
        posteriors = _network_posteriors(model, X)
        prior = posteriors.mean(axis=0)
        likelihood = np.maximum(posteriors / prior, 1e-12)
        gamma, xi, loglik = forward_backward(likelihood, A, pi)
        xi_sum = xi + 1e-3
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        pi = gamma[0].copy()
        occupancy = np.round(gamma.sum(axis=0)).astype(int).tolist()
        log(f'  state EM iteration {it + 1}/{em_iters} log-likelihood={loglik:.2f} clips per state={occupancy}')

    return model, A, pi, prior, gamma, scaler


def filter_states(model, features, A, pi, prior, scaler):
    """Causal (online) state posteriors p(state_t | clips 1..t) for every clip."""
    mean, std = scaler
    X = torch.tensor((features - mean) / std, dtype=torch.float32)
    likelihood = np.maximum(_network_posteriors(model, X) / prior, 1e-12)
    T, K = likelihood.shape
    alpha = np.zeros((T, K))
    a = pi * likelihood[0]
    alpha[0] = a / a.sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * likelihood[t]
        alpha[t] = a / a.sum()
    return alpha


def apply_silent_rule(pred, silent):
    """Thesis eq. (14): a silent clip keeps the state of the clip before it."""
    pred = pred.copy()
    for t in range(1, len(pred)):
        if silent[t]:
            pred[t] = pred[t - 1]
    return pred
