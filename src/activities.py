"""Activity discovery (thesis chapter 4, step 2) with a neural topic model.

The thesis learns activities with HDP over clip word documents. Here a
variational topic model (encoder network + mixture decoder) learns topics
from the same documents. The selection rules of the thesis (eqs. 3-8) and the
clip activity vectors (eq. 9) are kept unchanged.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TopicModel(nn.Module):
    def __init__(self, vocab, n_topics, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(vocab, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, n_topics)
        self.logvar = nn.Linear(hidden, n_topics)
        self.beta_logits = nn.Parameter(torch.randn(n_topics, vocab))  # random init breaks topic symmetry

    def encode(self, x):
        h = self.encoder(torch.log1p(x))
        return self.mu(h), self.logvar(h)

    def topics(self):
        # Each row is one activity's distribution over the word vocabulary.
        return torch.softmax(self.beta_logits, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        theta = torch.softmax(z, dim=1)                    # clip-level activity mixture
        word_probs = theta @ self.topics()                 # mixture decoder
        recon = -(x * torch.log(word_probs + 1e-10)).sum(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        return recon, kl


def train_topic_model(counts, n_topics, hidden, epochs, lr, batch, seed, log=print):
    torch.manual_seed(seed)
    x = torch.tensor(counts, dtype=torch.float32)
    model = TopicModel(counts.shape[1], n_topics, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x), batch_size=batch, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (xb,) in loader:
            recon, kl = model(xb)
            loss = (recon + kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch == 0 or (epoch + 1) % 50 == 0:
            log(f'  topic model epoch {epoch + 1}/{epochs} loss={total / len(x):.3f}')
    model.eval()
    with torch.no_grad():
        theta = torch.softmax(model.encode(x)[0], dim=1).numpy()
        beta = model.topics().numpy()
    return theta, beta


def assign_tokens(counts, theta, beta):
    """Token count per activity: each (clip, word) goes to its most likely activity."""
    scores = theta[:, :, None] * beta[None, :, :]           # clips x activities x words
    token_activity = scores.argmax(axis=1)                  # clips x words
    return np.bincount(token_activity.ravel(), weights=counts.ravel(),
                       minlength=beta.shape[0])


def select_activities(n_k, beta, typical_ratio, noise_threshold):
    """Thesis eqs. (3)-(8): keep typical activities and drop noisy words."""
    order = np.argsort(-n_k)
    ratios = n_k[order] / max(n_k.sum(), 1e-12)             # eq. (3), sorted (eq. 4)
    cumulative = np.cumsum(ratios)                          # eq. (5)
    typical = order[cumulative < typical_ratio]             # eq. (6)
    if len(typical) < 2:
        typical = order[:2]

    activities = []
    for k in typical:
        p = beta[k]
        ranked = np.argsort(-p)
        cum = np.cumsum(p[ranked])
        keep = min(np.searchsorted(cum, noise_threshold) + 1, len(ranked))  # eqs. (7)-(8)
        activities.append({'topic': int(k), 'ratio': float(n_k[k] / max(n_k.sum(), 1e-12)),
                           'words': np.sort(ranked[:keep]), 'probs': p})
    return activities


def activity_vectors(counts, activities):
    """Thesis eq. (9): fraction of a clip's words that belong to each activity."""
    totals = counts.sum(axis=1)
    totals = np.where(totals > 0, totals, 1.0)
    vectors = np.zeros((counts.shape[0], len(activities)), dtype=np.float64)
    for i, act in enumerate(activities):
        vectors[:, i] = counts[:, act['words']].sum(axis=1) / totals
    return vectors
