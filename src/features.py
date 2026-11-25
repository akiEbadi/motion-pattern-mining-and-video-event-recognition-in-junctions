"""Step 1 of the thesis: video -> clips -> optical flow -> (cell, direction) words.

- extract_cell_flow: decode the video (in parallel) and store the mean flow of each grid cell
- quantize_flow: turn cell flow into one of 8 compass directions (or -1 = no motion)
- bag_of_words: count (cell, direction) words per clip = the clip "document"
- empty_clips: the empty-junction test (no vehicle moves through the junction)
- clip_directions: main direction of each clip
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np


def cell_mean_flow(flow, rows, cols):
    """Mean (dx, dy) of each grid cell for one optical-flow field -> (rows*cols, 2)."""
    h, w = flow.shape[:2]
    re = np.linspace(0, h, rows + 1, dtype=int)
    ce = np.linspace(0, w, cols + 1, dtype=int)
    cell_sizes = np.outer(np.diff(re), np.diff(ce)).astype(np.float64)
    # Block sums per cell; reduceat handles cells that differ by one pixel in size.
    sum_dx = np.add.reduceat(np.add.reduceat(flow[..., 0], re[:-1], axis=0), ce[:-1], axis=1)
    sum_dy = np.add.reduceat(np.add.reduceat(flow[..., 1], re[:-1], axis=0), ce[:-1], axis=1)
    return np.stack([(sum_dx / cell_sizes).ravel(), (sum_dy / cell_sizes).ravel()], axis=1)


def quantize_flow(cell_flow, bins=8, min_magnitude=0.5):
    """Direction word per cell (thesis chapter 4, step 1).

    Bins are CENTRED on the compass directions: with 8 bins, bin 0 = E covers
    -22.5..22.5 deg, bin 2 = S (down in the image), bin 4 = W, bin 6 = N (up).
    Image coordinates: +y points down. Cells weaker than min_magnitude get -1.
    """
    dx = cell_flow[..., 0].astype(np.float32)
    dy = cell_flow[..., 1].astype(np.float32)
    magnitude = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    width = 2 * np.pi / bins
    codes = np.floor((angle + width / 2) / width).astype(int) % bins
    return np.where(magnitude >= min_magnitude, codes, -1).astype(np.int8)


def _video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def _flow_worker(job):
    """Decode from the start (cheap) and compute flow only for clips [c0, c1)."""
    video_path, c0, c1, clip_length, rows, cols, wid = job
    cv2.setNumThreads(1)
    out = np.zeros((c1 - c0, clip_length - 1, rows * cols, 2), dtype=np.float16)
    first, last = c0 * clip_length, c1 * clip_length   # frames [first, last)
    cap = cv2.VideoCapture(video_path)
    prev, idx = None, 0
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= first:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos = idx % clip_length
            if pos != 0 and prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                out[idx // clip_length - c0, pos - 1] = cell_mean_flow(flow, rows, cols)
            prev = gray
            if (idx - first) % 5000 == 0 and idx > first:
                print(f'  worker {wid}: {idx - first}/{last - first} frames', flush=True)
        idx += 1
    cap.release()
    return c0, out


def extract_cell_flow(video_path, clip_length=75, rows=10, cols=10, workers=None):
    """Mean optical flow of every grid cell for every complete clip.

    Output shape: [clips, clip_length - 1, rows * cols, 2] (dx, dy), float16.
    Clips do not overlap: clip k covers frames [k*clip_length, (k+1)*clip_length).
    Storing the flow (not the quantized words) lets the direction bins change
    later without decoding the video again.
    """
    fps, n_frames = _video_info(video_path)
    if n_frames <= 0:  # some containers do not report the frame count
        cap = cv2.VideoCapture(video_path)
        n_frames = 0
        while cap.grab():
            n_frames += 1
        cap.release()
    n_clips = n_frames // clip_length
    if n_clips == 0:
        raise ValueError(f'Need at least {clip_length} frames for one complete clip')

    workers = max(1, min(workers or os.cpu_count() or 1, n_clips))
    edges = np.linspace(0, n_clips, workers + 1, dtype=int)
    jobs = [(video_path, int(edges[i]), int(edges[i + 1]), clip_length, rows, cols, i)
            for i in range(workers)]
    if workers == 1:
        parts = [_flow_worker(jobs[0])]
    else:
        with Pool(workers) as pool:
            parts = pool.map(_flow_worker, jobs)
    flow = np.concatenate([p for _, p in sorted(parts, key=lambda x: x[0])], axis=0)

    meta = {'fps': fps, 'total_frames': n_frames, 'clips': n_clips, 'clip_length': clip_length,
            'grid_rows': rows, 'grid_cols': cols, 'source': str(video_path)}
    return flow, meta


def bag_of_words(words, bins):
    """Clip documents: counts over the vocabulary of (cell, direction) words.

    Word id = cell * bins + direction, so the vocabulary size is rows*cols*bins.
    """
    n_clips, _, n_cells = words.shape
    vocab = n_cells * bins
    word_ids = np.arange(n_cells)[None, None, :] * bins + words.astype(np.int64)
    valid = words >= 0
    counts = np.zeros((n_clips, vocab), dtype=np.float32)
    for i in range(n_clips):
        counts[i] = np.bincount(word_ids[i][valid[i]], minlength=vocab)
    return counts


def empty_clips(words, rows, cols, min_steps, inner_rows, inner_cols, max_inner, max_full):
    """True for clips where no vehicle moves through the junction (see config.py)."""
    active = ((words >= 0).sum(axis=1) >= min_steps).reshape(len(words), rows, cols)
    inner = active[:, inner_rows[0]:inner_rows[1], inner_cols[0]:inner_cols[1]].sum(axis=(1, 2))
    return (inner <= max_inner) & (active.sum(axis=(1, 2)) <= max_full)


def reach_features(words, rows, cols, bins):
    """How far across the frame each horizontal flow reaches (one row per clip).

    A car driving along the horizontal road crosses the whole frame, so its westward
    motion appears in every column. A car turning left out of the main road only exits
    on the left, so its westward motion stops around the middle. Measured on the clips:
    westward motion in the right half of the frame is 44% for 'left' clips but only 14%
    for 'turning' clips, so the two really do differ in reach, not in direction.

    Returns [W_mean_column, W_share_in_right_half, E_mean_column, E_share_in_left_half].

    NOTE: these features did NOT improve the classifier (see README). The bag of words
    already carries the same information per cell, and the 'turning' class is too small
    for the networks to exploit it. Kept because it describes the junction usefully and
    because it is the honest answer to "how do you tell a left turn from westward flow".
    """
    if bins != 8:
        raise ValueError('reach_features assumes 8 compass directions')
    E, SE, S, SW, W, NW, N, NE = range(8)
    n = len(words)
    hist = np.stack([(words == d).sum(axis=1).reshape(n, rows, cols) for d in range(bins)], axis=-1)
    column = np.arange(cols)
    out = []
    for dirs in ([W, NW], [E, NE]):
        per_column = hist[..., dirs].sum(-1).sum(1)                 # clips x columns
        total = np.maximum(per_column.sum(1), 1e-6)
        mean_column = (per_column * column).sum(1) / total / (cols - 1)
        right_half = per_column[:, cols // 2:].sum(1) / total
        out += [mean_column, right_half if dirs[0] == W else 1 - right_half]
    return np.stack(out, axis=1).astype(np.float32)


def clip_directions(words, bins):
    """Dominant direction of each clip (-1 if the clip has no moving cells) and its share."""
    n = len(words)
    hist = np.zeros((n, bins))
    for d in range(bins):
        hist[:, d] = (words == d).reshape(n, -1).sum(axis=1)
    total = hist.sum(axis=1)
    dom = np.where(total > 0, hist.argmax(axis=1), -1)
    share = np.where(total > 0, hist.max(axis=1) / np.maximum(total, 1), 0.0)
    return dom, share, hist
