"""Image sheets for checking the results by eye.

For every predicted class, a few random test clips are drawn: one frame from the middle of
the clip with the mean optical flow of each grid cell as a red arrow, plus the clip number,
the class and the clip's own main direction. A CSV template for writing manual labels is saved too.
"""
import os

import cv2
import numpy as np
import pandas as pd


def clip_preview(cap, clip_id, clip_length, rows, cols, text_lines, min_magnitude):
    """Two frames from the middle of a clip -> frame with the mean flow of each cell as an arrow."""
    mid = clip_id * clip_length + clip_length // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid - 1)
    ok1, a = cap.read()
    ok2, b = cap.read()
    if not (ok1 and ok2):
        return None
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = ga.shape
    vis = a.copy()
    re = np.linspace(0, h, rows + 1, dtype=int)
    ce = np.linspace(0, w, cols + 1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            cy, cx = (re[r] + re[r + 1]) // 2, (ce[c] + ce[c + 1]) // 2
            region = flow[re[r]:re[r + 1], ce[c]:ce[c + 1]]
            mdx, mdy = region[..., 0].mean(), region[..., 1].mean()
            if np.hypot(mdx, mdy) >= min_magnitude:
                tip = (int(cx + mdx * 5), int(cy + mdy * 5))
                cv2.arrowedLine(vis, (cx, cy), tip, (0, 0, 255), 1, tipLength=0.35)
    # Caption band above the frame.
    band = np.zeros((18 * len(text_lines) + 4, w, 3), dtype=np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(band, line, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([band, vis])


def build_review_sheets(video, predictions_csv, out_dir, clip_length, rows, cols,
                        per_class=6, columns=3, seed=0, min_magnitude=0.5, log=print):
    """One PNG per class with `per_class` random test clips, and a labels template CSV."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(predictions_csv)
    if 'class_name' not in df.columns and 'predicted' in df.columns:   # output of `classify`
        df['class_name'] = df['predicted']
    cap = cv2.VideoCapture(video)
    rng = np.random.default_rng(seed)
    for name in sorted(df['class_name'].unique()):
        pool = df[(df['class_name'] == name) & (df['split'] == 'test')]
        if pool.empty:
            continue
        picks = pool.sample(min(per_class, len(pool)), random_state=int(rng.integers(1 << 30)))
        tiles = []
        for _, row in picks.iterrows():
            lines = [f"clip {int(row['clip_id'])} | class: {row['class_name']}",
                     f"this clip: {row.get('clip_direction', '?')} "
                     f"({float(row.get('clip_direction_share', 0)):.0%} of its moving cells)"]
            tile = clip_preview(cap, int(row['clip_id']), clip_length, rows, cols, lines, min_magnitude)
            if tile is not None:
                tiles.append(tile)
        if not tiles:
            continue
        # Pad with black tiles so the grid is rectangular, then stack rows of tiles.
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
        safe = ''.join(ch if ch.isalnum() else '_' for ch in name)[:60]
        path = os.path.join(out_dir, f'review_{safe}.png')
        cv2.imwrite(path, grid)
        log(f'  {name}: {len(tiles)} sample clips -> {path}')
    cap.release()
    df.assign(label='').to_csv(os.path.join(out_dir, 'manual_labels_template.csv'), index=False,
                               columns=[c for c in ['clip_id', 'split', 'class_name', 'clip_direction', 'label']
                                        if c in df.columns or c == 'label'])
    log(f'Template for your labels: {os.path.join(out_dir, "manual_labels_template.csv")}')
