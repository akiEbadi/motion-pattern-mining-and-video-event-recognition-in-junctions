import numpy as np, pandas as pd
s = np.load('/home/claude/work/labeling/shares.npy'); mov = np.load('/home/claude/work/labeling/moving.npy')
E, SE, S, SW, W, NW, N, NE = s.T
COMP = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
import sys; sys.path.insert(0, '/home/claude/work/project')
from config import *
from src.features import quantize_flow, empty_clips
_w = quantize_flow(np.load('/home/claude/work/outputs/flow.npz', allow_pickle=True)['flow'].astype(np.float32), 8, 0.5)
EMPTY = empty_clips(_w, 10, 10, EMPTY_MIN_STEPS, EMPTY_INNER_ROWS, EMPTY_INNER_COLS, EMPTY_MAX_INNER, EMPTY_MAX_FULL)
rows = []
for i in range(len(s)):
    up = N[i] + NE[i] + NW[i]
    doubt = ''
    if EMPTY[i]:
        lab = 'empty'
    elif W[i] >= 0.2 and E[i] >= 0.2 and up + SE[i] < 0.15:
        lab = 'left' if W[i] >= E[i] else 'right'
        doubt = 'both horizontal directions, no turning vehicles'
    elif W[i] >= 0.2 and SE[i] >= 0.2:
        lab = 'turning' if N[i] >= 0.15 else 'left'
        doubt = 'W+SE: westbound flow (or left turn) + southbound (SE) at the same time'
    elif W[i] >= 0.55:
        lab = 'left'
    elif E[i] >= 0.4 and E[i] + SE[i] >= 0.8:
        lab = 'right'
    elif N[i] + NE[i] >= 0.2 and E[i] >= 0.2:
        lab = 'turning'                      # right turn from the main road
    elif N[i] >= 0.15 and W[i] >= 0.2:
        lab = 'turning'                      # left turn from the main road
    elif W[i] >= 0.15 and E[i] >= 0.15:
        lab = 'turning'
    elif E[i] >= 0.55:
        lab = 'right'
    elif 1 - E[i] - W[i] >= 0.6 and E[i] < 0.2 and W[i] < 0.2:
        lab = 'vertical'
        if NW[i] >= max(N[i], SE[i]):
            doubt = 'mainly NW (up-left), not a clear N/SE pattern'
    else:
        grp = {'left': W[i], 'right': E[i], 'vertical': 1 - E[i] - W[i]}
        lab = max(grp, key=grp.get)
        doubt = 'no clear pattern'
    low = (not EMPTY[i]) and mov[i] < 0.045
    top = ' '.join(f'{COMP[d]} {s[i, d]:.0%}' for d in np.argsort(-s[i])[:3] if s[i, d] >= 0.1)
    rows.append((i, lab, doubt, bool(low), round(float(mov[i]), 3), top))
USER = {1111: 'left'}   # 749, 661, 1126, 672, 1087, 785: now decided by the single-car rule
rows = [(i, USER.get(i, lab), '' if i in USER else doubt, low, m, top) for i, lab, doubt, low, m, top in rows]
VISUAL = {385: 'turning', 422: 'turning', 616: 'turning', 617: 'turning', 766: 'turning', 1072: 'turning',
          1173: 'turning', 505: 'turning', 432: 'turning', 5: 'vertical', 857: 'vertical', 36: 'right'}
rows = [(i, VISUAL.get(i, lab), '' if i in VISUAL else doubt, low, m, top) for i, lab, doubt, low, m, top in rows]
df = pd.DataFrame(rows, columns=['clip_id', 'label', 'doubt_reason', 'low_traffic', 'moving_fraction', 'directions'])
df['source'] = ['you' if i in USER else ('visual check' if i in VISUAL else 'rule') for i in df.clip_id]
df.to_csv('/home/claude/work/labeling/labels_auto.csv', index=False)
print(df.label.value_counts().to_dict())
print('doubtful:', (df.doubt_reason != '').sum(), df[df.doubt_reason != ''].doubt_reason.value_counts().to_dict())
print('low traffic (0.015-0.045):', df.low_traffic.sum(), df[df.low_traffic].label.value_counts().to_dict())
for i in [1111, 1171, 749, 661, 1126, 1087, 785, 672]:
    r = df.iloc[i]; print(i, r.label, r.low_traffic, r.directions, r.doubt_reason)
