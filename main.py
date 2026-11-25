"""Motion-pattern mining and video event recognition for the QMUL junction video.

Follows chapter 4 of the thesis, with deep learning in place of HDP / HMM-LDA / SVM.

Commands:
  extract  : video -> 75-frame clips -> optical flow -> mean flow per grid cell   (step 1)
  run      : unsupervised: activities (neural topic model, step 2) -> event classes
             (hybrid neural HMM, step 3) -> online classification of test clips (step 4)
  classify : supervised: train on labelled training clips, classify test clips (best accuracy)
  review   : image sheets of sample test clips per class, for checking by eye
  all      : extract + run

Clip split, as in the thesis: the first TRAIN_CLIPS (500) clips are for learning, the rest for testing.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from config import *
from src.activities import (train_topic_model, assign_tokens, select_activities,
                            activity_vectors)
from src.features import extract_cell_flow, quantize_flow, bag_of_words, clip_directions, empty_clips
from src.kmeans import kmeans_numpy
from src.states import train_state_model, filter_states
from src.review import build_review_sheets
from src.classifier import bow_shares, train_ensemble, predict_ensemble, CLASSES
from src.evaluation import (describe_profile, describe_clips, map_states_to_labels, evaluate_test,
                            cluster_agreement, plot_class_motion, COMPASS)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def labels_for(path, n):
    """Read an optional labels CSV (clip_id,label). Clips missing from the file stay None."""
    if not path:
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if not {'clip_id', 'label'}.issubset(df.columns):
        raise ValueError('labels CSV requires clip_id,label columns')
    y = np.array([None] * n, dtype=object)
    for _, row in df.iterrows():
        i = int(row['clip_id'])
        if 0 <= i < n:
            y[i] = str(row['label']).strip()
    return y


def load_words(path):
    """Load flow.npz and turn the stored cell flow into direction words (-1 = no motion)."""
    d = np.load(path, allow_pickle=True)
    meta = d['meta'].item()
    if 'flow' in d.files:
        words = quantize_flow(d['flow'], DIRECTION_BINS, MIN_FLOW_MAGNITUDE)
        meta.update(direction_bins=DIRECTION_BINS, min_flow_magnitude=MIN_FLOW_MAGNITUDE)
    else:
        # Very old output format (words already quantized with the old, shifted bins).
        print('Warning: this file holds old pre-quantized words (bins not centred on compass '
              'directions). Re-run extract for correct direction names.')
        words = d['words']
    return words, meta


def prepare(words_path, train_clips, max_activities):
    """Shared steps 1-2 for `run` and `classify`.

    Returns a dict with: words, meta, counts (bag of words), activities, C (activity
    vectors, thesis eq. 9), empty (empty-junction flags), n_total, n_train.
    """
    words, meta = load_words(words_path)
    counts = bag_of_words(words, meta['direction_bins'])
    n_total = len(counts)
    n_train = min(train_clips, n_total - 1)
    print(f'Clips: {n_total} (train {n_train}, test {n_total - n_train})')

    # Step 2: activities, learned from the training clips only (no peeking at test clips).
    print('Step 2: activity discovery with the neural topic model')
    theta_tr, beta = train_topic_model(counts[:n_train], max_activities, TOPIC_HIDDEN,
                                       TOPIC_EPOCHS, TOPIC_LR, TOPIC_BATCH, RANDOM_SEED)
    n_k = assign_tokens(counts[:n_train], theta_tr, beta)
    activities = select_activities(n_k, beta, TYPICAL_RATIO, NOISE_THRESHOLD)   # eqs. 3-8
    print(f'  topics: {len(beta)}, typical activities kept: {len(activities)}')
    C = activity_vectors(counts, activities)                                       # eq. 9, all clips

    # Empty junction: eq. 9 is normalised per clip, so it cannot tell an empty junction from
    # a busy one. A separate test checks whether any vehicle moves through the junction area.
    empty = empty_clips(words, meta['grid_rows'], meta['grid_cols'], EMPTY_MIN_STEPS,
                        EMPTY_INNER_ROWS, EMPTY_INNER_COLS, EMPTY_MAX_INNER, EMPTY_MAX_FULL)
    print(f'  empty-junction clips (no vehicle through the junction): '
          f'{int(empty.sum())} of {n_total} (train {int(empty[:n_train].sum())})')
    return dict(words=words, meta=meta, counts=counts, activities=activities, C=C,
                empty=empty, n_total=n_total, n_train=n_train)


# ---------------------------------------------------------------------------
# Command: extract (step 1)
# ---------------------------------------------------------------------------

def extract(args):
    """Decode the video, compute optical flow and save the mean flow of every grid cell."""
    flow, meta = extract_cell_flow(args.video, args.clip_length, args.rows, args.cols, args.workers)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, flow=flow, meta=np.array(meta, dtype=object))
    print(f"Cell flow: {flow.shape} (clips, flow steps per clip, cells, dx/dy) saved to {args.out}")
    print(f"Frames: {meta['total_frames']}, clips: {meta['clips']}, fps: {meta['fps']:.2f}")


# ---------------------------------------------------------------------------
# Command: run (unsupervised, steps 2-4)
# ---------------------------------------------------------------------------

def run(args):
    d = prepare(args.words, args.train_clips, args.max_activities)
    words, meta, counts, C, empty = d['words'], d['meta'], d['counts'], d['C'], d['empty']
    n_total, n_train, activities = d['n_total'], d['n_train'], d['activities']
    cells = meta['grid_rows'] * meta['grid_cols']
    bins = meta['direction_bins']
    clip_dir, clip_share, _ = clip_directions(words, bins)    # each clip's own main direction
    moving = (words >= 0).mean(axis=(1, 2))
    labels = labels_for(args.labels, n_total)
    os.makedirs(args.out, exist_ok=True)
    EMPTY_ID = args.clusters                    # the empty class gets the last state number
    busy_train = np.where(~empty[:n_train])[0]

    # Step 3: event classes with the hybrid neural HMM (replaces HMM-LDA), learned on the
    # non-empty training clips. Step 4: test clips are classified online with the causal filter.
    print('Step 3: event-class learning with the hybrid neural HMM')
    model, A, pi, prior, gamma_tr, scaler = train_state_model(
        C[busy_train], args.clusters, HMM_HIDDEN, HMM_EM_ITERS, HMM_MLP_STEPS, HMM_LR, RANDOM_SEED)
    alpha = filter_states(model, C, A, pi, prior, scaler)
    pred = np.where(empty, EMPTY_ID, alpha.argmax(axis=1))
    train_states = np.full(n_train, EMPTY_ID)
    train_states[busy_train] = gamma_tr.argmax(axis=1)

    # Baseline from thesis approach 3: K-means on the same activity vectors.
    _, km_centers = kmeans_numpy(C[busy_train], k=args.clusters, random_state=RANDOM_SEED)
    km_all = np.linalg.norm(C[:, None, :] - km_centers[None], axis=2).argmin(axis=1)
    km_pred = np.where(empty, EMPTY_ID, km_all)

    # Describe each learned class from its training clips (main directions, spread R, sizes).
    profiles, names, resultants, train_n, test_n, top_acts = [], [], [], [], [], []
    for s in range(args.clusters + 1):
        members = np.where(train_states == s)[0]
        if len(members) == 0:
            profiles.append(np.zeros((cells, bins)))
        else:
            norm = counts[members] / np.maximum(counts[members].sum(axis=1, keepdims=True), 1)
            profiles.append(norm.mean(axis=0).reshape(cells, bins))
        _, r = describe_profile(profiles[-1])
        names.append('empty junction' if s == EMPTY_ID else describe_clips(clip_dir[members], bins))
        resultants.append(round(r, 3))
        train_n.append(int(len(members)))
        test_n.append(int(np.sum(pred[n_train:] == s)))
        top_acts.append(int(np.argmax(C[members].mean(axis=0))) if len(members) else -1)

    # If labels exist, give each learned state a label name (Hungarian matching on training clips).
    mapping = {}
    if labels is not None:
        lab_train = [i for i in range(n_train) if labels[i] is not None]
        if lab_train:
            mapping = map_states_to_labels(train_states[lab_train], labels[lab_train].tolist())
            print('Learned state -> label mapping (from training clips):', mapping)
    class_names = [mapping.get(s, names[s]) for s in range(args.clusters + 1)]

    # --- Save results -----------------------------------------------------
    split = np.where(np.arange(n_total) < n_train, 'train', 'test')
    table = {'clip_id': np.arange(n_total), 'split': split, 'empty': empty,
             'moving_fraction': np.round(moving, 4),
             'state_id': pred, 'class_name': [class_names[s] for s in pred],
             'clip_direction': [COMPASS[d] if d >= 0 else 'none' for d in clip_dir],
             'clip_direction_share': np.round(clip_share, 3),
             'kmeans_state': km_pred}
    if labels is not None:
        table['label'] = labels
    pd.DataFrame(table).to_csv(os.path.join(args.out, 'predictions.csv'), index=False)

    pd.DataFrame({'state_id': range(args.clusters + 1), 'class_name': class_names,
                  'motion_description': names, 'resultant_length': resultants,
                  'train_clips': train_n, 'test_clips': test_n,
                  'top_activity': top_acts}).to_csv(os.path.join(args.out, 'classes.csv'), index=False)

    # Activities: the top (cell, direction) words of each one, for inspection.
    act_report = []
    for i, act in enumerate(activities):
        top = np.argsort(-act['probs'])[:10]
        words_txt = []
        for w in top:
            if act['probs'][w] <= 0:
                continue
            cell, direction = divmod(int(w), bins)
            words_txt.append({'cell': [cell // meta['grid_cols'], cell % meta['grid_cols']],
                              'direction': COMPASS[direction] if bins == 8 else direction,
                              'probability': round(float(act['probs'][w]), 5)})
        act_report.append({'activity': i, 'topic': act['topic'], 'share_of_tokens': round(act['ratio'], 4),
                           'words_kept': int(len(act['words'])), 'top_words': words_txt})
    with open(os.path.join(args.out, 'activities.json'), 'w') as f:
        json.dump(act_report, f, indent=2)

    plot_class_motion(profiles[:args.clusters], class_names[:args.clusters], meta['grid_rows'],
                      meta['grid_cols'], os.path.join(args.out, 'class_motion.png'))

    joblib.dump({'activities': activities, 'A': A, 'pi': pi, 'prior': prior, 'scaler': scaler,
                 'km_centers': km_centers}, os.path.join(args.out, 'model_info.joblib'))
    torch.save(model.state_dict(), os.path.join(args.out, 'state_network.pt'))

    # --- Evaluation (only if labels were supplied) ------------------------
    print('Event classes:')
    for s in range(args.clusters + 1):
        print(f'  state {s}: {class_names[s]} | train {train_n[s]} | test {test_n[s]} | R={resultants[s]}')
    if labels is None:
        print('No labels supplied: class discovery and test classification ran, accuracy was skipped.')
    else:
        test_idx = [i for i in range(n_train, n_total) if labels[i] is not None]
        if not test_idx:
            print('No labelled test clips in the labels file: test metrics skipped.')
        else:
            if mapping:
                evaluate_test(pred[test_idx], [labels[i] for i in test_idx], mapping, args.out)
            ari, nmi = cluster_agreement(pred[test_idx], [labels[i] for i in test_idx])
            print(f'Test ARI: {ari:.4f}  NMI: {nmi:.4f}')
            # K-means states get their own Hungarian mapping (their numbers differ from the HMM's).
            lab_train = [i for i in range(n_train) if labels[i] is not None]
            km_map = map_states_to_labels(km_pred[lab_train], labels[lab_train].tolist())
            km_named = np.array([km_map.get(int(s), 'unmatched') for s in km_pred[test_idx]])
            km_acc = float(np.mean(km_named == np.array([labels[i] for i in test_idx])))
            print('K-means baseline (own mapping) test accuracy:', round(km_acc, 4))
    print(f'Outputs written to {args.out}')


# ---------------------------------------------------------------------------
# Command: classify (supervised, best accuracy)
# ---------------------------------------------------------------------------

def classify(args):
    d = prepare(args.words, args.train_clips, args.max_activities)
    counts, C, empty = d['counts'], d['C'], d['empty']
    n_total, n_train = d['n_total'], d['n_train']
    labels = labels_for(args.labels, n_total)
    if labels is None:
        raise SystemExit('classify needs --labels (at least for the training clips)')
    os.makedirs(args.out, exist_ok=True)

    # Features for the two networks: word shares (A) and activity vectors (B).
    bow = bow_shares(counts)
    act = C.astype(np.float32)

    # Which classes to learn: those that actually appear in the labels ('empty' is decided
    # by the empty-junction test, not by the networks). This way the same command works
    # with or without the 'mixed' class in the labels file.
    classes = [c for c in CLASSES if c in set(labels[labels != None])]
    print(f'  direction classes in the labels: {classes}')

    # Train only on labelled, non-empty training clips.
    train_idx = np.array([i for i in range(n_train)
                          if labels[i] in classes and not empty[i]])
    print('Step 4: supervised classification with an ensemble of neural networks')
    models = train_ensemble(bow, act, labels, train_idx, classes=classes, seeds=args.seeds)
    names, probs = predict_ensemble(models, bow, act, empty)

    # Save predictions (with class probabilities) and the trained networks.
    split = np.where(np.arange(n_total) < n_train, 'train', 'test')
    clip_dir, clip_share, _ = clip_directions(d['words'], d['meta']['direction_bins'])
    table = pd.DataFrame({'clip_id': np.arange(n_total), 'split': split, 'empty': empty,
                          'predicted': names,
                          'clip_direction': [COMPASS[k] if k >= 0 else 'none' for k in clip_dir],
                          'clip_direction_share': np.round(clip_share, 3)})
    for k, c in enumerate(classes):
        table[f'p_{c}'] = np.round(probs[:, k], 4)
    table['label'] = labels
    table.to_csv(os.path.join(args.out, 'predictions.csv'), index=False)
    torch.save({'classes': classes,
                'networks': {k: [m.state_dict() for m in models[k]] for k in ('bow', 'act')}},
               os.path.join(args.out, 'classifier_networks.pt'))

    # Accuracy on the labelled test clips (the 700 clips after the training part).
    test_idx = [i for i in range(n_train, n_total) if labels[i] is not None]
    if test_idx:
        all_names = sorted(set(classes) | {'empty'})
        ids = {c: i for i, c in enumerate(all_names)}
        evaluate_test(np.array([ids[names[i]] for i in test_idx]), [labels[i] for i in test_idx],
                      {i: c for c, i in ids.items()}, args.out)
    print(f'Outputs written to {args.out}')


# ---------------------------------------------------------------------------
# Command: review (image sheets for checking by eye)
# ---------------------------------------------------------------------------

def review(args):
    build_review_sheets(args.video, args.predictions, args.out, CLIP_LENGTH, GRID_ROWS, GRID_COLS,
                        per_class=args.per_class, min_magnitude=MIN_FLOW_MAGNITUDE)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='cmd', required=True)

    a = s.add_parser('extract', help='video -> mean optical flow per grid cell')
    a.add_argument('--video', required=True)
    a.add_argument('--out', default='outputs/flow.npz')
    a.add_argument('--clip-length', type=int, default=CLIP_LENGTH)
    a.add_argument('--rows', type=int, default=GRID_ROWS)
    a.add_argument('--cols', type=int, default=GRID_COLS)
    a.add_argument('--workers', type=int, default=None, help='parallel processes (default: all cores)')
    a.set_defaults(func=extract)

    a = s.add_parser('run', help='unsupervised: activities -> event classes -> test classification')
    a.add_argument('--words', required=True, help='flow.npz from extract')
    a.add_argument('--labels', help='optional clip_id,label CSV for evaluation')
    a.add_argument('--out', default='outputs/run')
    a.add_argument('--train-clips', type=int, default=TRAIN_CLIPS)
    a.add_argument('--max-activities', type=int, default=MAX_ACTIVITIES)
    a.add_argument('--clusters', type=int, default=N_CLUSTERS)
    a.set_defaults(func=run)

    a = s.add_parser('classify', help='supervised: train on labelled training clips, classify test clips')
    a.add_argument('--words', required=True, help='flow.npz from extract')
    a.add_argument('--labels', required=True, help='clip_id,label CSV (training clips needed)')
    a.add_argument('--out', default='outputs/classify')
    a.add_argument('--train-clips', type=int, default=TRAIN_CLIPS)
    a.add_argument('--max-activities', type=int, default=MAX_ACTIVITIES)
    a.add_argument('--seeds', type=int, default=3, help='networks per feature type in the ensemble')
    a.set_defaults(func=classify)

    a = s.add_parser('review', help='sample clips per detected class for checking by eye')
    a.add_argument('--video', required=True)
    a.add_argument('--predictions', required=True)
    a.add_argument('--out', default='outputs/review')
    a.add_argument('--per-class', type=int, default=6)
    a.set_defaults(func=review)

    a = s.add_parser('all', help='extract and run in one go')
    a.add_argument('--video', required=True)
    a.add_argument('--labels')
    a.add_argument('--out', default='outputs')
    a.set_defaults(func=None)

    args = p.parse_args()
    if args.cmd == 'all':
        os.makedirs(args.out, exist_ok=True)
        words_path = os.path.join(args.out, 'flow.npz')
        extract(argparse.Namespace(video=args.video, out=words_path, clip_length=CLIP_LENGTH,
                                   rows=GRID_ROWS, cols=GRID_COLS, workers=None))
        run(argparse.Namespace(words=words_path, labels=args.labels,
                               out=os.path.join(args.out, 'run'), train_clips=TRAIN_CLIPS,
                               max_activities=MAX_ACTIVITIES, clusters=N_CLUSTERS))
    else:
        args.func(args)


if __name__ == '__main__':
    main()
