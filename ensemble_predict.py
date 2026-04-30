"""
ensemble predictor

loads the saved softmax outputs from multiple training runs (different seeds)
and averages them. prints per-seed accuracy and the ensemble accuracy plus
a full classification report.

usage (after running train_model.py with --seed 42, 123, 456):
    python ensemble_predict.py
    python ensemble_predict.py --seeds 42 123 456 789

writes:
    models/class_report_ensemble.json
    models/confusion_matrix_ensemble.png
"""

import os, glob, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']
MODEL_DIR = "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='which seeds to ensemble. defaults to whatever is found on disk.')
    args = parser.parse_args()

    # auto-detect seeds if not given on the cli
    if args.seeds is None:
        files = sorted(glob.glob(os.path.join(MODEL_DIR, 'preds_v5b_seed*.npy')))
        seeds = []
        for f in files:
            base = os.path.basename(f)
            seed = int(base.replace('preds_v5b_seed', '').replace('.npy', ''))
            seeds.append(seed)
        seeds.sort()
    else:
        seeds = args.seeds

    if len(seeds) < 2:
        print(f"[error] need at least 2 seeds for an ensemble. found: {seeds}")
        print("[hint] run train_model.py with --seed 42, --seed 123, --seed 456")
        return

    print(f"[ensemble] combining seeds: {seeds}")

    # ground truth
    ytrue_path = os.path.join(MODEL_DIR, 'ytrue_v5b.npy')
    if not os.path.exists(ytrue_path):
        print(f"[error] {ytrue_path} not found. run train_model.py first.")
        return
    y_true = np.load(ytrue_path)

    # load each seed's predictions
    all_preds = []
    print("\nper-seed accuracy:")
    for seed in seeds:
        path = os.path.join(MODEL_DIR, f'preds_v5b_seed{seed}.npy')
        if not os.path.exists(path):
            print(f"  seed {seed}: not found ({path})")
            continue
        preds = np.load(path)
        # truncate to ytrue length if they differ slightly (last-batch oddity)
        n = min(len(preds), len(y_true))
        preds = preds[:n]
        acc = np.mean(np.argmax(preds, axis=1) == y_true[:n])
        print(f"  seed {seed}: {acc*100:.2f}%")
        all_preds.append(preds)

    if len(all_preds) < 2:
        print("[error] fewer than 2 valid prediction files loaded.")
        return

    # ensemble: simple softmax averaging across seeds
    n = min(len(p) for p in all_preds)
    stacked = np.stack([p[:n] for p in all_preds], axis=0)
    ensemble = stacked.mean(axis=0)
    y_pred = np.argmax(ensemble, axis=1)
    y_true = y_true[:n]

    acc = float(np.mean(y_pred == y_true))
    print(f"\n[ensemble] accuracy: {acc*100:.2f}%")

    # use the saved class label map so order matches what the generator saw
    label_map_path = os.path.join(MODEL_DIR, 'class_labels.json')
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            label_map = json.load(f)
        class_names = [label_map[str(i)] for i in range(len(label_map))]
    else:
        class_names = EMOTIONS

    print("\nensemble classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    rpt = classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0, output_dict=True)
    rpt['ensemble_seeds'] = seeds
    rpt['ensemble_size']  = len(all_preds)
    rpt['tta']            = True
    with open(os.path.join(MODEL_DIR, 'class_report_ensemble.json'), 'w') as f:
        json.dump(rpt, f, indent=2)
    print("[saved] models/class_report_ensemble.json")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Ensemble confusion matrix ({len(all_preds)} models, with TTA)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_ensemble.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("[saved] models/confusion_matrix_ensemble.png")

    print("\n" + "=" * 55)
    print(f"  ensemble - {len(all_preds)} models, seeds {seeds}")
    print(f"  final accuracy (ensemble + tta): {acc*100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
