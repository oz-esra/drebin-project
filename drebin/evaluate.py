"""Evaluation and benchmarking pipeline for Drebin malware detection.

Handles leak-free train/test splitting, feature vectorization, LinearSVC training,
and paper-aligned evaluation metrics (F1, ROC-AUC, Detection Rate @ 1% FPR).

If ground truth labels are missing (--labels omitted), uniform dummy labels are
generated to validate end-to-end execution. Under random labels, metrics should 
converge to F1 ~0.5 and AUC ~0.5; higher scores indicate data leakage.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.svm import LinearSVC

from .vectorize import fit, load_records, transform


def deduplicate(records):
    """Deduplicate records by SHA256 hash prior to splitting.

    Drebin dataset contains ~50% repackaged samples. Deduplicating before train/test 
    partitioning prevents identical apps from appearing in both sets, which would 
    otherwise cause optimistic metric inflation.
    """
    seen = set()
    unique = []
    duplicates = 0
    for r in records:
        if r["sha256"] in seen:
            duplicates += 1
            continue
        seen.add(r["sha256"])
        unique.append(r)
    return unique, duplicates


def split(records, test_fraction=0.33, seed=0):
    """Perform deterministic train/test split.

    Uses a fixed random seed to ensure reproducibility. Default test split (0.33) 
    matches the 66% / 33% partitioning strategy used in Arp et al. (2014).
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]


def labels_of(records, label_map):
    """Map SHA256 hashes to binary targets (0: Goodware, 1: Malware)."""
    return np.array([label_map[r["sha256"]] for r in records], dtype=np.int8)


def placeholder_labels(records, seed=0):
    """Generate reproducible dummy binary labels for end-to-end testing."""
    rng = random.Random(seed)
    return {r["sha256"]: rng.randint(0, 1) for r in records}


def detection_rate_at_fpr(y_true, scores, target_fpr=0.01):
    """Compute True Positive Rate (TPR) at a fixed target False Positive Rate (FPR).

    Primary metric evaluated in Arp et al. (2014), where they reported ~94% TPR at 1% FPR.
    More representative than raw accuracy under strict operational false-alarm limits.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="drebin-evaluate")
    parser.add_argument("indir", help="directory of extraction JSON records")
    parser.add_argument("--labels", help="JSON mapping sha256 -> 0/1")
    parser.add_argument("--test-fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = load_records(Path(args.indir))
    records, duplicates = deduplicate(records)
    print(f"records      : {len(records)}  ({duplicates} duplicate hashes dropped)")

    if args.labels:
        label_map = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        print("labels       : loaded from file")
    else:
        label_map = placeholder_labels(records, args.seed)
        print("labels       : PLACEHOLDER (random) - results are not meaningful")

    train, test = split(records, args.test_fraction, args.seed)
    print(f"train / test : {len(train)} / {len(test)}")

    # Core anti-leakage mechanism: vocabulary is fitted EXCLUSIVELY on train data.
    # Test set is vectorized strictly using the learned training vocabulary.
    vocabulary = fit(train)
    X_train = transform(train, vocabulary)
    X_test = transform(test, vocabulary)
    y_train = labels_of(train, label_map)
    y_test = labels_of(test, label_map)

    seen_test = set()
    for r in test:
        seen_test.update(r["features"])
    print(f"|S| (train)  : {len(vocabulary)}")
    print(f"OOV in test  : {len(seen_test - set(vocabulary))} features dropped")

    if len(set(y_train)) < 2:
        print("\ntraining set has a single class - cannot fit a classifier")
        return 1

    # LinearSVC matches original implementation and allows direct feature weight 
    # analysis for model interpretability (Section II-D).
    model = LinearSVC(C=1.0, max_iter=10000)
    model.fit(X_train, y_train)
    scores = model.decision_function(X_test)
    y_pred = (scores > 0).astype(np.int8)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    print(f"\nprecision    : {precision:.3f}")
    print(f"recall       : {recall:.3f}")
    print(f"f1           : {f1:.3f}")
    if len(set(y_test)) == 2:
        print(f"roc auc      : {roc_auc_score(y_test, scores):.3f}")
        print(f"TPR @ 1% FPR : {detection_rate_at_fpr(y_test, scores):.3f}")
    else:
        print("roc auc      : n/a (test set has a single class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())