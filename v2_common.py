"""Shared settings and helpers for the corrected protocol (PROTOCOL v2).

Every v2 script imports its settings from here, so the frozen protocol lives in
one place. Nothing in this module trains a model.

The 3 GB feature file is parsed once and cached in cache/ as a sparse matrix
plus a sample index; later runs reuse the cache only if the input files still
have the same SHA-256.
"""
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from array import array
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.svm import LinearSVC

# ------------------------------------------------------------------------------
# Frozen protocol settings (PROTOCOL v2)
# ------------------------------------------------------------------------------
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"
FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"   # original (2021-2023)
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results/v2")

DATE_COL = "gp_date"
VTT = 2                               # malware if vt_detection >= 2 (Hypercube definition)
DEV_START, DEV_END = "2021-01", "2023-12"
TEST_START, TEST_END = "2022-01", "2023-12"
EXPECTED_DEV_SAMPLES = 196_444        # from the dataset audit (3 empty vectors removed)

# C selection, done once inside 2021 (see select_c.py). Selection metric:
# unweighted mean of the monthly F1 scores of October, November and December.
C_GRID = (0.01, 0.1, 1.0, 10.0)
C_TRAIN = ("2021-01", "2021-09")
C_VALID_MONTHS = ("2021-10", "2021-11", "2021-12")
C_TIE_DECIMALS = 3                    # equal mean F1 at 3 decimals -> smaller C wins

# Classifier (LIBLINEAR through scikit-learn's LinearSVC)
SEED = 42
SVM_PARAMS = dict(
    penalty="l2",
    loss="squared_hinge",             # LIBLINEAR default L2-loss SVM
    dual=True,                        # many more features than samples
    class_weight=None,                # primary Drebin replication: no error-cost reweighting
    fit_intercept=True,
    intercept_scaling=1.0,
    tol=1e-4,
    max_iter=10_000,
    random_state=SEED,
)
DECISION_THRESHOLD = 0.0              # standard SVM threshold, not tuned

# Variants. "primary" is the Drebin replication; "balanced" is the pre-registered
# cost-sensitive sensitivity analysis. Each variant selects its own C with the
# same rule (class weights can change the best C) and writes to its own folder.
VARIANTS = {
    "primary": {"class_weight": None, "results_dir": Path("results/v2")},
    "balanced": {"class_weight": "balanced", "results_dir": Path("results/v2/sensitivity_balanced")},
}


def svm_params(variant="primary"):
    return {**SVM_PARAMS, "class_weight": VARIANTS[variant]["class_weight"]}


def results_dir(variant="primary"):
    return VARIANTS[variant]["results_dir"]

# Uncertainty
N_BOOTSTRAP = 1000                    # within-month bootstrap for F1 CIs

# Feature groups (Drebin S1-S8). S1 (hardware) is not present in Hypercube.
PREFIX_TO_GROUP = {
    "hardware": "S1",
    "app_permissions": "S2",
    "permission": "S2",
    "activities": "S3",
    "services": "S3",
    "providers": "S3",
    "receivers": "S3",
    "intents": "S4",
    "api_calls": "S5",                # restricted API calls
    "api_permissions": "S6",          # used permissions
    "interesting_calls": "S7",
    "urls": "S8",
}
GROUP_NAMES = {
    "S1": "Hardware components", "S2": "Requested permissions", "S3": "App components",
    "S4": "Filtered intents", "S5": "Restricted API calls", "S6": "Used permissions",
    "S7": "Suspicious API calls", "S8": "Network addresses",
}


# ------------------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_range(start, end):
    """Inclusive list of 'YYYY-MM' strings."""
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def shift_month(month, k):
    return str(pd.Period(month, freq="M") + k)


def run_info():
    """Provenance stored next to every result file."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_tag": git("describe", "--tags", "--exact-match"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------------------
# Data loading and caching
# ------------------------------------------------------------------------------
def _build_cache(meta_hash, feat_hash):
    """Parse metadata + features once and write the cache files."""
    print("Building the development cache (first run only, this takes a while)...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = pd.DataFrame.from_dict(raw, orient="index") if isinstance(raw, dict) else pd.DataFrame(raw)
    del raw
    if "sha256" not in meta.columns:
        meta["sha256"] = meta.index
    meta = meta.reset_index(drop=True)

    meta["month"] = pd.to_datetime(meta[DATE_COL], errors="coerce").dt.strftime("%Y-%m")
    vt = pd.to_numeric(meta["vt_detection"], errors="raise")
    meta["label"] = (vt >= VTT).astype(np.int8)
    meta = meta[meta["month"].between(DEV_START, DEV_END)]

    print("  loading feature file...")
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feats = json.load(f)

    # Keep development samples that have a non-empty feature vector.
    meta = meta[meta["sha256"].map(lambda s: bool(feats.get(s)))]
    # Deterministic order: by month, then hash.
    meta = meta.sort_values(["month", "sha256"]).reset_index(drop=True)
    if len(meta) != EXPECTED_DEV_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_DEV_SAMPLES:,} development samples "
                           f"(dataset audit), found {len(meta):,}.")

    # One global column index is only an indexing device: every model later
    # keeps just the columns observed in its own training rows, which is the
    # same as fitting the vocabulary on the training split alone.
    print("  building sparse matrix...")
    vocab = {}
    indptr, indices = array("q", [0]), array("i")
    digests = []
    for sha in meta["sha256"]:
        keys = sorted(k for k, v in feats[sha].items() if v)
        for k in keys:
            j = vocab.get(k)
            if j is None:
                j = vocab[k] = len(vocab)
            indices.append(j)
        indptr.append(len(indices))
        # Digest of the feature set, used for the train-test overlap report.
        digests.append(hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest())
    del feats

    X = csr_matrix(
        (np.ones(len(indices), dtype=np.float32),
         np.frombuffer(indices, dtype=np.int32),
         np.frombuffer(indptr, dtype=np.int64)),
        shape=(len(meta), len(vocab)),
    )
    X.sort_indices()

    index = meta[["sha256", "month", "label"]].copy()
    index["digest"] = digests
    feature_names = [None] * len(vocab)
    for k, j in vocab.items():
        feature_names[j] = k

    CACHE_DIR.mkdir(exist_ok=True)
    save_npz(CACHE_DIR / "dev_X.npz", X)
    index.to_csv(CACHE_DIR / "dev_index.csv", index=False)
    (CACHE_DIR / "dev_features.txt").write_text("\n".join(feature_names), encoding="utf-8")
    write_json(CACHE_DIR / "dev_cache.json", {
        "metadata_sha256": meta_hash, "features_sha256": feat_hash,
        "n_samples": X.shape[0], "n_features": X.shape[1], "nnz": int(X.nnz),
    })
    print(f"  cached {X.shape[0]:,} samples x {X.shape[1]:,} features ({X.nnz:,} non-zeros)")


def load_development_data():
    """Return (X, index, feature_names, provenance) for 2021-2023.

    X is a CSR matrix with one row per sample (sorted by month, then hash);
    index has columns sha256, month, label, digest.
    """
    print("Hashing input files...")
    meta_hash, feat_hash = sha256_file(METADATA_PATH), sha256_file(FEATURES_PATH)
    info_path = CACHE_DIR / "dev_cache.json"
    cached = json.loads(info_path.read_text()) if info_path.exists() else {}
    if cached.get("metadata_sha256") != meta_hash or cached.get("features_sha256") != feat_hash:
        _build_cache(meta_hash, feat_hash)
    else:
        print("Using cached development matrix (input hashes match).")

    X = load_npz(CACHE_DIR / "dev_X.npz").tocsr()
    index = pd.read_csv(CACHE_DIR / "dev_index.csv", dtype={"month": str})
    feature_names = (CACHE_DIR / "dev_features.txt").read_text(encoding="utf-8").split("\n")
    assert X.shape == (len(index), len(feature_names))
    provenance = {
        "metadata": {"path": METADATA_PATH, "sha256": meta_hash},
        "features": {"path": FEATURES_PATH, "sha256": feat_hash},
        "n_samples": int(X.shape[0]), "n_features_dev": int(X.shape[1]),
    }
    return X, index, feature_names, provenance


def feature_groups(feature_names):
    """Array with the S-group of every column; unknown prefixes are an error."""
    groups = []
    for name in feature_names:
        prefix = name.split("::", 1)[0]
        if prefix not in PREFIX_TO_GROUP:
            raise ValueError(f"Unmapped feature prefix '{prefix}' in '{name}'")
        groups.append(PREFIX_TO_GROUP[prefix])
    return np.array(groups)


# ------------------------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------------------------
def fit_svm(X_train, y_train, C, allowed_cols=None, variant="primary"):
    """Fit LinearSVC on the columns seen in the training rows.

    allowed_cols optionally restricts the candidate columns (used for the
    feature-group models). Returns (model, kept_columns, fit_info).
    """
    seen = np.unique(X_train.indices)
    cols = seen if allowed_cols is None else np.intersect1d(seen, allowed_cols, assume_unique=True)
    params = svm_params(variant)
    clf = LinearSVC(C=C, **params)
    start = datetime.now()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train[:, cols], y_train)
    seconds = (datetime.now() - start).total_seconds()
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    n_iter = int(np.max(clf.n_iter_))
    info = {"n_train": int(X_train.shape[0]), "n_features": int(len(cols)),
            "n_iter": n_iter, "converged": bool(converged and n_iter < params["max_iter"]),
            "fit_seconds": round(seconds, 1)}
    return clf, cols, info


def decision_scores(clf, cols, X):
    return clf.decision_function(X[:, cols])


def evaluate(y_true, scores, rng=None):
    """Metrics at the frozen threshold, AP from the scores, bootstrap CI for F1."""
    y_true = np.asarray(y_true).astype(np.int8)
    if len(y_true) == 0:   # e.g. every test sample removed by the overlap sensitivity analysis
        keys = ["precision", "recall", "f1", "fpr", "average_precision", "f1_ci_low", "f1_ci_high"]
        return {"n": 0, "n_malware": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, **{k: np.nan for k in keys}}
    y_pred = (scores > DECISION_THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def f1_from(tp, fp, fn):
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)

    out = {
        "n": int(len(y_true)), "n_malware": int(y_true.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": float(f1_from(tp, fp, fn)),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "average_precision": float(average_precision_score(y_true, scores)) if 0 < y_true.sum() < len(y_true) else np.nan,
    }

    # Percentile bootstrap over the samples of this month (F1 only).
    rng = rng or np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_true), size=(N_BOOTSTRAP, len(y_true)))
    yt, yp = y_true[idx], y_pred[idx]
    b_tp = (yt & yp).sum(1)
    b_fp = ((1 - yt) & yp).sum(1)
    b_fn = (yt & (1 - yp)).sum(1)
    b_f1 = f1_from(b_tp, b_fp, b_fn)
    out["f1_ci_low"], out["f1_ci_high"] = (float(v) for v in np.percentile(b_f1, [2.5, 97.5]))
    return out
