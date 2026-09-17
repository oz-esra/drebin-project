"""Vectorizes extracted features into sparse binary matrices (Arp et al., Section II-B, $\phi(x)$ mapping).

Explicitly separates fit and transform operations to prevent data leakage during pipeline 
evaluation. The feature space must be derived exclusively from the training split. When 
vectorizing test samples, a pre-fitted vocabulary is provided, and unseen out-of-vocabulary 
(OOV) features are discarded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def load_records(indir: Path) -> list[dict]:
    """Load extraction JSON records from target directory.

    Sorts records explicitly by SHA256 to guarantee deterministic row alignment. 
    This ensures identical matrix construction across execution environments.
    """
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(indir.glob("*.json"))]
    records.sort(key=lambda r: r["sha256"])
    return records


def fit(records: list[dict]) -> dict[str, int]:
    """Construct feature space S mapping feature strings to unique column indices.

    Matches feature set S from Arp et al. Features are lexicographically sorted 
    to enforce reproducible column indexing across builds.
    """
    vocabulary = set()
    for r in records:
        vocabulary.update(r.get("features", []))
    return {feature: i for i, feature in enumerate(sorted(vocabulary))}


def transform(records: list[dict], vocabulary: dict[str, int]) -> sparse.csr_matrix:
    """Transform extraction records into a sparse binary matrix $\phi(x)$.

    Constructs CSR arrays (indices, indptr) manually for memory efficiency, avoiding 
    dense allocations over high-dimensional feature spaces ($|S| > 10^5$). Features 
    absent from the provided vocabulary are omitted.
    """
    indptr = [0]
    indices = []
    for r in records:
        # Enforce binary representation φ(x) ∈ {0, 1}^|S| via set reduction
        columns = {vocabulary[f] for f in r.get("features", []) if f in vocabulary}
        indices.extend(sorted(columns))
        indptr.append(len(indices))

    # Binary indicators stored as int8 to minimize memory footprint
    data = np.ones(len(indices), dtype=np.int8)
    return sparse.csr_matrix(
        (data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
        shape=(len(records), len(vocabulary)),
    )


def sample_index(records: list[dict]) -> list[dict]:
    """Generate row index metadata mapping matrix rows to sample IDs.

    Decouples raw matrix data from sample identity. Ground truth labels default 
    to None and are populated downstream without re-indexing the matrix.
    """
    return [
        {
            "row": i,
            "sha256": r["sha256"],
            "package": r.get("package"),
            "file": r.get("file"),
            "partial": r.get("partial", False),
            "label": None,
        }
        for i, r in enumerate(records)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="drebin-vectorize")
    parser.add_argument("indir", help="directory of extraction JSON records")
    parser.add_argument("-o", "--outdir", default="vectors")
    parser.add_argument(
        "--vocabulary",
        help="reuse an existing vocabulary.json (test-set mode, prevents leakage)",
    )
    args = parser.parse_args(argv)

    records = load_records(Path(args.indir))
    if not records:
        print("no records found")
        return 1

    if args.vocabulary:
        vocabulary = json.loads(Path(args.vocabulary).read_text(encoding="utf-8"))
        mode = f"reused ({args.vocabulary})"
    else:
        vocabulary = fit(records)
        mode = "fitted on this set"

    matrix = transform(records, vocabulary)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(outdir / "matrix.npz", matrix)
    (outdir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "samples.json").write_text(
        json.dumps(sample_index(records), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Monitor Out-Of-Vocabulary (OOV) rate: high numbers indicate distribution shift
    seen = set()
    for r in records:
        seen.update(r.get("features", []))
    dropped = len(seen - set(vocabulary))
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100

    print(f"samples      : {matrix.shape[0]}")
    print(f"features |S| : {matrix.shape[1]}  ({mode})")
    print(f"non-zeros    : {matrix.nnz}  (density {density:.2f}%)")
    print(f"dropped OOV  : {dropped}")
    print(f"written to   : {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())