"""Summarise all extraction records in out/ as a single table."""

import glob
import json

# =============================================================================
# Pipeline Architecture:
# 1. Pipeline Config: Map feature categories (S1-S8) to output display labels.
# 2. Data Ingestion: Parse output JSON files and sort deterministically.
# 3. Table Formatting: Render formatted ASCII table with fallback error logs.
# 4. Metrics & Aggregate Stats: Calculate coverage ratios and feature space |S|.
# =============================================================================

# Step 1: Define feature set mappings in paper order (S1 to S8)
SETS = [
    ("feature", "S1"),
    ("permission", "S2"),
    ("component", "S3"),
    ("intent", "S4"),
    ("api_call", "S5"),
    ("real_permission", "S6"),
    ("call", "S7"),
    ("url", "S8"),
]

# Step 2: Load raw extraction logs from directory
records = [json.load(open(f, encoding="utf-8")) for f in glob.glob("out/*.json")]

# Ensure consistent row ordering across runs
records.sort(key=lambda r: r.get("package", ""))

# Step 3: Build table layout and print header
header = f"{'package':32}" + "".join(f"{lbl:>7}" for _, lbl in SETS)
header += f"{'total':>8}{'partial':>9}"
print(header)
print("-" * len(header))

# Stream records to stdout
for r in records:
    # Bound package string to preserve 32-char column offset
    row = f"{r.get('package', '?')[:31]:32}"
    
    # Extract set counts for S1-S8
    row += "".join(f"{len(r['sets'].get(key, [])):>7}" for key, _ in SETS)
    
    # Append global counters and status flag
    row += f"{r.get('n_features', 0):>8}"
    row += f"{'yes' if r.get('partial') else 'no':>9}"
    print(row)

    # Attach stage-level tracebacks for failed/partial extractions
    if r["errors"]:
        print(f"    stages: {r.get('stages')}")
        for e in r["errors"]:
            print(f"    error: {e}")

print()

# Step 4: Compute dataset-level extraction metadata
print(f"APKs processed : {len(records)}")
print(f"Partial records: {sum(1 for r in records if r.get('partial'))}")
print()

# Measure non-empty set density per feature type
print("Coverage (how many APKs have a non-empty set):")
for key, lbl in SETS:
    filled = sum(1 for r in records if r["sets"].get(key))
    print(f"  {lbl} {key:16} {filled}/{len(records)}")

# Deduplicate total feature vocabulary across all records to obtain |S|
vocabulary = set()
for r in records:
    vocabulary.update(r["features"])

print()
print(f"Combined feature space |S| = {len(vocabulary)}")