"""Benchmark: cost of the current extraction pipeline vs. full AnalyzeAPK.

AnalyzeAPK is required to access androguard's bundled API-to-permission
mapping (needed for S5/S6). This script measures what that costs.
"""
import time
from pathlib import Path

from loguru import logger
logger.remove()

from androguard.misc import AnalyzeAPK
from drebin.extract import extract_apk

print(f"{'APK':32}{'size MB':>10}{'current':>10}{'AnalyzeAPK':>14}{'factor':>9}")
print("-" * 75)

total_current = total_analyze = 0.0

for path in sorted(Path("apks").glob("*.apk")):
    size_mb = path.stat().st_size / 1_000_000

    start = time.perf_counter()
    extract_apk(path)
    t_current = time.perf_counter() - start

    start = time.perf_counter()
    AnalyzeAPK(str(path))
    t_analyze = time.perf_counter() - start

    total_current += t_current
    total_analyze += t_analyze
    print(f"{path.name[:31]:32}{size_mb:>10.1f}{t_current:>10.2f}{t_analyze:>14.2f}{t_analyze / t_current:>8.1f}x")

print("-" * 75)
print(f"{'TOTAL':32}{'':>10}{total_current:>10.2f}{total_analyze:>14.2f}{total_analyze / total_current:>8.1f}x")