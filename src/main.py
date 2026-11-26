# -*- coding: utf-8 -*-
"""
TSN-LDoS Intrusion Detection Framework — Unified Entry Point

Modes
-----
generate : Convert raw OMNeT++ traces (.txt) to labeled CSVs.
classify : Run ML classification on existing labeled CSVs.
full     : generate + classify in sequence.

Augmentation options (for classify / full modes)
------------------------------------------------
none   : No augmentation, standard KFold (default).
tsmote : tSMOTE oversampling + StratifiedKFold.
tsaug  : tsaug augmentation + StratifiedKFold.

Usage examples (from the repository root; any directory works)
--------------------------------------------------------------
    python src/main.py --mode classify --augmentation none
    python src/main.py --mode classify --augmentation tsmote
    python src/main.py --mode generate
    python src/main.py --mode full --augmentation tsaug
"""

import argparse
import os
import time

import numpy as np

import classification
import data_handling
import trace_parser
from utils import natural_key


# ── Paths (built from the repository root) ──────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TRACES_DIR = os.path.join(REPO_ROOT, "data", "raw_traces")
RAW_CSV_DIR = os.path.join(REPO_ROOT, "data", "raw_csv")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

RESULTS_FILES = {
    "none": "plain_results",
    "tsmote": "tsmote_results",
    "tsaug": "tsaug_results",
}

# Each run writes to its own file, so repeated runs never mix
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")


# ── Helper ──────────────────────────────────────────────────────────────────

def _results_path(augmentation):
    return os.path.join(
        OUTPUT_DIR, f"{RESULTS_FILES[augmentation]}_{RUN_STAMP}.txt"
    )


def _write_line(path, text):
    with open(path, "a+", encoding="utf-8") as f:
        f.write(text)


# ── Mode: generate ──────────────────────────────────────────────────────────

def run_generate():
    """Parse raw OMNeT++ traces and write labeled CSVs to raw_csv/."""
    print("=" * 60)
    print("MODE: generate — Converting raw traces to labeled CSVs")
    print("=" * 60)
    trace_parser.process_trace_directory(RAW_TRACES_DIR, RAW_CSV_DIR)
    print("\nTrace parsing completed.")


# ── Mode: classify ──────────────────────────────────────────────────────────

def run_classify(augmentation):
    """Run ML classification on labeled CSVs."""
    use_tsmote = augmentation == "tsmote"
    use_tsaug = augmentation == "tsaug"
    results_file = _results_path(augmentation)

    normal_dir = os.path.join(RAW_CSV_DIR, "normal")
    attack_dir = os.path.join(RAW_CSV_DIR, "attack")
    processed_dir = PROCESSED_DIR
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"MODE: classify — augmentation={augmentation}")
    print("=" * 60)

    # ── Process normal scenario ──────────────────────────────────────────
    ns_files = sorted(
        (f for f in os.listdir(normal_dir) if f.endswith(".csv")),
        key=natural_key,
    )
    if ns_files:
        ns_file = ns_files[0]
        ns_path = os.path.join(normal_dir, ns_file)
        print(f"\nProcessing normal scenario: {ns_file}")

        data_n, labels_n, _, _ = data_handling.load_csv(ns_path)
        data_n = data_handling.feature_extraction(data_n)
        labels_n = labels_n[data_handling.FE_START_INDEX:]
        zeros_n = labels_n.count(0)
        ones_n = labels_n.count(1)
        print(f"Zeros: {zeros_n}  Ones: {ones_n}  "
              f"Total: {zeros_n + ones_n}")

        data_handling.write_output(
            data_n, labels_n, os.path.join(processed_dir, ns_file)
        )

        _write_line(results_file,
                     f"{ns_file}\n"
                     f"Zeros: {zeros_n}  Ones: {ones_n}  "
                     f"Total: {zeros_n + ones_n}\n"
                     f"{'─' * 40}\n")

    # ── Process attack scenarios ─────────────────────────────────────────
    attack_files = sorted(
        (f for f in os.listdir(attack_dir) if f.endswith(".csv")),
        key=natural_key,
    )

    for attack_file in attack_files:
        attack_path = os.path.join(attack_dir, attack_file)
        print(f"\nProcessing attack scenario: {attack_file}")

        data_a, labels_a, _, _ = data_handling.load_csv(attack_path)
        data_a = data_handling.feature_extraction(data_a)
        labels_a = labels_a[data_handling.FE_START_INDEX:]
        zeros_a = labels_a.count(0)
        ones_a = labels_a.count(1)
        print(f"Zeros: {zeros_a}  Ones: {ones_a}  "
              f"Total: {zeros_a + ones_a}")

        data_handling.write_output(
            data_a, labels_a, os.path.join(processed_dir, attack_file)
        )

        _write_line(results_file,
                     f"\n{attack_file}\n"
                     f"Zeros: {zeros_a}  Ones: {ones_a}  "
                     f"Total: {zeros_a + ones_a}\n"
                     f"{'─' * 40}\n")

        all_data = np.array(data_a)
        all_labels = np.array(labels_a)

        for idx in range(len(classification.ML_METHODS)):
            model_name = classification.ML_METHODS[idx][0]
            print(f"\n  Training {model_name} on {attack_file}")

            result_text = classification.classify(
                all_data, all_labels, idx, use_tsmote, use_tsaug,
            )

            _write_line(results_file,
                         f"{attack_file}\n{result_text}"
                         f"{'─' * 40}\n")
            print(f"  Results written to {results_file}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TSN-LDoS Intrusion Detection Framework",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "classify", "full"],
        default="classify",
        help="Operation mode (default: classify)",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "tsmote", "tsaug"],
        default="none",
        help="Augmentation strategy for classification (default: none)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.perf_counter()

    if args.mode in ("generate", "full"):
        run_generate()

    if args.mode in ("classify", "full"):
        run_classify(args.augmentation)

    elapsed = time.perf_counter() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
