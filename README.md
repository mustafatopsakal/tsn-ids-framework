# TSN-IDS Framework

A unified Python framework for detecting Low-Rate Denial-of-Service (LDoS) attacks in IEEE 802.1 Time-Sensitive Networking (TSN) based in-vehicle communication networks.

This project consolidates three related research pipelines into a single, configurable tool:

1. **Trace parsing** — Convert raw OMNeT++ simulation traces into labeled CSV datasets.
2. **ML classification** — Train and evaluate six machine learning models on the generated datasets.
3. **Imbalanced data handling** — Optionally apply tSMOTE or tsaug augmentation to address class imbalance.

## Project Structure

```
tsn-ids-framework/
├── src/
│   ├── main.py              # Unified CLI entry point
│   ├── trace_parser.py      # Raw OMNeT++ trace → labeled CSV
│   ├── data_handling.py     # CSV loading, feature extraction, writing
│   ├── classification.py    # 6 ML models + cross-validation
│   ├── augmentation.py      # tSMOTE and tsaug augmentation strategies
│   ├── tsmote.py            # Third-party tSMOTE library (Hadlock-Lab)
│   └── utils.py             # MAC/node name → numeric conversion helpers
│
├── data/
│   ├── raw_traces/          # Raw OMNeT++ simulation outputs (.txt)
│   │   ├── normal/
│   │   └── attack/
│   ├── raw_csv/             # Labeled CSV files (7 features + label)
│   │   ├── normal/
│   │   └── attack/
│   └── processed/           # ML-ready datasets (11 features + label)
│
├── output/                  # Classification results
│   ├── plain_results.txt
│   ├── tsmote_results.txt
│   └── tsaug_results.txt
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.7+.

```bash
pip install -r requirements.txt
```

## Usage

All commands are run from the `src/` directory:

```bash
cd src
```

### Classify with existing CSV data (no augmentation)

```bash
python main.py --mode classify --augmentation none
```

### Classify with tSMOTE augmentation

```bash
python main.py --mode classify --augmentation tsmote
```

### Classify with tsaug augmentation

```bash
python main.py --mode classify --augmentation tsaug
```

### Generate CSV datasets from raw OMNeT++ traces

```bash
python main.py --mode generate
```

### Full pipeline (generate + classify)

```bash
python main.py --mode full --augmentation tsmote
```

## ML Models

| Model | Library |
|-------|---------|
| K-Nearest Neighbors (k=5) | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest (100 trees) | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| SVM (RBF kernel) | scikit-learn |

All models are evaluated using **5-fold cross-validation** with **StandardScaler** preprocessing. Metrics: accuracy, precision, recall, F1-score, training time, and test time.

## Dataset Format

### raw_csv (7 features + label)

| Column | Description |
|--------|-------------|
| f1 | timestamp |
| f2 | stream_id |
| f3 | source_id |
| f4 | destination_id |
| f5 | source_mac |
| f6 | destination_mac |
| f7 | packet_length |
| label | 0: benign, 1: malicious |

### processed (11 features + label)

| Column | Description |
|--------|-------------|
| f1 | timestamp |
| f2 | last_remote_timestamp (inter-arrival time) |
| f3 | stream_id |
| f4 | prev_stream_id |
| f5 | prev_prev_stream_id |
| f6 | prev_prev_prev_stream_id |
| f7 | source_id |
| f8 | destination_id |
| f9 | packet_source_mac |
| f10 | packet_dest_mac |
| f11 | packet_length |
| label | 0: benign, 1: malicious |

### Node ID Mapping

| ID | Node |
|----|------|
| 1 | Cam1 |
| 2 | Cam2 |
| 3 | Cam3 |
| 4 | DA-Cam |
| 5 | HU |
| 6 | RSE |
| 7 | Telematics |
| 8 | CU |
| 9 | CD-Audio DVD |
| 10 | Cam4 |
| 11 | Switch1 |
| 12 | Switch2 |

## Citation

If you use this repository or the accompanying dataset in academic work, please cite:

```bibtex
@article{topsakal2025machine,
  author    = {Topsakal, Mustafa and Cevher, Sel{\c{c}}uk and Ergen{\c{c}}, Do{\u{g}}analp},
  title     = {{A Machine Learning-based Intrusion Detection Framework with Labeled Dataset Generation for IEEE 802.1 Time-Sensitive Networking}},
  journal   = {Journal of Systems Architecture},
  volume    = {164},
  pages     = {103408},
  year      = {2025},
  month     = jul,
  publisher = {Elsevier},
  doi       = {10.1016/j.sysarc.2025.103408}
}
```

A ready-to-use citation file is provided in `CITATION.cff`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
