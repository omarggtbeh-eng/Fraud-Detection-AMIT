# Fraud Detection | AMIT Graduation Project

An end-to-end machine learning project for identifying suspicious financial transactions, with exploratory data analysis, feature engineering, model comparison and an interactive Streamlit demo.

![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B)

## Project highlights

- Analysis of over 6.3 million simulated transactions.
- Class imbalance analysis and chronological train/validation/test splitting.
- Comparison of logistic regression and a class-weighted decision tree.
- Feature engineering, preprocessing pipelines and validation-based decision threshold selection.
- Interactive transaction scoring and decision-tree explanations using Streamlit.

## Dataset

The project uses the [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1), containing 6,362,620 simulated financial transactions and 8,213 fraud labels (0.129%). Training uses steps before 323, validation uses steps 323–377, and testing uses steps 378 onward.

Download the dataset separately and place `PS_20174392719_1491204439457_log.csv` in the repository root to reproduce training. The raw dataset is intentionally not tracked.

## Machine learning pipeline

Data exploration → feature engineering → chronological split → preprocessing → model training → validation threshold selection → held-out evaluation → model export → Streamlit demo.

The final demo uses a class-weighted decision tree (maximum depth 10) and a validation-selected cutoff of 0.50. Input features include transaction amount, opening balances, transaction type, simulation time and derived balance features.

## Reported held-out test results

The test set contains 955,744 transactions, including 4,010 frauds. Metrics are for the fraud class and are reported from the project evaluation.

| Model | Threshold | Precision | Recall | F1 | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.05 | 99.95% | 99.23% | 99.59% | 0.9966 |
| Weighted decision tree | 0.50 | 99.98% | 99.93% | 99.95% | 0.9990 |

**Important limitation:** These results are from simulated data, not real-world fraud detection. PaySim cancels detected fraud, and its publisher cautions against using balance fields; opening-balance features may capture simulator-specific artifacts. Independent validation on real-world data is necessary before operational use.

## Getting started

Use Python 3.11. Clone the repository and install the supplied dependencies:

```powershell
git clone https://github.com/omarggtbeh-eng/Fraud-Detection-AMIT.git
cd Fraud-Detection-AMIT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `python3.11 -m venv .venv` and `.venv/bin/python` instead.

### Train the model

Download the PaySim CSV as described above. Open the final training notebook in Jupyter or VS Code, select the virtual environment kernel and run the cells in order. Its export step generates `fraud_detector.joblib`. Alternatively, place the original trusted model artifact in the repository root beside `app.py`.

### Launch the Streamlit demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux: `.venv/bin/python -m streamlit run app.py`.

The demo accepts transaction details and displays a fraud score, classification and up to four decision-path factors. Only load trusted Joblib model artifacts.

## Technology stack

Python · Pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Joblib · Jupyter · Streamlit

## Acknowledgments

Developed as an AMIT graduation project. Dataset: PaySim. This repository is presented as an educational portfolio project, not a production fraud prevention system.
