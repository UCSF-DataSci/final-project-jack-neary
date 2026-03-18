<!-- brew install google-cloud-sdk -->
# ECG-Based ICU Mortality Prediction
 
## Overview
 
This project investigates which ECG features are most predictive of in-hospital mortality among ICU patients using the MIMIC-IV clinical database. We build and compare multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting, and XGBoost) trained on ECG measurements, patient demographics, and vital sign extracted from BigQuery.
 
The core research question is:
 
> **Among ICU patients with ECG recordings, which ECG features are most predictive of in-hospital mortality?**
 
---

## Dataset

### Source
- **MIMIC-IV v3.1**: Medical Information Mart for Intensive Care
- **MIMIC-IV-ECG**: Diagnostic Electrocardiogram Matched Subset
- Hosted on Google BigQuery via PhysioNet
- Access requires credentialed registration at [physionet.org](https://physionet.org/content/mimic-iv-ecg/)

### Tables Used

| Table | Description |
|-------|-------------|
| `mimiciv_3_1_icu.icustays` | ICU stay records (admission/discharge times, care unit) |
| `mimiciv_3_1_hosp.admissions` | Hospital admissions (mortality flag, demographics) |
| `mimiciv_3_1_hosp.patients` | Patient demographics (age, gender) |
| `mimiciv_ecg.machine_measurements` | ECG machine-generated measurements |
| `mimiciv_3_1_icu.chartevents` | ICU charted vitals (heart rate, BP, SpO2, temperature, etc.) |
| `mimiciv_3_1_hosp.labevents` |  |
 
### Input Features
 
**ECG Features (continuous)**
 
| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `rr_interval` | Time between R peaks (ms) |  |
| `qrs_onset` | Start of QRS complex (ms) |  |
| `qrs_end` | End of QRS complex (ms) |  |
| `t_end` | End of T wave (ms) |  |
| `qrs_axis` | QRS electrical axis (degrees) |  |
| `t_axis` | T wave axis (degrees) |  |
 
**Demographic Features (numeric)**
 
| Feature | Description |
|---------|-------------|
| `age` |  |
| `gender` |  |
 
**Vital Sign Features (continuous, averaged over ICU stay)**
 
| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `heart_rate` |  |  |
| `sbp` | Systolic blood pressure (mmHg) |  |
| `dbp` | Diastolic blood pressure (mmHg) |  |
| `mbp` |  |  |
| `resp_rate` |  |  |
| `spo2` |  |  |
| `temperature` | Body temperature (°C) |  |
| `glucose` | Blood glucose (mg/dL) |  |

**Lab Features (continuous, closest value to ICU admission)**
 
| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `lactate` |  |  |
| `bun` |  |  |
| `creatinine` |  |  |
| `bicarbonate` |  |  |
  
**Categorical Features (one-hot encoded)**
 
| Feature | Categories |
|---------|-----------|
| `care_unit` | CVICU, MICU, CCU, SICU, NEURO, MICU/SICU |
| `admission_type` | Emergency, Elective, Surgical same-day, EW emergency |
| `admission_location` | Physician referral, Procedure site, Transfer, etc. |
| `language` | English, Spanish, Other, Unknown |
| `marital_status` | Married, Single, Divorced, Unknown |
| `race` | White, Black, Asian, Hispanic, Unknown, Other |
| `ecg_bucket` | Normal sinus, Atrial fibrillation, Other |
 
### Target Variable
- `hospital_expire_flag`: binary (0 = survived, 1 = died in hospital)
 
### Dataset Dimensions
 
| Split | Rows | Mortality Rate |
|-------|------|---------------|
| Full dataset | ~35,000 | ~12% |
| Training set (80%) | ~28,000 | ~12% |
| Test set (20%) | ~7,000 | ~12% |

---

## How to Run

### 1. Prerequisites
 
**Python version:** 3.13+
 
**Create and activate a virtual environment:**
```bash
# Create virtual environment
python -m venv venv
 
# Activate — macOS/Linux
source venv/bin/activate
 
# Activate — Windows
venv\Scripts\activate
```
 
**Install dependencies:**
```bash
pip install -r requirements.txt
```
 
> Make sure your virtual environment is activated before running `pip install`. You should see `(venv)` in your terminal prompt when it is active.
 
### 2. Google Cloud Setup
 
```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```
 
### 3. PhysioNet Access
 
1. Register at [physionet.org](https://physionet.org/register/)
2. Complete CITI training
3. Request access to [MIMIC-IV ECG](https://physionet.org/content/mimic-iv-ecg/)
4. Link your Google account at [physionet.org/settings/cloud](https://physionet.org/settings/cloud/)
 
### 4. Environment Setup
 
Create a `.env` file in the project root:
```
BIG_QUERY_PROJECT_ID=your-gcp-project-id
```

### 5. Run the Analysis
 
The analysis is split across three Jupyter notebooks. Run them in order:
 
**Step 1 — Data cleaning**
```bash
jupyter notebook cleaning.ipynb
```
Connects to BigQuery, extracts ICU stays, ECG measurements, vital signs, and demographics, cleans and preprocesses the data, and saves the output for modeling.
 
**Step 2 — Baseline modeling**
```bash
jupyter notebook modeling.ipynb
```
Trains and evaluates Logistic Regression, Random Forest, Gradient Boosting, and untuned XGBoost. Includes threshold tuning, ROC curves, confusion matrices, and SHAP analysis.
 
**Step 3 — XGBoost modeling**
```bash
jupyter notebook XGBoost.ipynb
```
Trains and tunes XGBoost with `scale_pos_weight` for class imbalance. Includes hyperparameter search, early stopping, and final model evaluation.
 
> `project.py` contains shared utility functions (e.g. `plot_roc_curve`, `plot_confusion_matrix`, `find_best_threshold`, `run_shap`) imported by the notebooks. Do not run it directly.

 
---
 
## Project Structure (Need fix)
 
```
ds223-final/
├── project.py               # Main script
├── .env                     # GCP project ID (not committed)
├── .gitignore               # Excludes .env and cached data
├── README.md                # This file
└── data/
    ├── cleaned_data.csv # cleaned data after running cleaning.ipynb
└── sanity_outputs/
    ├── model_evaluation.png # ROC curves + confusion matrices
    ├── shap_summary.png     # SHAP dot plot
    ├── shap_bar.png         # SHAP bar chart
    └── icu_ecg_data.parquet # Cached BigQuery results
```
 
---

## Decisions & Trade-offs

### Data Decisions
 
| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
|  |  |  |

### Modeling Decisions
 
| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| `class_weight='balanced'` for LR and RF | Handles ~88/12 class imbalance | May reduce precision |
| `scale_pos_weight` for XGBoost | Built-in imbalance handling | Requires tuning |
| `sample_weight` for Gradient Boosting | No native class_weight parameter | Manual step needed |
| Threshold tuning (not default 0.5) | Default 0.5 caused RF/GB to predict almost no deaths | Lower threshold increases false positives |
| `StratifiedKFold(5)` | Preserves class ratio in each fold | More compute than KFold |
| SHAP on 500-sample subset | Full test set SHAP takes 30+ mins | Slightly less precise SHAP estimates |
 
### Features Cut for Time
 
- **Comorbidity scores** (Charlson index, SOFA score) — standard ICU risk scores, widely used in clinical mortality research
- **Longitudinal ECG features** — using multiple ECGs per stay instead of just the first would capture deterioration over time
- **Medication data** (vasopressors, sedatives) — proxy for severity of illness
 
---

## Example Output
 
### Model Performance
 
| Model | Test AUC | Threshold | Deaths Caught (TP) | Deaths Missed (FN) | Recall |
|-------|----------|-----------|-------------------|-------------------|--------|
| Logistic Regression | 0.834 | 0.55 | 580/839 | 259 | 69.1% |
| Random Forest | 0.840 | 0.20 | 484/839 | 355 | 57.7% |
| Gradient Boosting | 0.849 | 0.55 | 566/839 | 273 | 67.5% |
| XGBoost | 0.816 | 0.50 | 398/839 | 441 | 47.4% |
| XGBoost (Tuned) | **0.852** | 0.55 | **572/839** | 267 | **68.2%** |
 
> All models use tuned classification thresholds instead of the default 0.5. Default threshold caused Random Forest and Gradient Boosting to predict almost no deaths due to class imbalance... (More explaination maybe)

### Top Predictive Features (SHAP — [final model]) Need Fix after deciding which final model to use

**Lab & Metabolic Features (strongest predictors):**
1. `bun` — top overall feature; high BUN (blood urea nitrogen) strongly pushes toward death, indicating kidney dysfunction
2. 
 
**Vital Sign Features:**
1. `resp_rate` — high respiratory rate strongly associated with death; marker of respiratory distress
2. 
 
**ECG Features:**
1. `rr_interval` — still a meaningful ECG predictor; high RR (bradycardia) pushes toward death
2. `
 
**Non-ECG Clinical Features:**
1. `care_unit_cvicu` — being in CVICU strongly protective (specialized cardiac care)
2. 
 
> Short summary of findings.

## Citations
 
### Data
- Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58. 
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. 
- Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C., Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E., Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/4nqg-sb35. 
- https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/measurement (adapted vitals_query from here)


