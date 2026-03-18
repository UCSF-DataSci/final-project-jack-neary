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
| `mimiciv_3_1_hosp.labevents` | Laboratory results (lactate, creatinine, BUN, bicarbonate)|
 
### Input Features
 
**ECG Features (continuous)**
 
| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `rr_interval` | Time between R peaks (ms) |  Reflects heart rate and rhythm regularity; prolonged or irregular RR indicates arrhythmia|
| `qrs_onset` | Start of QRS complex (ms) | Marks onset of ventricular depolarization; used to calculate QRS duration |
| `qrs_end` | End of QRS complex (ms) | Marks end of ventricular depolarization; wide QRS indicates conduction delay or bundle branch block |
| `t_end` | End of T wave (ms) | Marks end of ventricular repolarization; prolonged QT (QRS onset to T end) associated with arrhythmia risk |
| `qrs_axis` | QRS electrical axis (degrees) | Direction of ventricular depolarization; axis deviation indicates ventricular hypertrophy or conduction abnormality |
| `t_axis` | T wave axis (degrees) | Direction of repolarization; T axis deviation associated with ischemia and electrolyte abnormalities |
 
**Demographic Features (numeric)**

| Feature | Description |
|---------|-------------|
| `age` | Patient age at time of ICU admission |
| `gender` | Patient sex encoded as 0 (male) or 1 (female) |

**Vital Sign Features (continuous, averaged over first 24h of ICU admission)**

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `heart_rate` | Heart rate (bpm) | Tachycardia and bradycardia both indicate physiological stress |
| `sbp` | Systolic blood pressure (mmHg) | Low SBP indicates hemodynamic compromise and shock |
| `dbp` | Diastolic blood pressure (mmHg) | Reflects vascular resistance; low DBP seen in sepsis |
| `mbp` | Mean arterial pressure (mmHg) | Best single measure of perfusion pressure; target in ICU resuscitation |
| `resp_rate` | Respiratory rate (breaths/min) | Tachypnea is an early marker of respiratory failure and sepsis |
| `spo2` | Peripheral oxygen saturation (%) | Low SpO2 indicates hypoxemia and respiratory failure |
| `temperature` | Body temperature (°C) | Fever and hypothermia both associated with infection and poor outcomes |
| `glucose` | Blood glucose (mg/dL) | Hyperglycemia common in critical illness; both extremes associated with mortality |

**Lab Features (continuous, averaged over first 24h of ICU admission)**

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `lactate` | Serum lactate (mmol/L) | Best single predictor of sepsis mortality; elevated lactate indicates tissue hypoperfusion |
| `bun` | Blood urea nitrogen (mg/dL) | Marker of kidney function and protein catabolism; elevated in AKI and critical illness |
| `creatinine` | Serum creatinine (mg/dL) | Primary marker of kidney function; elevated in acute kidney injury |
| `bicarbonate` | Serum bicarbonate (mEq/L) | Marker of metabolic acid-base status; low bicarb indicates metabolic acidosis |
  
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
 
> `project.py` contains shared utility functions (e.g. `plot_roc_curve`, `plot_confusion_matrix`) imported by the notebooks. Do not run it directly.

 
---
 
## Project Structure (Might Need Updates)
 
```
ds223-final/
├── project.py               # Shared utility functions (imported by notebooks)
├── cleaning.ipynb           # Step 1 — data extraction and preprocessing
├── modeling.ipynb           # Step 2 — LR, RF, Gradient Boosting, Basic XGBoost
├── XGBoost.ipynb            # Step 3 — XGBoost training and tuning (Merged into modeling.ipynb)
├── requirements.txt         # Python dependencies
├── notes.md                 # Development notes
├── .env                     # GCP project ID (not committed)
├── .gitignore               # Excludes .env, venv, and cached data
├── README.md                # This file
├── data/
│   └── cleaned_data.csv     # Cleaned dataset output from cleaning.ipynb
├── icu_data.csv             # Raw ICU data extracted from BigQuery (might need to remvoe this)
└── sanity_outputs/
    ├── feature_importance.png              # Top 15 feature importances (model name)
    ├── model_evaluation.png               # ROC curves + confusion matrices (all models)
    ├── xgboost_tuned_evaluation.png       # Tuned XGBoost ROC + confusion matrix
    ├── shap_summary.png                   # SHAP dot plot ()
    ├── shap_bar.png                       # SHAP bar chart ()
    ├── initial_drop.csv                   # [description]
    ├── original_ecg_report0_frequency.csv # [description]
    ├── post_bucket_function.csv           # [description]
    └── report0_cleaned.csv               # [description]
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
| Logistic Regression | 0.834 | 0.55 | **580/839** | 259 | **69.1%** |
| Random Forest | 0.840 | 0.20 | 484/839 | 355 | 57.7% |
| Gradient Boosting | 0.849 | 0.55 | 566/839 | 273 | 67.5% |
| XGBoost | 0.816 | 0.50 | 398/839 | 441 | 47.4% |
| XGBoost (Tuned) | **0.852** | 0.55 | 572/839 | 267 | 68.2% |
 
> All models use tuned classification thresholds instead of the default 0.5. Default threshold caused Random Forest and Gradient Boosting to predict almost no deaths due to class imbalance... (More explaination maybe)

### Top Predictive Features (SHAP — XGBoost)

**Clinical & Care Context Features:**
1. `care_unit_cvicu` — strongest single predictor; being in CVICU strongly predicts **survival** (large negative SHAP), not being in CVICU pushes toward **death**
2. `age` — older age pushes toward **death**; younger age toward **survival**
3. `admission_type_surgical_same_day` — planned same-day surgical admission strongly predicts **survival**; non-surgical emergency admissions push toward **death**
4. `care_unit_ccu` — CCU admission mildly protective, similar pattern to CVICU but smaller magnitude
5. `race_unknown` / `marital_status_unknown` — missing demographics pushes toward **death**; proxy for emergency/unplanned admissions
 
**Lab & Metabolic Features:**
1. `bun` — high BUN pushes toward **death** (kidney dysfunction); low BUN toward **survival**
2. `lactate` — high lactate strongly pushes toward **death** (tissue hypoperfusion); long right tail indicates some patients with very high lactate have extremely elevated mortality probability
3. `bicarbonate` — low bicarbonate pushes toward **death** (metabolic acidosis); high bicarbonate protective — direction opposite to most features
4. `creatinine` — mixed signal; moderately elevated creatinine pushes toward **death**, very high creatinine patients may already be managed (dialysis)
 
**Vital Sign Features:**
1. `resp_rate` — high respiratory rate pushes toward **death**; marker of respiratory distress
2. `sbp` — low systolic BP pushes toward **death**; high SBP protective (hemodynamic stability)
3. `spo2` — mixed signal; low SpO2 pushes toward **death**, but very high SpO2 may reflect supplemental oxygen use masking underlying severity
4. `dbp` — low diastolic BP pushes toward **death**; consistent with SBP finding
5. `heart_rate` — low heart rate (bradycardia) pushes toward **death**; consistent with rr_interval finding
6. `temperature` — bidirectional effect; both fever and hypothermia associated with mortality
 
**ECG Features:**
1. `rr_interval` — high RR interval (bradycardia) pushes toward **death**; remains independently predictive after controlling for vitals and labs
2. `ecg_bucket_normal_sinus` — weak predictor; clustered near zero
3. `t_end` — weakest ECG predictor in top 20; minimal spread around zero
 
> Short summary of findings.

## Citations
 
### Data
- Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58. 
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. 
- Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C., Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E., Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/4nqg-sb35. 
- Johnson, A., et al. (2024). mimic-code: MIMIC-IV concepts/measurement [Software]. GitHub. https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/measurement


