# ECG-Based ICU Mortality Prediction
**Authors:** Bob Ho, Jack Neary
## Introduction
ICU mortality is one of the most time-sensitive predictions in medicine. Clinicians currently rely on manual scoring systems like APACHE-II, which require bedside calculation of 12 physiological variables. These scores are labor-intensive, subject to human error, and notably exclude cardiac rhythm data entirely — despite arrhythmias being common and clinically significant in the ICU.

This project asks whether a fully automated model built from structured EHR data — including ECG findings, vital signs, and lab values — can match APACHE-II's predictive performance without manual scoring.
## Overview
This project investigates which ECG features, vital signs, and biomarkers are most predictive of in-hospital mortality among ICU patients using the MIMIC-IV clinical database. We build and compare multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting, and XGBoost) trained on ECG measurements, patient demographics, and vital sign extracted from BigQuery.
 
The core research question is:
 
> **Can routinely collected EHR data — including ECG rhythm, vital signs, and lab values — predict ICU mortality in the MIMIC-IV ICU database?**
 
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
| `ecg_bucket` | `normal_sinus`, `afib`, `afib_rvr`, `sinus_tachy`, `sinus_brady`, `paced`, `pvc`, `pac`, `atrial_ectopic`, `supraventricular`, `accelerated_junctional`, `idioventricular`, `other` |
 
### Target Variable
- `hospital_expire_flag`: binary (0 = survived, 1 = died in icu)

### Dataset Dimensions
 
| Split | Rows | Mortality Rate |
|-------|------|---------------|
| Full dataset | ~35,000 | ~12% |
| Training set (80%) | ~28,000 | ~12% |
| Test set (20%) | ~7,000 | ~12% |

---
## Data Cleaning and Preprocessing Decisions

### ECG Report Bucketing
Each ICU stay had a free-text ECG report field (`report_0`). Rather than one-hot encoding thousands of unique report strings, we wrote a keyword-based bucketing function that maps each report to one of 13 clinical rhythm categories. Categories were ordered by clinical priority — for example, `afib_rvr` (atrial fibrillation with rapid ventricular response) is checked before `afib` so that more specific diagnoses take precedence. Reports that didn't match any category were labeled `other` (~1.3% of records). This preserves clinically meaningful rhythm distinctions while keeping the feature space manageable.

### ECG Measurement Cleaning
Three columns — `p_onset`, `p_end`, and `p_axis` — were dropped entirely because over 50% of values were the sentinel value 29999, indicating the machine could not compute P wave timing. Retaining these would have required imputing the majority of a column with no real signal. For remaining ECG measurements, values above 4000ms or axes outside ±180° were nulled as physiologically impossible before imputation. Missing values were then imputed using the median of each patient's ECG bucket category, since patients in the same rhythm group have similar underlying electrophysiology.

### Vitals and Lab Averaging
Rather than taking a single admission snapshot, all vital signs and lab values were averaged over the first 24 hours of ICU admission. This reduced missingness from ~40% to ~6% and better represents a patient's early ICU course than any single measurement. Remaining missing values were imputed using the ECG bucket median, with a global median fallback for buckets with no available values.

### Categorical Feature Simplification
Race was collapsed from 30+ inconsistently formatted MIMIC-IV strings into six broad categories (White, Black, Asian, Hispanic, Unknown, Other). Care unit was simplified from raw unit names to six clinical groupings (CVICU, MICU, CCU, SICU, NEURO, MICU/SICU). First and last care unit were identical for all patients in the dataset, so they were collapsed into a single column.

### Leakage Prevention
`in_time`, `out_time`, and `los` (length of stay) were dropped from the feature set. These variables are only known after discharge and would constitute data leakage — the model cannot use information that isn't available at the time of prediction.

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
 
## Project Structure
 
```
ds223-final/
├── project.py               # Shared utility functions (imported by notebooks)
├── cleaning.ipynb           # Step 1 — data extraction and preprocessing
├── modeling.ipynb           # Step 2 — LR, RF, Gradient Boosting, Basic XGBoost, & tuned XGBoost results
├── XGBoost.ipynb            # Step 3 — XGBoost training and tuning (Merged into modeling.ipynb)
├── requirements.txt         # Python dependencies
├── notes.md                 # Development notes
├── .env                     # GCP project ID (not committed)
├── .gitignore               # Excludes .env, venv, and cached data
├── README.md                # This file
├── data/
│   └── cleaned_data.csv     # Cleaned dataset output from cleaning.ipynb
└── sanity_outputs/
    ├── feature_importance.png              # Top 15 feature importances (XGBoost)
    ├── model_evaluation.png               # ROC curves + confusion matrices (all models)
    ├── xgboost_tuned_evaluation.png       # Tuned XGBoost ROC + confusion matrix
    ├── shap_summary.png                   # SHAP dot plot (XGBoost)
    ├── shap_bar.png                       # SHAP bar chart (XGBoost)
    ├── initial_drop.csv                   # ECG report_0 value counts after lowercasing and dropping warning rows, before bucketing
    ├── original_ecg_report0_frequency.csv # ECG report_0 value counts before any cleaning (raw strings)
    ├── post_bucket_function.csv           # ECG bucket counts after bucketing function applied
    └── report0_cleaned.csv               # ECG bucket counts after unbucketed reports relabeled as 'other'
```
 
---

## Decisions & Trade-offs

### Data Decisions
 
| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Drop `p_onset`, `p_end`, `p_axis` | >50% of values were sentinel value 29999 — not physiologically meaningful | Lose P wave features which can indicate atrial pathology |
| Null ECG measurements >4000ms; axes outside ±180° | Values outside these ranges are physiologically impossible | Small amount of real extreme values may be lost |
| Impute missing ECG measurements with ECG bucket median | Patients in the same rhythm category have similar underlying physiology | Reduces variance within buckets; may mask true missingness patterns |
| Average vitals and labs over first 24h (not admission snapshot) | Reduces missingness from ~40% to ~6%; more representative of early ICU course | Loses time-series dynamics within the 24h window |
| Impute missing vitals/labs with ECG bucket median, then global median | Maintains physiological grouping structure | Introduces bias if buckets are not clinically homogeneous |
| Simplify race into broad categories | MIMIC-IV has 30+ race strings with inconsistent formatting | Loses granularity; broad categories may obscure health disparity signals |
| Collapse first/last care unit into single column | No patient in the dataset had a different first vs. last care unit | Assumes care unit is static across ICU stay |
| Drop `in_time`, `out_time`, `los` | Would cause data leakage — LOS and discharge time are only known retrospectively | Lose potentially useful severity signal |
| ECG bucket "other" category | ~1.3% of reports don't match any named rhythm pattern | Small heterogeneous group; model may not learn meaningful signal from it |

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
 
| Model | ROC AUC | PR AUC | Threshold | Deaths Caught (TP) | Deaths Missed (FN) | Recall |
|-------|---------|--------|-----------|-------------------|-------------------|--------|
| Logistic Regression | 0.834 | 0.458 | 0.55 | **580/839** | 259 | **69.1%** |
| Random Forest | 0.840 | 0.461 | 0.20 | 484/839 | 355 | 57.7% |
| Gradient Boosting | 0.849 | 0.492 | 0.55 | 566/839 | 273 | 67.5% |
| XGBoost | 0.816 | 0.461 | 0.50 | 398/839 | 441 | 47.4% |
| XGBoost (Tuned) | **0.852** | **0.501** | 0.55 | 572/839 | 267 | 68.2% |
 
> All models use tuned classification thresholds instead of the default 0.5. Default threshold caused Random Forest and Gradient Boosting to predict almost no deaths due to class imbalance.

> Five models were trained and evaluated on a test set of 6,966 patients (839 deaths, 6,127 survivors). All models significantly outperform the random baseline PR AUC of ~0.12 (equal to the mortality rate), with ROC AUC scores ranging from 0.816 to 0.852 and PR AUC scores ranging from 0.458 to 0.501. XGBoost (Tuned) achieved the highest scores on both metrics (ROC AUC = 0.852, PR AUC = 0.501), confirming it as the best overall model. PR AUC is reported alongside ROC AUC because the dataset is heavily imbalanced (~88% survivors, ~12% deaths) since ROC AUC can appear inflated in such settings, while PR AUC focuses specifically on the model's ability to detect the minority class (deaths) and is therefore the more informative metric for this task.


### XGBoost (Tuned) — Threshold Analysis
 
| Threshold | Deaths Caught (TP) | Deaths Missed (FN) | Recall |
|-----------|-------------------|-------------------|--------|
| 0.55 | 572/839 | 267 | 68.2% |
| 0.4 | 701/839 | 138 | 83.6% |

> The classification threshold controls the tradeoff between recall (deaths correctly identified) and precision (false alarm rate). At the default F1-optimised threshold of 0.55, XGBoost (Tuned) correctly identified 572 out of 839 deaths (recall = 68.2%). Lowering the threshold to 0.40 increased recall to 83.6%, correctly identifying 701 deaths and reducing missed deaths from 267 to 138. This is a clinically meaningful improvement of 129 additional deaths caught. This came at the cost of increased false positives, with more survivors incorrectly flagged as high risk. The threshold of 0.40 was selected as the final operating point, prioritising sensitivity given the high cost of missing a true death in an ICU setting.

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

---
## Limitations

- **Single-center retrospective data** — all patients are from Beth Israel Deaconess Medical Center (MIMIC-IV). Generalizability to other institutions is unknown.
- **Low precision at clinical threshold** — at threshold=0.40, precision is 0.27, meaning roughly 3 in 4 flagged patients do not die. Sustained deployment could cause alert fatigue.
- **First ECG only** — only the ECG nearest to ICU admission is used. Rhythm changes over the ICU stay are not captured.
- **Missing severity variables** — SOFA score, GCS, and vasopressor use are not included. These are standard ICU mortality predictors.
- **Imputation may mask clinical signal** — missing lab values were imputed with bucket medians. In practice, a missing lactate often means the clinician didn't suspect sepsis — missingness itself carries information.
- **No prospective validation** — model performance on future patients or at a different institution has not been tested.
---
## Conclusion

Across all four model types, XGBoost with tuned hyperparameters and class imbalance weighting achieved the best performance (ROC AUC = 0.852, PR AUC = 0.501). At a classification threshold of 0.40, the model correctly identified 701 of 839 deaths in the test set (recall = 83.6%). SHAP analysis identified lactate, BUN, bicarbonate, and respiratory rate as the strongest mortality predictors, consistent with established clinical understanding of sepsis and organ failure. ECG features — particularly RR interval — contributed independently beyond vitals and labs.

APACHE-II, the clinical standard for ICU mortality prediction, typically achieves ROC AUC in the range of 0.83–0.88 depending on the cohort. Our model performs within this range while being fully automated from structured EHR data — no manual bedside scoring required. Importantly, APACHE-II does not incorporate cardiac rhythm data. The independent contribution of ECG features in our model suggests that rhythm information adds predictive signal that standard severity scores miss entirely.

This model is not intended to replace APACHE-II. It demonstrates that an automated pipeline combining ECG findings, vital signs, and lab values from EHR data can match the discriminative performance of manual severity scoring — and may serve as a useful complementary screening tool, particularly in settings where APACHE-II scoring is delayed or inconsistently applied. It fills gaps that existing severity scores leave open: APACHE-II ignores cardiac rhythm entirely, SOFA score does not incorporate ECG data, and neither accounts for the context of arrhythmias that ECG bucketing captures. By integrating rhythm classification directly into a mortality prediction pipeline, this approach surfaces a clinically relevant signal that standard ICU scoring systems were never designed to use.

---

## Citations
 
### Data
- Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58. 
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. 
- Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C., Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E., Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/4nqg-sb35. 
- Johnson, A., et al. (2024). mimic-code: MIMIC-IV concepts/measurement [Software]. GitHub. https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/measurement
