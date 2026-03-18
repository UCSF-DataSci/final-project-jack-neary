<!-- brew install google-cloud-sdk -->
# ECG-Based ICU Mortality Prediction
 
## Overview
 
This project investigates which ECG features are most predictive of in-hospital mortality among ICU patients using the MIMIC-IV clinical database. We build and compare multiple machine learning models — Logistic Regression, Random Forest, Gradient Boosting, and XGBoost — trained on ECG measurements and patient demographics extracted from BigQuery.
 
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

## How to Run


## Decisions & Trade-offs


## Example Output


## Citations
 
### Data
- Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58. 
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. 
- Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C., Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E., Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/4nqg-sb35. 
- https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/measurement (adapted vitals_query from here)


