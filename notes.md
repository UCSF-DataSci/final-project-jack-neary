Hospital
 - patients table --> date of death (dod)
 - prescriptions table --> 
 - admissions table --> admission type
 - omr table --> result_name, results value (bp, weight)

DB Schema
https://mimic.mit.edu/docs/iv/modules 

## potential research question
 - Among ICU patients with ECG recordings, which ECG features are most predictive of in-hospital mortality?

 - outcome:
    - mortality (hospital expire flag)
 - predictors:
    - reports from ecg table
    - demographic variables (table?)
    - ecg features from ecg table

### report_0 --> ecg_buckets
   - categories
      - sinus_tachy
      - normal_sinus
      - afib
      - sinus_brady
      - stemi_alert
      - pvc
      - pac
      - paced
      - accelerated_junctional
      - other
      - idioventricular
      - supraventricular
      - afib_rvr
      - atrial_ectopic

### predictors
   - ecg_bucket
   - care_unit
   - gender
   - age
   - race
   - marital_status
   - language
   - admission_type
   - admission_location
   - rr_interval
   - qrs_onset
   - qrs_end
   - t_end
   - qrs_axis
   - t_axis
   
### outcome = hospital_expire_flag


### labs measurement units
bicarb mmEq/L
creatinine mg/dL
lactate mmol/L
bun mg/dL

### possible conclusion
Our model achieves AUC comparable to published APACHE-II performance (0.83–0.88) while being fully automated from structured EHR data. Rather than replacing APACHE-II, it demonstrates that ECG-derived features combined with standard vitals and labs can produce similar discriminative power without manual severity scoring — suggesting potential utility as a complementary automated screening tool.

whereas the APACHE-ii score requires a trained clinician to manually pull data and score variables, this measurement is automated from EHR data. Uses EHR signal which the apache-ii score does not, filling that gap. is more scalable, could potentially run on every admitted patient continuously. **Comparable AUC with zero manual input and additional ECG signal that apache-ii ignores.**

## things that need changed in the readme
- Line 194 — "Project Structure (Might Need Updates)"
- Line 201 — XGBoost.ipynb says "(Merged into modeling.ipynb)" — is that true or are they still separate?
- Line 209 — icu_data.csv typo + note
- Lines 216-219 — [description] placeholders for CSVs
- Line 269 — XGBoost (Tuned) still shows 0.55/572/68.2%