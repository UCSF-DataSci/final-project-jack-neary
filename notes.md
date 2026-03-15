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

### potential predictors
   - ecg_bucket
   - first_careunit
   - last_careunit
   - gender
   - age
   - race
   - marital_status
   - language
   - admission_type
   - rr_interval
   - p_onset
   - p_end
   - qrs_onset
   - qrs_end
   - t_end
   - p_axis
   - qrs_axis
   - t_axis

## to do
   - set boundaries and impute ecg measurements
   - calculate age from anchor_age
   - 