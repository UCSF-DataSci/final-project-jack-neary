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


bicarb on chemistry 50882
creatinine on chemistry 50912
lactate on blood gas 50813 (and 52442)