import os
from dotenv import load_dotenv
from google.cloud import bigquery

# To use it, create a .env file and put BIG_QUERY_PROJECT_ID="you project ID" in it.
# From Bob: Made the change to use environment variable for project ID instead of hardcoding it in the code. 
# This way, we can keep our project ID private and easily switch between different projects.
load_dotenv()

if os.environ.get("BIG_QUERY_PROJECT_ID"):
    client = bigquery.Client(project=os.environ["BIG_QUERY_PROJECT_ID"])
else:
    raise ValueError("BIG_QUERY_PROJECT_ID environment variable not set. Please set it in your .env file.")

def icu_query():
    query = """
        SELECT
            icu.subject_id,
            icu.hadm_id,
            icu.stay_id,
            icu.first_careunit,
            icu.last_careunit,
            icu.intime,
            icu.outtime,
            icu.los,
            adm.admittime,
            adm.dischtime,
            adm.deathtime,
            adm.admission_type,
            adm.admission_location,
            adm.language,
            adm.marital_status,
            adm.race,
            adm.edregtime, 
            adm.edouttime,
            adm.hospital_expire_flag,
            pat.gender,
            pat.anchor_age,
            pat.dod,
            mac.rr_interval,
            mac.p_onset,
            mac.p_end,
            mac.qrs_onset,
            mac.qrs_end,
            mac.t_end,
            mac.p_axis,
            mac.qrs_axis,
            mac.t_axis,
            mac.ecg_time,
            mac.report_0
        FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
        INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
            ON icu.hadm_id = adm.hadm_id
        INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` pat
            ON icu.subject_id = pat.subject_id
        INNER JOIN `physionet-data.mimiciv_ecg.machine_measurements` mac
            ON icu.subject_id = mac.subject_id
            AND mac.ecg_time BETWEEN icu.intime AND icu.outtime
        QUALIFY ROW_NUMBER() over (PARTITION BY icu.stay_id ORDER BY mac.ecg_time) = 1
    """
    return client.query(query).to_dataframe()

def bucket_ecg_report_0(report):
    if 'rapid ventricular' in report or 'uncontrolled ventricular' in report: # categorize these as 'afib_rvr' includes 'flutter..' rhythms as well
        return 'afib_rvr'
    elif 'pvcs' in report or 'pvc(s)' in report:
        return 'pvc'
    elif 'pacs' in report or 'pac(s)' in report:
        return 'pac'
    elif 'pacing' in report or 'paced' in report or 'pacer' in report or 'pacemaker' in report:
        return 'paced'
    elif 'st elevation mi' in report:
        return 'stemi_alert'
    elif 'atrial fibrillation' in report:
        return 'afib'
    elif 'sinus tachycardia' in report:
        return 'sinus_tachy'
    elif 'sinus bradycardia' in report:
        return 'sinus_brady'
    elif 'sinus rhythm' in report:
        return 'normal_sinus'
    else:
        return report
    
if __name__ == "__main__":
    # Test the functions
    icu_query()
