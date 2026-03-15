from google.cloud import bigquery

client = bigquery.Client(project='project-fc29ae8d-18cc-4c2c-adc')

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
icu_query()

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
    elif 'ectopic atrial' in report or 'probable atrial tachycardia' in report:
        return 'atrial_ectopic'
    elif 'atrial fibrillation' in report or 'atrial flutter' in report:
        return 'afib'
    elif 'idioventricular' in report:
        return 'idioventricular'
    elif 'supraventricular' in report:
        return 'supraventricular'
    elif 'accelerated junctional' in report or 'probable accelerated junctional rhythm' in report or 'possible accelerated junctional rhythm' in report:
        return 'accelerated_junctional'
    elif 'sinus tachycardia' in report:
        return 'sinus_tachy'
    elif 'sinus bradycardia' in report:
        return 'sinus_brady'
    elif 'sinus rhythm' in report or 'sinus arrhythmia' in report:
        return 'normal_sinus'
    else:
        return report
    