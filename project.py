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