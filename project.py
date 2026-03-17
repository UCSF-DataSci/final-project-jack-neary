import os
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, recall_score
)
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

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

def vitals_query():
    query = """
    SELECT
        ce.stay_id
        , AVG(CASE WHEN itemid IN (220045)
                AND valuenum > 0
                AND valuenum < 300
                THEN valuenum END
        ) AS heart_rate
        , AVG(CASE WHEN itemid IN (220179, 220050, 225309)
                AND valuenum > 0
                AND valuenum < 400
                THEN valuenum END
        ) AS sbp
        , AVG(CASE WHEN itemid IN (220180, 220051, 225310)
                    AND valuenum > 0
                    AND valuenum < 300
                    THEN valuenum END
        ) AS dbp
        , AVG(CASE WHEN itemid IN (220052, 220181, 225312)
                    AND valuenum > 0
                    AND valuenum < 300
                    THEN valuenum END
        ) AS mbp
        , AVG(CASE WHEN itemid IN (220210, 224690)
                    AND valuenum > 0
                    AND valuenum < 70
                    THEN valuenum END
        ) AS resp_rate
        , ROUND(CAST(
                AVG(CASE
                    WHEN itemid IN (223761)
                        AND valuenum > 70
                        AND valuenum < 120
                        THEN (valuenum - 32) / 1.8
                    WHEN itemid IN (223762)
                        AND valuenum > 10
                        AND valuenum < 50
                        THEN valuenum END)
                AS NUMERIC), 2) AS temperature
        , AVG(CASE WHEN itemid IN (220277)
                    AND valuenum > 0
                    AND valuenum <= 100
                    THEN valuenum END
        ) AS spo2
        , AVG(CASE WHEN itemid IN (225664, 220621, 226537)
                    AND valuenum > 0
                    THEN valuenum END
        ) AS glucose
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
    INNER JOIN `physionet-data.mimiciv_3_1_icu.icustays` icu
        ON ce.stay_id = icu.stay_id
    WHERE ce.stay_id IS NOT NULL
        AND ce.charttime BETWEEN icu.intime AND DATETIME_ADD(icu.intime, INTERVAL 24 HOUR)
        AND ce.itemid IN
        (
            220045 -- Heart Rate
            , 225309 -- ART BP Systolic
            , 225310 -- ART BP Diastolic
            , 225312 -- ART BP Mean
            , 220050 -- Arterial Blood Pressure systolic
            , 220051 -- Arterial Blood Pressure diastolic
            , 220052 -- Arterial Blood Pressure mean
            , 220179 -- Non Invasive Blood Pressure systolic
            , 220180 -- Non Invasive Blood Pressure diastolic
            , 220181 -- Non Invasive Blood Pressure mean
            , 220210 -- Respiratory Rate
            , 224690 -- Respiratory Rate (Total)
            , 220277 -- SPO2, peripheral
            , 225664 -- Glucose finger stick
            , 220621 -- Glucose (serum)
            , 226537 -- Glucose (whole blood)
            , 223762 -- Temperature Celsius
            , 223761  -- Temperature Fahrenheit
        )
    GROUP BY ce.stay_id
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

def simplify_race(race):
    if pd.isna(race):
        return 'unknown'
    race = race.upper()
    if race in ['UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER']:
        return 'unknown'
    elif 'WHITE' in race or race == 'PORTUGUESE':
        return 'white'
    elif 'BLACK/AFRICAN AMERICAN' in race or 'BLACK/CAPE VERDEAN' in race or 'BLACK/CARIBBEAN ISLAND' in race:
        return 'black_american'
    elif 'BLACK/AFRICAN' in race:
        return 'black_african'
    elif 'HISPANIC' in race or 'LATINO' in race or 'SOUTH AMERICAN' in race:
        return 'hispanic_latino'
    elif 'ASIAN' in race:
        return 'asian'
    elif 'NATIVE HAWAIIAN' in race or 'PACIFIC ISLANDER' in race:
        return 'pacific_islander'
    elif 'AMERICAN INDIAN' in race or 'ALASKA NATIVE' in race:
        return 'native_american'
    elif race == 'MULTIPLE RACE/ETHNICITY':
        return 'multiple'
    else:
        return 'other'

def simplify_careunit(unit):
    if pd.isna(unit):
        return 'unknown'
    unit = unit.upper()
    if 'CVICU' in unit:
        return 'cvicu'
    elif 'MICU/SICU' in unit:
        return 'micu_sicu'
    elif 'MICU' in unit:
        return 'micu'
    elif 'CCU' in unit:
        return 'ccu'
    elif 'TSICU' in unit:
        return 'tsicu'
    elif 'SICU' in unit:
        return 'sicu'
    elif 'NEURO' in unit:
        return 'neuro'
    else:
        return 'other'
    
def plot_roc_curve(pipe, X_test, y_test, model_name, 
                   color='steelblue', threshold=None, ax=None):
    """
    Plot ROC curve for a fitted pipeline.
    
    Parameters:
        pipe       : fitted sklearn Pipeline
        X_test     : test features (raw, untransformed)
        y_test     : true labels
        model_name : string label for the plot title
        color      : line color
        threshold  : if provided, marks the operating point on the curve
        ax         : matplotlib axis to plot on (creates new if None)
    
    Returns:
        ax, roc_auc
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    # Mark threshold operating point if provided
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
        tpr_thresh = recall_score(y_test, y_pred, pos_label=1)
        fpr_thresh = 1 - recall_score(y_test, y_pred, pos_label=0)
        ax.scatter(fpr_thresh, tpr_thresh, color=color,
                   s=100, zorder=5, label=f"Threshold = {threshold}")

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{model_name} — ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    return ax, roc_auc

def plot_confusion_matrix(pipe, X_test, y_test, model_name,
                          threshold=0.5, ax=None):
    """
    Plot confusion matrix for a fitted pipeline.
    
    Parameters:
        pipe       : fitted sklearn Pipeline
        X_test     : test features (raw, untransformed)
        y_test     : true labels
        model_name : string label for the plot title
        threshold  : classification threshold (default 0.5)
        ax         : matplotlib axis to plot on (creates new if None)
    
    Returns:
        ax, cm (confusion matrix array)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Survived', 'Died']
    ).plot(ax=ax, colorbar=False, cmap='Blues')

    ax.set_title(f'{model_name}\n(threshold = {threshold})',
                  fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    return ax, cm

def get_pr_auc(pipe, X_test, y_test):
    """
    Calculate Precision-Recall AUC for a fitted pipeline.
    Average Precision Score = area under PR curve.
    
    Returns:
        pr_auc : float
    """
    y_prob = pipe.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    return pr_auc

if __name__ == "__main__":
    # Test the functions
    # icu_query()
    print("This is a test run of the project.py functions.")