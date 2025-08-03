import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from .main import EHRPipeline
# from featureset.data_processing import Data
# from featureset.imputation import Imputer


BASE_PATH = "./eicu_demo" 
patient_path = os.path.join(BASE_PATH, "patient.csv")
vitals_path = os.path.join(BASE_PATH, "vitalPeriodic.csv")
lab_path = os.path.join(BASE_PATH, "lab.csv")
apache_path = os.path.join(BASE_PATH, "apachePatientResult.csv")
diagnosis_path = os.path.join(BASE_PATH, "diagnosis.csv")
treatment_path = os.path.join(BASE_PATH, "treatment.csv")

q_patient = pd.read_csv(patient_path)[['patientunitstayid', 'gender', 'hospitaldischargestatus']]

# Load vitals data
q_vitals = pd.read_csv(vitals_path)

# Load and filter apache data for apacheversion == 'IV', then select columns
q_apache = pd.read_csv(apache_path)
q_apache = q_apache[q_apache['apacheversion'] == 'IV'][['patientunitstayid', 'apachescore', 'predictedhospitalmortality', 'predictedicumortality']]

# Perform left join operations
# df_merged = q_patient.merge(q_apache, on='patientunitstayid', how='left').merge(q_vitals, on='patientunitstayid', how='left')
df_merged = q_patient.merge(q_vitals, on='patientunitstayid', how='left')
df_merged['mortality'] = np.where(df_merged['hospitaldischargestatus'] == 'Expired', 1, 0).astype(float)

df_merged['gender'] = np.select(
    [df_merged['gender'] == 'Male', df_merged['gender'] == 'Female'],
    [1, 0],
    default=None
).astype(float) 
# Identify columns with string (object) dtype
cols_to_convert = df_merged.select_dtypes(include=['object']).columns.tolist()

# Convert the selected string columns to float64
for col in cols_to_convert:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce', downcast='float')

# Drop specified columns
df_clean = df_merged.drop(columns=['hospitaldischargestatus', 'vitalperiodicid'])
feature_cols = [ col for col in df_clean.columns if col not in ['patientunitstayid', 'mortality', 'observationoffset', 'gender', 'vitalperiodicid','hospitaldischargestatus']]  # Replace with actual feature column names

pipeline = EHRPipeline(df_clean, feature_cols, 'most_common', 'lstm', 4, len(feature_cols), 400, 5, 2)
pipeline.run_pipeline()
print('......................done......................')
print(pipeline.performance_dict)

# dataset_cl = Data(df_clean)
# dataset_cl.split_df()
# dataset_cl.normalise_data()
# train_df = dataset_cl.train_df
# train_batches, val_batches, test_batches = dataset_cl.process_data() 
# imputation_model = Imputer(train_batches, val_batches, test_batches)
# imputation_model.prepare_data()
# imputation_model.train_brits()
# imputed_train, imputed_val, imputed_test = imputation_model.impute()