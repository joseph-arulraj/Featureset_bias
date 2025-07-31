import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

class FeatureImp:
    def __init__(self, df, feature_cols, xgb_agg_type='mean'):
        self.df = df
        self.feature_cols = feature_cols
        self.xgb_agg_type = xgb_agg_type
    def most_common(self):
        freq_per_patient = self.df.groupby('patientunitstayid')[self.feature_columns].apply(lambda x: x.notnull().sum())
        mean_freq = freq_per_patient.mean()
        ordered_features = mean_freq.sort_values(ascending=False).index.tolist()
        return ordered_features
    
    def xgb(self):
        train_agg = self.df.groupby('patientunitstayid')[self.feature_columns].agg(self.xgb_agg_type)
        train_labels_agg = self.df.groupby('patientunitstayid')['mortality'].last()
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(train_agg, train_labels_agg)
        importances = xgb_model.feature_importances_
        feature_importance = pd.Series(importances, index=self.feature_columns).sort_values(ascending=False)
        ordered_features = feature_importance.index.tolist()
        return ordered_features

