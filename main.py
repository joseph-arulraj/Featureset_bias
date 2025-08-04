import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from .data_processing import Data
from .imputation import Imputer
from .feature_selection import FeatureImp
from .classification import LSTMClassifier, MLPClassifier
from .utils import train_model, evaluate_model
class EHRPipeline:
    def __init__(self, df, feature_cols, feature_importance_type, classification_model, epochs, n_features, n_steps, batch_size, patience, imputation_before_feature_selection=True):
        self.original_df = df
        self.feature_importance_type = feature_importance_type
        self.performance_dict = {}
        self.ordered_features = None
        self.imputation_before_feature_selection = imputation_before_feature_selection
        self.original_features = feature_cols
        self.classification_model = classification_model
        self.epochs = epochs
        self.n_features = n_features
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.patience = patience




    def run_pipeline(self):
        dataset_cl = Data(self.original_df)
        dataset_cl.split_df()
        dataset_cl.normalise_data()
        train_df = dataset_cl.train_df

        if self.imputation_before_feature_selection:
            self.train_batches, self.val_batches, self.test_batches = dataset_cl.process_data() 
            imputation_model = Imputer(self.train_batches, self.val_batches, self.test_batches, self.epochs, self.n_features, self.n_steps, self.batch_size, self.patience)
            imputation_model.prepare_data()
            imputation_model.train_brits()
            self.imputed_train, self.imputed_val, self.imputed_test = imputation_model.impute()
            print('Imputation done...................................')


        feature_importance = FeatureImp(train_df, self.original_features)
        if self.feature_importance_type == 'most_common':
            self.ordered_features = feature_importance.most_common()
        else:
            self.ordered_features = feature_importance.xgb()
        n = len(self.ordered_features)
        feature_dict = {
            'top25': self.ordered_features[:int(n * 0.25)],
            'top50': self.ordered_features[:int(n * 0.50)],
            'top75': self.ordered_features[:int(n * 0.75)],
            'all': self.ordered_features
        }
        
        for name, selected_features in feature_dict.items():
            print(f"\nTraining with {name} features: {selected_features}")
            
            if self.imputation_before_feature_selection:
                train_subset = self.imputed_train
                test_subset = self.imputed_test
                val_subset = self.imputed_val
                feature_indices = [self.original_features.index(f) for f in selected_features]

            for i, item in enumerate(self.train_batches):
                seq_len = item['seq_length']
                imputed_tensor = train_subset[seq_len]['imputation']  
                item['sequences'] = imputed_tensor[:, :, feature_indices]
            for i, item in enumerate(self.val_batches):
                seq_len = item['seq_length']
                imputed_tensor = val_subset[seq_len]['imputation']  
                item['sequences'] = imputed_tensor[:, :, feature_indices]
            for i, item in enumerate(self.test_batches):
                seq_len = item['seq_length']
                imputed_tensor = test_subset[seq_len]['imputation']  
                item['sequences'] = imputed_tensor[:, :, feature_indices]


                # self.train_batches['sequences'] = [seq[:, :, feature_indices]for seq in train_subset]
                # self.val_batches['sequences'] = [seq[:, :, feature_indices]for seq in val_subset]
                # self.test_batches['sequences'] = [seq[:, :, feature_indices]for seq in test_subset]

            input_size = len(selected_features)
            hidden_size = 128
            num_layers = 2
            num_classes = 2
            if self.classification_model == 'lstm':
                classifier = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
            elif self.classification_model == 'mlp':
                pass
                # classifier = MLPClassifier(mlp_input_size, hidden_size, num_classes)
            else:
                raise ValueError("Invalid classification model")
            
            train_model(classifier, self.train_batches, self.val_batches, num_epochs=10)
            test_acc, test_f1 = evaluate_model(classifier, self.test_batches)

            
            self.performance_dict[name] = test_acc

            # else:
            #     pass
        


    def plot_performance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(list(self.performance_dict.keys()), list(self.performance_dict.values()), marker='o')
        plt.xlabel('Number of Top Features')
        plt.ylabel('Accuracy')
        plt.title('Performance vs. Number of Top Features (Imputed Data)')
        plt.grid(True)
        plt.show()


        
        
        # # Step 3: Send to imputation.py
        # brits_model = train_brits(normalized_train)
        # imputed_train = impute_data(brits_model, normalized_train)
        # imputed_test = impute_data(brits_model, normalized_test)
        # imputed_val = impute_data(brits_model, normalized_val)

        # # Step 4: Call utils.py
        # train_data = reverse_batching(imputed_train)
        # test_data = reverse_batching(imputed_test)
        # val_data = reverse_batching(imputed_val)

        # # Step 5: Call feature_selection.py
        # if self.feature_importance_type == 'most_common':
        #     selected_features = most_common_features(train_data)
        # elif self.feature_importance_type == 'least_common':
        #     selected_features = least_common_features(train_data)
        # elif self.feature_importance_type == 'xgboost':
        #     selected_features = xgboost_features(train_data)
        # elif self.feature_importance_type == 'shap':
        #     selected_features = shap_features(train_data)
        # else:
        #     raise ValueError("Invalid feature importance type")

        # # Step 6: Loop over n top features with imputed data
        # for n in range(1, len(selected_features) + 1):
        #     top_features = selected_features[:n]
        #     train_subset = train_data[top_features]
        #     test_subset = test_data[top_features]
        #     val_subset = val_data[top_features]

        #     # Call classification model
        #     model = RNNClassifier(input_size=len(top_features))
        #     accuracy = classify_rnn(model, train_subset, test_subset, val_subset)
        #     self.performance_dict[n] = accuracy

        # # Step 7: Performance plot
        # self.plot_performance()

        # # Step 8: Loop over n top features from whole data
        # for n in range(1, len(selected_features) + 1):
        #     top_features = selected_features[:n]
        #     whole_data_subset = self.original_df[top_features]

        #     # Send to data_processing.py
        #     batches_whole = split_into_batches(whole_data_subset)
        #     train_batches_whole, test_batches_whole, val_batches_whole = split_train_test_val(batches_whole)
        #     normalized_train_whole, norm_factors_whole = normalize_data(train_batches_whole)
        #     normalized_test_whole = normalize_data(test_batches_whole, norm_factors_whole)
        #     normalized_val_whole = normalize_data(val_batches_whole, norm_factors_whole)

        #     # Send to imputation.py
        #     imputed_train_whole = impute_data(brits_model, normalized_train_whole)
        #     imputed_test_whole = impute_data(brits_model, normalized_test_whole)
        #     imputed_val_whole = impute_data(brits_model, normalized_val_whole)

        #     # Call classification model
        #     train_data_whole = reverse_batching(imputed_train_whole)
        #     test_data_whole = reverse_batching(imputed_test_whole)
        #     val_data_whole = reverse_batching(imputed_val_whole)
        #     model_whole = RNNClassifier(input_size=len(top_features))
        #     accuracy_whole = classify_rnn(model_whole, train_data_whole, test_data_whole, val_data_whole)
        #     self.performance_dict[f'whole_{n}'] = accuracy_whole

        # # Step 9: Performance plot
        # self.plot_performance_whole()

    # def plot_performance(self):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(list(self.performance_dict.keys()), list(self.performance_dict.values()), marker='o')
    #     plt.xlabel('Number of Top Features')
    #     plt.ylabel('Accuracy')
    #     plt.title('Performance vs. Number of Top Features (Imputed Data)')
    #     plt.grid(True)
    #     plt.show()

    # def plot_performance_whole(self):
    #     whole_keys = [key for key in self.performance_dict.keys() if 'whole' in key]
    #     whole_values = [self.performance_dict[key] for key in whole_keys]
    #     plt.figure(figsize=(10, 6))
    #     plt.plot([int(key.split('_')[1]) for key in whole_keys], whole_values, marker='o')
    #     plt.xlabel('Number of Top Features')
    #     plt.ylabel('Accuracy')
    #     plt.title('Performance vs. Number of Top Features (Whole Data)')
    #     plt.grid(True)
    #     plt.show()