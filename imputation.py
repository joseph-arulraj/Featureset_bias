from benchpots.utils.missingness import create_missingness 
from collections import defaultdict
from pypots.optim import Adam
from pypots.imputation import BRITS
from pypots.utils.metrics import calc_mae
from .customBRITS import CustomBRITS
import torch
class Imputer:
    def __init__(self, train_batches, val_batches, test_batches):
        self.train_X  = [seq['sequences'] for seq in train_batches]
        self.val_X_ori = [seq['sequences'] for seq in val_batches]
        self.test_X_ori = [seq['sequences'] for seq in test_batches]
        self.val_X = [create_missingness(seq, rate=0.1, pattern='point') for seq in self.val_X_ori]
        self.test_X = [create_missingness(seq, rate=0.1, pattern='point') for seq in self.test_X_ori]
        self.dataset_for_training = defaultdict(lambda: {'X': torch.tensor, 'X_ori': torch.tensor})
        self.dataset_for_validation = defaultdict(lambda: {'X': torch.tensor, 'X_ori': torch.tensor})
        self.dataset_for_testing = defaultdict(lambda: {'X': torch.tensor, 'X_ori': torch.tensor})


    def prepare_data(self):

        for seq in self.train_X:
            seq_len = seq.shape[1]
            self.dataset_for_training[seq_len]['X'] = seq
            self.dataset_for_training[seq_len]['X_ori'] = seq

        for ori, missing in zip(self.val_X_ori, self.val_X):
            seq_len = ori.shape[1]
            self.dataset_for_validation[seq_len]['X'] = seq
            self.dataset_for_validation[seq_len]['X_ori'] = seq
        for ori, missing in zip(self.test_X_ori, self.test_X):
            seq_len = ori.shape[1]
            self.dataset_for_testing[seq_len]['X'] = seq
            self.dataset_for_testing[seq_len]['X_ori'] = seq

    def train_brits(self):
        # initialize the model
        self.brits = CustomBRITS(
            n_steps=289,
            n_features=17,
            rnn_hidden_size=128,
            batch_size=2,
            epochs=10,
            patience=3,
            optimizer=Adam(lr=1e-3),
            num_workers=0,
            device=None,
            saving_path="tutorial_results/imputation/brits",
            model_saving_strategy="best",
        )

        # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
        self.brits.fit(train_set=self.dataset_for_training, val_set=self.dataset_for_validation)

    def impute(self):
        train_imputation_results_groups = self.brits.predict(self.dataset_for_training)
        val_imputation_results_groups = self.brits.predict(self.dataset_for_validation)
        test_imputation_results_groups = self.brits.predict(self.dataset_for_testing)
        
        train_imputed_data_group = [train_imputation_results_groups[seq_len]['imputation'] for seq_len in train_imputation_results_groups.keys()]
        val_imputed_data_group = [val_imputation_results_groups[seq_len]['imputation'] for seq_len in val_imputation_results_groups.keys()]
        test_imputed_data_group = [test_imputation_results_groups[seq_len]['imputation'] for seq_len in test_imputation_results_groups.keys()]
        
        return train_imputed_data_group, val_imputed_data_group, test_imputed_data_group

