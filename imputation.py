from benchpots.utils.missingness import create_missingness 
from collections import defaultdict
from pypots.optim import Adam
from pypots.imputation import BRITS
from pypots.utils.metrics import calc_mae
from .customBRITS import CustomBRITS
import torch
class Imputer:
    def __init__(self, train_batches, val_batches, test_batches, epochs, n_features, n_steps, batch_size, patience):
        self.train_X  = [seq['sequences'] for seq in train_batches]
        self.val_X_ori = [seq['sequences'] for seq in val_batches]
        self.test_X_ori = [seq['sequences'] for seq in test_batches]
        self.val_X = [create_missingness(seq, rate=0.1, pattern='point') for seq in self.val_X_ori]
        self.test_X = [create_missingness(seq, rate=0.1, pattern='point') for seq in self.test_X_ori]
        self.dataset_for_training = defaultdict(lambda: {'X': torch.tensor, 'X_ori': torch.tensor})
        self.dataset_for_validation = defaultdict(lambda: {'X': torch.tensor, 'X_ori': torch.tensor})
        self.dataset_for_testing = defaultdict(lambda: {'X': torch.tensor, 'X_ori': torch.tensor})
        self.epochs = epochs
        self.n_features = n_features
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.patience = patience 



    def prepare_data(self):

        for seq in self.train_X:
            seq_len = seq.shape[1]
            self.dataset_for_training[seq_len]['X'] = seq
            self.dataset_for_training[seq_len]['X_ori'] = seq

        for ori, missing in zip(self.val_X_ori, self.val_X):
            seq_len = ori.shape[1]
            self.dataset_for_validation[seq_len]['X'] = missing
            self.dataset_for_validation[seq_len]['X_ori'] = ori
        for ori, missing in zip(self.test_X_ori, self.test_X):
            seq_len = ori.shape[1]
            self.dataset_for_testing[seq_len]['X'] = missing
            self.dataset_for_testing[seq_len]['X_ori'] = ori

    def train_brits(self):
        # initialize the model
        self.brits = CustomBRITS(
            n_steps=self.n_steps,
            n_features=self.n_features,
            rnn_hidden_size=128,
            batch_size=self.batch_size,
            epochs=self.epochs,
            patience=self.patience,
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
        
        return train_imputation_results_groups, val_imputation_results_groups, test_imputation_results_groups