from featureset.utils import prepare_sequences, group_by_length
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, seq_lengths):
        self.sequences = sequences
        self.labels = labels
        self.seq_lengths = seq_lengths
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.seq_lengths[idx]
        )


class Data:
    def __init__(self,df):
        self.orig_df = df
        self.feature_columns = [ col for col in df.columns if col not in ['patientunitstayid', 'mortality', 'observationoffset']]  
        
        timeframe_counts = self.orig_df.groupby('patientunitstayid').size().reset_index(name='timeframe_count')

        duplicate_counts = timeframe_counts.groupby('timeframe_count').size().reset_index(name='n_patients')
        duplicate_counts = duplicate_counts[duplicate_counts['n_patients'] > 1]

        patients_with_same_timeframes = timeframe_counts.merge(duplicate_counts, on='timeframe_count')

        common_timeframe_counts = duplicate_counts['timeframe_count'].unique().tolist()
        all_ids = []
        # Iterate through the common counts and print the patient IDs for each
        for count in common_timeframe_counts:
            # Filter the joined dataframe for the current count
            patients_with_this_count = patients_with_same_timeframes[patients_with_same_timeframes['timeframe_count'] == count]
            
            # Get the list of patient IDs
            patient_ids = patients_with_this_count['patientunitstayid'].tolist()
            
            # Apply the condition: only include groups with 5 < len(patient_ids) < 80
            if len(patient_ids) > 5:
                all_ids.extend(patient_ids)
        
        self.df = self.orig_df[self.orig_df['patientunitstayid'].isin(all_ids)]
        

    def split_df(self):
        last_labels = self.df.sort_values(['patientunitstayid', 'observationoffset']).groupby('patientunitstayid').tail(1)[['patientunitstayid', 'mortality']]
        train_ratio = 0.7  # 70% for training
        val_ratio = 0.15   # 15% for validation
        # Test ratio is 0.15 (remainder)

        # First split: train + val vs test
        train_val_ids, test_ids = train_test_split(
            last_labels['patientunitstayid'],
            test_size=0.15,  # 15% for test
            random_state=42,
            stratify=last_labels['mortality']  # Stratify based on last label
        )

        # Second split: train vs val
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio / (train_ratio + val_ratio),  # 15% / (70% + 15%) ≈ 0.1765
            random_state=42,
            stratify=last_labels[last_labels['patientunitstayid'].isin(train_val_ids)]['mortality']
        )

        # Step 3: Filter df_merged into train, validation, and test sets
        self.train_df = self.df[self.df['patientunitstayid'].isin(train_ids)]
        self.val_df = self.df[self.df['patientunitstayid'].isin(val_ids)]
        self.test_df = self.df[self.df['patientunitstayid'].isin(test_ids)]

        self.train_patient_ids = train_ids.tolist()
        self.val_patient_ids = val_ids.tolist()
        self.test_patient_ids = test_ids.tolist()

    def normalise_data(self):
        scaler = StandardScaler()
        scaler.fit(self.train_df[self.feature_columns])

        self.train_df.loc[:, self.feature_columns] = scaler.transform(self.train_df[self.feature_columns])
        self.val_df.loc[:, self.feature_columns] = scaler.transform(self.val_df[self.feature_columns])
        self.test_df.loc[:, self.feature_columns] = scaler.transform(self.test_df[self.feature_columns])

    def process_data(self):
        # Prepare data for train, validation, and test
        train_sequences, train_labels, train_seq_lengths = prepare_sequences(self.train_df, self.train_patient_ids, self.feature_columns)
        val_sequences, val_labels, val_seq_lengths = prepare_sequences(self.val_df, self.val_patient_ids, self.feature_columns)
        test_sequences, test_labels, test_seq_lengths = prepare_sequences(self.test_df, self.test_patient_ids, self.feature_columns)
        # Create datasets and group by length
        # train_dataset = TimeSeriesDataset(train_sequences, train_labels, train_seq_lengths)
        # val_dataset = TimeSeriesDataset(val_sequences, val_labels, val_seq_lengths)
        # test_dataset = TimeSeriesDataset(test_sequences, test_labels, test_seq_lengths)

        train_batches = group_by_length(train_sequences, train_labels, train_seq_lengths)
        val_batches = group_by_length(val_sequences, val_labels, val_seq_lengths)
        test_batches = group_by_length(test_sequences, test_labels, test_seq_lengths)
        return train_batches, val_batches, test_batches
    

