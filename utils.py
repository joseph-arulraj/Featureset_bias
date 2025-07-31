from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

def prepare_sequences(df, patient_ids, feature_columns):
    sequences = []
    labels = []
    seq_lengths = []
    
    # Get last label per patient
    last_labels = df.sort_values(['patientunitstayid', 'observationoffset']).groupby('patientunitstayid').tail(1)[['patientunitstayid', 'mortality']]
    
    for pid in patient_ids:
        patient_data = df[df['patientunitstayid'] == pid].sort_values('observationoffset')
        # sequence = patient_data[feature_columns].values  # Shape: (seq_len, num_features)
        sequence = patient_data[feature_columns].to_numpy(dtype=np.float32)

        label = last_labels[last_labels['patientunitstayid'] == pid]['mortality'].iloc[0]
        seq_length = len(sequence)
        
        sequences.append(sequence)
        labels.append(label)
        seq_lengths.append(seq_length)
    
    return sequences, labels, seq_lengths

def group_by_length(sequences, labels, seq_lengths):
    length_to_indices = defaultdict(list)
    for idx, seq_len in enumerate(seq_lengths):
        length_to_indices[seq_len].append(idx)
    
    batches = []
    for seq_len, indices in length_to_indices.items():
        batch_sequences = [sequences[i] for i in indices]
        batch_labels = [labels[i] for i in indices]
        # for s in batch_sequences:
        #     print(s.shape)

        batches.append({
            'sequences': torch.tensor(np.stack(batch_sequences), dtype=torch.float32),  # Shape: (batch_size, seq_len, num_features)
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'seq_length': seq_len
        })
    return batches

# Step 6: Training Function
def train_model(model, train_batches, val_batches, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        
        np.random.shuffle(train_batches)
        
        for batch in train_batches:
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_true.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average='weighted')
        
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_batches:
                sequences = batch['sequences'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_batches):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss/len(val_batches):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')

# Step 7: Evaluation Function
def evaluate_model(model, test_batches, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    test_preds, test_true = [], []
    
    with torch.no_grad():
        for batch in test_batches:
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(sequences)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average='weighted')
    print(f'Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    return test_acc, test_f1