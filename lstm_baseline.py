import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
from tqdm import tqdm

# 1. The LSTM Baseline Architecture
class LSTMBeamTracker(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=2, output_dim=64, dropout=0.2):
        super(LSTMBeamTracker, self).__init__()
        # 2-Layer LSTM to match your 2-Layer LIF SNN
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        out, _ = self.lstm(x)
        # Decode the hidden states into beam logits
        logits = self.fc(out) 
        return logits

def train_lstm_baseline(train_loader, val_loader, epochs=40, lr=0.0005, device='cuda', save_path="lstm_baseline_weights.pth"):
    model = LSTMBeamTracker().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    training_history = []

    print("\n--- Starting LSTM Baseline Training ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=False)
        
        # FIX: Unpack safely regardless of how many items the dataloader returns
        for batch in pbar:
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            
            loss = criterion(logits.view(-1, 64), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} Completed | Avg Loss: {avg_loss:.4f}")
        training_history.append([epoch+1, avg_loss])
        
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Model weights saved to {save_path}")
    
    log_file = "lstm_training_logs.csv"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training_Loss"])
        writer.writerows(training_history)
    print(f"[+] Training logs saved to {log_file}")
        
    return model, log_file

def evaluate_baseline(model, test_loader, log_file, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        # FIX: Safe unpacking here too!
        for batch in test_loader:
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            logits = model(X_batch)
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()
            
    top1_acc = (correct / total) * 100
    print(f"LSTM Top-1 Accuracy: {top1_acc:.2f}%")
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Final_Top1_Accuracy", f"{top1_acc:.2f}%"])
        
    return top1_acc

if __name__ == "__main__":
    import torch
    import os
    import numpy as np
    
    print("\n[1] Preparing Data...")
    
    # --- 1. IMPORTS ---
    from deepmimo_loader import (
        load_deepmimo_multifile, DeepMIMODataset,
        _synthetic_grid, _synthesize_channels, _generate_dft_codebook,
        compute_beam_gains
    )
    from trajectory_generator import generate_trajectories, trajectories_to_sequences
    from trainer import build_dataloaders
    
    # --- 2. CONFIGURATION ---
    N_TRAJ = 50
    N_STEPS = 850   
    SEQ_LEN = 10    
    N_BEAMS = 64    
    DATA_DIR = r"G:\Shree\6G Beam Switching enabled by SNN\6G Dataset creation\deepmimo_scenarios\O1_140"
    
    # NOTE: Set these to match your run_pipeline.py!
    T_INDEX = 1  
    TX_INDEX = 1 
    
    # --- 3. DATA GENERATION ---
    print(f"\n[Step 1] Loading dataset ...")
    if os.path.isdir(DATA_DIR):
        ds = load_deepmimo_multifile(DATA_DIR, t_index=T_INDEX, tx_index=TX_INDEX, n_beams=N_BEAMS)
    else:
        print("CRITICAL ERROR: Could not find DATA_DIR. Check your path!")
        exit()

    # --- MASTER DATA SCRUBBER ---
    print("\n[Step 1.5] Master Purge: Eradicating NaN/Inf artifacts from all MATLAB tensors...")
    ds.user_locations = np.nan_to_num(ds.user_locations, nan=0.0, posinf=0.0, neginf=0.0)
    ds.tx_location = np.nan_to_num(ds.tx_location, nan=0.0, posinf=0.0, neginf=0.0)
    ds.path_power = np.nan_to_num(ds.path_power, nan=-200.0, posinf=-200.0, neginf=-200.0)
    for attr in ['path_delay', 'path_phase', 'aod_az', 'aod_el', 'aoa_az', 'aoa_el']:
        setattr(ds, attr, np.nan_to_num(getattr(ds, attr), nan=0.0, posinf=0.0, neginf=0.0))
    ds.channels = _synthesize_channels(ds, N_rx=4, N_tx=64)
    ds.channels = np.nan_to_num(ds.channels, nan=0.0, posinf=0.0, neginf=0.0)
    # ----------------------------
    
    print("\nGenerating trajectories...")
    trajectories = generate_trajectories(ds, n_trajectories=N_TRAJ,
                                         n_steps=N_STEPS, dt=0.5, top_k=5, seed=42)
    
    print("Converting to sequences...")
    X, y, y_topk = trajectories_to_sequences(trajectories, seq_len=SEQ_LEN, stride=5)
    n_features   = X.shape[-1]
    
    # --- 4. DATALOADERS ---
    print(f"\nBuilding dataloaders for {n_features} features (75/15/10 split)...")
    train_loader, val_loader, test_loader = build_dataloaders(X, y, y_topk) 
    
    print("\n[2] Initializing LSTM Baseline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 5. START TRAINING ---
    trained_lstm, log_filename = train_lstm_baseline(train_loader, val_loader, epochs=40, device=device)
    evaluate_baseline(trained_lstm, test_loader, log_filename, device=device)