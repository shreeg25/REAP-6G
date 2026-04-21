import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINE THE GRU ARCHITECTURE
# ==========================================
class GRUBeamTracker(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=64):
        super(GRUBeamTracker, self).__init__()
        # batch_first=True expects shape [batch, seq_len, features]
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch, 10, 10] (assuming seq_len=10, features=10)
        out, _ = self.gru(x)
        # Grab the output from the final time step
        final_out = out[:, -1, :] 
        logits = self.fc(final_out)
        return logits

# ==========================================
# 2. CONFIGURATION & HYPERPARAMETERS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
LEARNING_RATE = 0.0005 # Matching your paper's AdamW parameters
INPUT_DIM = 10
HIDDEN_DIM = 128
OUTPUT_DIM = 64 # 64-beam codebook

# ==========================================
# 3. MAIN TRAINING & EVALUATION LOOP
# ==========================================
def train_and_evaluate_gru(train_loader, test_loader):
    print(f"[*] Initializing GRU Baseline on {device}...")
    model = GRUBeamTracker(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # --- TRAINING LOOP ---
    print("\n[*] Starting Training...")
    epoch_losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            # SLICE to grab only the final timestep's target, then convert to 1D integer
            batch_y = batch_y[:, -1].to(device).long()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_epoch_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")
            
    # --- EVALUATION LOOP ---
    print("\n[*] Starting Evaluation...")
    model.eval()
    correct_top1 = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y, _ in test_loader:
            batch_x = batch_x.to(device)
            # SLICE to grab only the final timestep's target, then convert to 1D integer
            batch_y = batch_y[:, -1].to(device).long()
            outputs = model(batch_x)
            
            # Store probabilities for saving
            probs = torch.softmax(outputs, dim=1)
            
            # Calculate Top-1
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct_top1 += (predicted == batch_y).sum().item()
            
            # Store arrays for the .npz file
            all_targets.extend(batch_y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
    top1_acc = 100 * correct_top1 / total
    print(f"\n======================================")
    print(f"[+] Final GRU Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"======================================")
    
    # ==========================================
    # 4. SAVE MODEL WEIGHTS AND PREDICTIONS
    # ==========================================
    torch.save(model.state_dict(), 'gru_baseline_weights.pth')
    print("\n[+] Model weights saved to 'gru_baseline_weights.pth'")
    
    np.savez_compressed(
        'gru_evaluation_results.npz', 
        targets=np.array(all_targets), 
        predictions=np.array(all_predictions), 
        probabilities=np.array(all_probabilities)
    )
    print("[+] Evaluation data saved to 'gru_evaluation_results.npz' for plotting.")
    
    return top1_acc

    # ==========================================
    # 5. GENERATE TRAINING VISUALS
    # ==========================================
    print("\n[*] Generating visualization graphics...")
    plt.figure(figsize=(14, 5))

    # Plot 1: The Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', color='#d62728', linewidth=2)
    plt.title('GRU Baseline: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: True vs Predicted Beams (First 100 samples)
    plt.subplot(1, 2, 2)
    samples_to_show = min(100, len(all_targets))
    plt.plot(all_targets[:samples_to_show], label='True Beam', marker='s', markersize=6, linestyle='-', color='black', alpha=0.6)
    plt.plot(all_predictions[:samples_to_show], label='GRU Prediction', marker='x', markersize=8, linestyle='', color='#1f77b4')
    plt.title(f'Tracking Accuracy (First {samples_to_show} Test Steps)')
    plt.xlabel('Sequential Time Step')
    plt.ylabel('Beam Index (0-63)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # Save it at 300 DPI just like your IEEE paper figures!
    plt.savefig('gru_training_visuals.png', dpi=300, bbox_inches='tight')
    print("[+] Visuals successfully saved to 'gru_training_visuals.png'. Open this file to verify training!")

# ==========================================
# 6. EXECUTION
# ==========================================
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
    N_TRAJ = 250
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
    
    print("\n[2] Initializing GRU Baseline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 5. START GRU TRAINING ---
    # Call the new GRU function we just wrote, passing in your fresh dataloaders
    train_and_evaluate_gru(train_loader, test_loader)