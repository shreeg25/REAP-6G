import torch
import time
import numpy as np

# --- 1. IMPORT YOUR MODELS ---
# Adjust these imports to match your actual files!
from lstm_baseline import LSTMBeamTracker
from snn_model import build_model # (Or whatever function builds your SNN)

def measure_latency(model, model_name, device='cuda', seq_len=10, features=10, num_runs=1000):
    model.eval()
    model.to(device)
    
    # Batch size 1 simulates online edge inference
    dummy_input = torch.randn(1, seq_len, features).to(device)
    
    print(f"Warming up {model_name}...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
            
    print(f"Benchmarking {model_name} for {num_runs} iterations...")
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize() # CRITICAL: Forces GPU to finish before stopping clock
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000) # Convert to milliseconds
            
    avg_latency = np.mean(latencies)
    print(f"[+] {model_name} Average Inference Latency: {avg_latency:.4f} ms\n")
    return avg_latency

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. INITIALIZE MODELS
    lstm_model = LSTMBeamTracker(input_dim=10, output_dim=64).to(device)
    
    # FIX: Added your required parameters here
    snn_model = build_model(n_features=10, n_beams=64, device=device).to(device) 
    
    # 3. RUN BENCHMARKS
    measure_latency(lstm_model, "LSTM Baseline", device=device, features=10)
    measure_latency(snn_model, "REAP-6G SNN", device=device, features=10)