import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from train_model import PointNetRegistration, FragmentDataset

def plot_learning_curve(model_dir):
    """Generates the required GSoC plot for training metrics."""
    history_file = os.path.join(model_dir, "loss_history.json")
    if not os.path.exists(history_file):
        print("Loss history not found. Has the model been trained yet?")
        return
        
    with open(history_file, "r") as f:
        loss_history = json.load(f)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.title("PointNet 3D Registration - Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    
    plot_path = os.path.join(model_dir, "training_loss_curve.png")
    plt.savefig(plot_path)
    print(f"\n[Plot Output] Saved learning curve to: {plot_path}")
    
def evaluate_metrics():
    """Evaluates the Integrity of generated data against Ground Truth."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "training_data", "synthetic_fractures")
    
    model_path = os.path.join(model_dir, "pointnet_healer.pth")
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Please run train_model.py first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Model
    model = PointNetRegistration().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Testing Data
    dataset = FragmentDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("\n========================================================")
    print(" GSoC Healing Stones - ML Evaluation & Metrics Report")
    print("========================================================")
    print(f"Total Test Fragments: {len(dataset)}")
    
    total_mse = 0.0
    total_mae = 0.0
    total_curv_loss = 0.0
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    from train_model import curvature_weighted_point_loss
    
    with torch.no_grad():
        for i, (data, weights, target) in enumerate(dataloader):
            data, weights, target = data.to(device), weights.to(device), target.to(device)
            
            # AI Inference
            prediction = model(data)
            
            # Calculate metrics
            mse = criterion_mse(prediction, target).item()
            mae = criterion_mae(prediction, target).item()
            curv = curvature_weighted_point_loss(prediction, target, data, weights).item()
            
            total_mse += mse
            total_mae += mae
            total_curv_loss += curv
            
            print(f"  Fragment {i:03d} | MSE: {mse:.4f} | MAE: {mae:.4f} | Curvature Loss: {curv:.4f}")
            
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    avg_curv = total_curv_loss / len(dataloader)
    
    print("\n--- Final 'Integrity of Generated Data' Metrics ---")
    print(f"Mean Absolute Error (Feature space): {avg_mae:.4f}")
    print(f"Mean Squared Error (Feature space): {avg_mse:.4f}")
    print(f"Curvature-Weighted Point Loss: {avg_curv:.4f}")
    print("--------------------------------------------------------\n")
    
    # Generate Plots
    plot_learning_curve(model_dir)

if __name__ == "__main__":
    evaluate_metrics()
