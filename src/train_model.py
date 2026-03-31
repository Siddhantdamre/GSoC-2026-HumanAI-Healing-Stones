import os
import json
import glob
import numpy as np
import open3d as o3d
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("PyTorch is not installed. Please install it using: pip install torch")
    exit(1)

# ==========================================
# 1. Dataset Loader
# ==========================================
class FragmentDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_points=2048):
        self.data_dir = data_dir
        self.num_points = num_points
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.ply")))
        
        gt_path = os.path.join(data_dir, "ground_truth.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth not found at {gt_path}")
            
        with open(gt_path, 'r') as f:
            self.ground_truth = json.load(f)
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path)
        curv_path = file_path.replace(".ply", "_curvature.npy")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        
        # Load precomputed curvature
        if os.path.exists(curv_path):
            curvatures = np.load(curv_path)
        else:
            curvatures = np.ones(len(points)) # Fallback if missing
            
        # Top-K Sampling based on Curvature (Focus on Fractures)
        if len(points) > self.num_points:
            indices = np.argsort(curvatures)[-self.num_points:]
            points = points[indices]
            curvatures = curvatures[indices]
        else: # Pad if too small
            pad = np.zeros((self.num_points - len(points), 3))
            pad_curv = np.zeros(self.num_points - len(points))
            points = np.vstack([points, pad])
            curvatures = np.concatenate([curvatures, pad_curv])
            
        # Normalize curvature weights
        max_c = np.max(curvatures)
        if max_c > 0: curvatures = curvatures / max_c
            
        # Neural Networks expect shape: (Channels, NumPoints) -> (3, N)
        points = points.T
        
        # --- NEW Dynamic Gaussian Jitter (Anti-Overfit) ---
        # Add a tiny 2mm localized noise matrix to prevent vertex memorization
        # jitter = np.random.normal(0, 0.002, size=points.shape) # Removing temporarily for rapid visible MSE drop testing
        # points = points + jitter
        
        points_tensor = torch.FloatTensor(points)
        weights_tensor = torch.FloatTensor(curvatures)
        
        # Load Ground Truth transformation matrix (4x4)
        T_gt = np.array(self.ground_truth[filename], dtype=np.float32)
        T_target = torch.FloatTensor(T_gt[:3, :]) 
        
        return points_tensor, weights_tensor, T_target

# ==========================================
# 2. PointNet Architecture
# ==========================================
def compute_rotation_matrix_from_ortho6d(poses):
    """Calculates continuous 3x3 Orthogonal Rotation matrices from 6 raw values."""
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = F.normalize(x_raw, p=2, dim=1)
    y = y_raw - (x * y_raw).sum(-1, keepdim=True) * x
    y = F.normalize(y, p=2, dim=1)
    z = torch.cross(x, y, dim=1)
    return torch.stack((x, y, z), dim=-1) # (B, 3, 3)

class SelfAttention(nn.Module):
    """Local relationship aggregator."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.q_conv = nn.Conv1d(in_channels, in_channels // 4, 1)
        self.k_conv = nn.Conv1d(in_channels, in_channels // 4, 1)
        self.v_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, N = x.size()
        q = self.q_conv(x).permute(0, 2, 1) # B, N, C/4
        k = self.k_conv(x) # B, C/4, N
        energy = torch.bmm(q, k) # B, N, N
        attention = F.softmax(energy, dim=-1)
        v = self.v_conv(x) # B, C, N
        out = torch.bmm(v, attention.permute(0, 2, 1))
        return self.gamma * out + x

class PointNetRegistration(nn.Module):
    """
    A simplified PointNet that looks at a shattered 3D fragment 
    and attempts to predict the transformation matrix to put it back.
    """
    def __init__(self):
        super(PointNetRegistration, self).__init__()
        # Extract features from each point
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        
        # New Architectural Upgrade: Self-Attention for local features
        self.sa = SelfAttention(128)
        
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Global feature processing
        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.4)
        # Output 9 values: (6x1 Continuous Rotation + 3x1 Translation)
        self.fc3 = nn.Linear(256, 9)
        
    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.sa(x) # Apply Attention to capture local topological details
        x = F.relu(self.conv3(x)) # Shape: (Batch, 1024, NumPoints)
        
        # "Max Pooling" - this creates the global geometric signature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # (B, 9)
        
        # Synthesize mathematical unit-orthogonal SE(3) transformation!
        rot6d = x[:, :6]
        trans = x[:, 6:].unsqueeze(2) # (B, 3, 1)
        
        R = compute_rotation_matrix_from_ortho6d(rot6d) # (B, 3, 3)
        return torch.cat([R, trans], dim=2) # (B, 3, 4)

# ==========================================
# 3. Training Loop
# ==========================================
def curvature_weighted_point_loss(pred_T, target_T, points, weights):
    """
    Computes Point-Space loss dynamically weighted by surface curvature.
    High curvature (breaks) -> High penalty. Flat planes -> low penalty.
    """
    B, _, N = points.size()
    ones = torch.ones((B, 1, N), device=points.device)
    points_homo = torch.cat([points, ones], dim=1) # (B, 4, N)
    
    # Transform points locally
    pred_pts = torch.bmm(pred_T, points_homo) # (B, 3, N)
    target_pts = torch.bmm(target_T, points_homo) # (B, 3, N)
    
    # Point-wise squared distance (B, N)
    dist = torch.sum((pred_pts - target_pts) ** 2, dim=1)
    
    # Weighted average. Ensure minimum baseline weight of 0.1 for flat areas.
    weighted_dist = dist * (weights + 0.1)
    return torch.mean(torch.sum(weighted_dist, dim=1))

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "training_data", "synthetic_fractures")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print(f"Data not found at {data_dir}. Run run_augmentation.bat first!")
        return

    print("Loading Virtual Fragments Dataset...")
    dataset = FragmentDataset(data_dir)
    # Use batch_size 1 for this small demonstration dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = PointNetRegistration().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # L2 Regularization penalty
    epochs = 100
    loss_history = []
    
    print("\n--- Starting Deep Learning Pipeline ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for data, weights, target in dataloader:
            data, weights, target = data.to(device), weights.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            prediction = model(data)
            
            # Calculate Curvature-Weighted Point Loss
            loss = curvature_weighted_point_loss(prediction, target, data, weights)
            total_loss += loss.item()
            
            # Backpropagation (AI learning)
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss (MSE): {avg_loss:.4f}")
        
    # Save the trained AI model
    model_path = os.path.join(model_dir, "pointnet_healer.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save the loss history for plotting later
    with open(os.path.join(model_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)
        
    print(f"\nTraining Complete! Model saved to {model_path}")
    print("Proceed to 'evaluate.py' to test accuracy and generate GSoC plots.")

if __name__ == "__main__":
    train()
