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
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        
        # Randomly sample fixed number of points for the Neural Network
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        else: # Pad if too small
            pad = np.zeros((self.num_points - len(points), 3))
            points = np.vstack([points, pad])
            
        # Neural Networks expect shape: (Channels, NumPoints) -> (3, N)
        points = points.T
        points_tensor = torch.FloatTensor(points)
        
        # Load Ground Truth transformation matrix (4x4)
        T_gt = np.array(self.ground_truth[filename], dtype=np.float32)
        # We only need the top 3 rows (rotation and translation) for training
        T_target = torch.FloatTensor(T_gt[:3, :]) 
        
        return points_tensor, T_target

# ==========================================
# 2. PointNet Architecture
# ==========================================
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
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Global feature processing
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # Output 12 values: (3x3 Rotation + 3x1 Translation)
        self.fc3 = nn.Linear(256, 12)
        
    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) # Shape: (Batch, 1024, NumPoints)
        
        # "Max Pooling" - this creates the global geometric signature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to 3x4 matrix
        return x.view(-1, 3, 4)

# ==========================================
# 3. Training Loop
# ==========================================
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Mean Squared Error
    
    epochs = 20
    loss_history = []
    
    print("\n--- Starting Deep Learning Pipeline ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            prediction = model(data)
            
            # Calculate error
            loss = criterion(prediction, target)
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
