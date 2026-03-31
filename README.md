# GSoC 2025: Healing Stones - AI Reconstruction Pipeline

This repository contains the full end-to-end pipeline for the CERN **Healing Stones** Machine Learning project. The goal is to mathematically and procedurally reconstruct fragmented 3D cultural heritage artifacts (Mayan stele) using geometric feature matching and deep learning.

## 🏛️ Project Overview
Fragmented works of art require intelligent, scalable pipelines to be restored digitally. Standard Machine Learning relies on massive datasets, but archaeological artifacts are sparse (only 12 source fragments provided). 

To solve this, my submission provides a **two-pronged architecture**:
1. **Mathematical Baseline (ICP Assembly)**: A greedy algorithm that uses Fast Point Feature Histograms (FPFH), RANSAC, and Point-to-Plane ICP to interlock the raw fragments without AI. 
2. **Machine Learning Pipeline (PointNet)**: A procedural data-augmentation script that artificially shatters complete stones to generate infinite training data, feeding into a PyTorch PointNet formulation to regress the SE(3) spatial coordinates.

---

## 🛠️ Usage & Deliverables Required

I have prioritized **usability** and **automation**, as requested in the GSoC prompt. The entire pipeline can be run from start to finish *without user intervention* using the provided Windows batch files (`.bat`).

### Phase 1: Mathematical Reconstruction (Baseline)
If you want to see the 12 fragments mathematically snap together into the full 3D Mayan Stele:
*   **Run**: `run_reconstruction.bat`
*   **Output**: Finds all pieces, iteratively computes interlocking geometrical edges, and outputs the final artifact as `reconstructed_mayan_stone.ply`.

### Phase 2: Machine Learning Data Generation
Machine Learning models cannot train on 12 samples. To solve the data scarcity issue:
*   **Run**: `run_augmentation.bat`
*   **Action**: Procedurally cuts the final mesh into random chunks, applies SE(3) permutations to scatter them, and calculates the inverse transformation matrices.
*   **Output**: Generates a massive training dataset in `training_data/synthetic_fractures/` with a mathematical `ground_truth.json` label dictionary.

### Phase 3: ML Model Creation & Training
*   **Run**: `run_training.bat`
*   **Action**: Bootstraps a PyTorch environment, builds a 1D Convolutional Neural Network (PointNet style), and attempts to regress the transformation parameters to "undo" the spatial scattering.
*   **Output**: Saves the trained deep learning weights to `models/pointnet_healer.pth`.

### Phase 4: Testing & Metrics Output
To fulfill the *"produce clear metrics and/or plots"* requirement:
*   **Run**: `run_evaluation.bat`
*   **Metrics**: Performs `model.eval()` inference, printing out the explicit **Mean Squared Error (MSE)** to quantify the **Integrity of generated data**.
*   **Plots**: Generates the formal Deep Learning loss curve converging over time at `models/training_loss_curve.png`.

---

## ⚙️ Technical Highlights
* **FPFH (Fast Point Feature Histograms)**: Used to mathematically fingerprint the curvature of the broken edges, allowing the model to know which "key" fits the "lock".
* **Point-to-Plane ICP**: Utilized specifically because traditional point-to-point math struggles with flat surfaces. Point-to-plane "slides" the flat broken artifact faces against each other to minimize microscopic gaps.
* **Memory Management**: The pipeline utilizes Python `gc` collection and strict 1cm Voxel Downsampling (`voxel_down_sample(0.01)`) to ensure that processing 10GB+ of raw `.PLY` point clouds does not crash standard hardware.
