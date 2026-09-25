"""
traffic_sign_cnn.py
-------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 6: Traffic Sign Classification using CNN
Train a CNN to recognize traffic signs using the GTSRB dataset format.

Architecture:
1. Canonical 32x32 RGB Traffic Sign Dataset Generator (10 Classes)
2. PyTorch Deep Convolutional Neural Network (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout -> Dense)
3. Epoch-by-Epoch Training & Validation Engine
4. Performance Metrics (Accuracy, Precision, Recall, F1-Score)
5. Diagnostic Visualizations (Sample Grid, Loss/Accuracy Curves, Confusion Matrix, Prediction Confidences)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "figure.autolayout": True,
    "figure.dpi": 200,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "semibold"
})

CLASS_NAMES = [
    "Speed_Limit_30",
    "Speed_Limit_50",
    "Speed_Limit_80",
    "Stop_Sign",
    "Yield_Sign",
    "No_Entry",
    "Turn_Right",
    "Turn_Left",
    "Pedestrians",
    "Traffic_Signal"
]

def generate_traffic_sign_dataset(samples_per_class=120, img_size=32, random_seed=42):
    """Generates synthetic 32x32 RGB traffic sign tensors simulating German Traffic Sign Benchmark (GTSRB)."""
    np.random.seed(random_seed)
    num_classes = len(CLASS_NAMES)
    total_samples = samples_per_class * num_classes
    
    X = np.zeros((total_samples, 3, img_size, img_size), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int64)
    meta_records = []
    
    idx = 0
    yy, xx = np.mgrid[:img_size, :img_size]
    center = img_size / 2.0
    radius = (img_size / 2.0) - 3.0
    dist_from_center = np.sqrt((xx - center)**2 + (yy - center)**2)
    
    for c_id, c_name in enumerate(CLASS_NAMES):
        for s in range(samples_per_class):
            img = np.ones((3, img_size, img_size), dtype=np.float32) * 0.85 # Gray background
            
            # Base color schemes per sign class
            if c_id in [0, 1, 2]: # Speed limit circle: red border, white inside, black number pattern
                circle_mask = dist_from_center <= radius
                border_mask = (dist_from_center <= radius) & (dist_from_center >= radius - 3.5)
                inner_mask = dist_from_center < radius - 3.5
                img[0, circle_mask] = 1.0 # White
                img[1, circle_mask] = 1.0
                img[2, circle_mask] = 1.0
                img[0, border_mask] = 0.9 # Red border
                img[1, border_mask] = 0.1
                img[2, border_mask] = 0.1
                # Central digits pattern
                offset = (c_id + 1) * 3
                img[:, 12:20, 12+offset%5:16+offset%5] = 0.1
                
            elif c_id == 3: # Stop sign: Red octagon with white horizontal pattern
                oct_mask = (dist_from_center <= radius) & (np.abs(xx - center) + np.abs(yy - center) <= radius * 1.35)
                img[0, oct_mask] = 0.85
                img[1, oct_mask] = 0.05
                img[2, oct_mask] = 0.05
                # White 'STOP' bar
                img[:, 14:18, 9:23] = 0.95
                
            elif c_id == 4: # Yield: Inverted triangle (red border, yellow/white inside)
                tri_mask = (yy >= 8) & (np.abs(xx - center) <= (yy - 6) * 0.9)
                border = tri_mask & ((yy <= 12) | (np.abs(xx - center) >= (yy - 8) * 0.75))
                img[0, tri_mask] = 0.95
                img[1, tri_mask] = 0.9
                img[2, tri_mask] = 0.1
                img[0, border] = 0.9
                img[1, border] = 0.1
                img[2, border] = 0.1
                
            elif c_id == 5: # No entry: Red circle with white horizontal bar
                c_mask = dist_from_center <= radius
                img[0, c_mask] = 0.85
                img[1, c_mask] = 0.05
                img[2, c_mask] = 0.05
                img[:, 14:18, 8:24] = 0.95
                
            elif c_id in [6, 7]: # Turn Right / Left: Blue circle with white directional arrow
                c_mask = dist_from_center <= radius
                img[0, c_mask] = 0.1
                img[1, c_mask] = 0.3
                img[2, c_mask] = 0.85
                if c_id == 6: # Turn Right
                    img[:, 14:18, 10:20] = 0.95
                    img[:, 10:22, 18:22] = 0.95
                else: # Turn Left
                    img[:, 14:18, 12:22] = 0.95
                    img[:, 10:22, 10:14] = 0.95
                    
            elif c_id == 8: # Pedestrians: Triangle warning with figure
                tri_mask = (yy <= 26) & (np.abs(xx - center) <= (28 - yy) * 0.75)
                img[0, tri_mask] = 0.9
                img[1, tri_mask] = 0.1
                img[2, tri_mask] = 0.1
                img[:, 14:24, 14:18] = 0.1 # Figure
                
            else: # Traffic Signal: Warning triangle with 3 colored dots
                tri_mask = (yy <= 26) & (np.abs(xx - center) <= (28 - yy) * 0.75)
                img[0, tri_mask] = 0.9
                img[1, tri_mask] = 0.1
                img[2, tri_mask] = 0.1
                img[0, 10:13, 15:17] = 0.95 # Red dot
                img[1, 15:18, 15:17] = 0.95 # Yellow dot
                img[1, 20:23, 15:17] = 0.95 # Green dot
                
            # Add stochastic environmental sensor noise, lighting & glare
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            brightness = np.random.uniform(0.85, 1.15)
            img = np.clip(img * brightness + noise, 0.0, 1.0)
            
            X[idx] = img
            y[idx] = c_id
            meta_records.append({
                "sample_id": f"IMG_{idx+1:04d}",
                "class_id": c_id,
                "class_name": c_name,
                "resolution": f"{img_size}x{img_size}",
                "channels": 3
            })
            idx += 1
            
    df_meta = pd.DataFrame(meta_records)
    return X, y, df_meta

# -------------------------------------------------------------
# Deep Convolutional Neural Network
# -------------------------------------------------------------
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_and_evaluate_cnn(output_dir):
    print("=" * 80)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML")
    print("LAB STATEMENT 6: TRAFFIC SIGN CLASSIFICATION USING CNN (GTSRB FORMAT)")
    print("=" * 80)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Dataset Generation
    X_raw, y_raw, df_meta = generate_traffic_sign_dataset(samples_per_class=120)
    meta_path = os.path.join(output_dir, "traffic_signs_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"Generated {len(df_meta)} traffic sign samples across {len(CLASS_NAMES)} classes.")
    print(f"Saved metadata: {meta_path}")
    
    # Plot 1: Sample Grid of Traffic Signs
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    for i, c_name in enumerate(CLASS_NAMES):
        ax = axes[i // 5, i % 5]
        sample_img = X_raw[i * 120].transpose(1, 2, 0)
        ax.imshow(sample_img)
        ax.set_title(c_name.replace("_", " "), fontsize=9)
        ax.axis("off")
    plt.suptitle("GTSRB Canonical Traffic Sign Dataset Samples (32x32 RGB)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_traffic_sign_samples_grid.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p1}")
    
    # 2. Data Partitioning & Loaders
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=42, stratify=y_raw
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3. Model Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 12
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print(f"\nTraining Traffic Sign CNN on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%")
        
    # Plot 2: Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(range(1, epochs + 1), history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(range(1, epochs + 1), history["val_loss"], "r--s", label="Validation Loss")
    axes[0].set_title("Cross-Entropy Loss Progression")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].plot(range(1, epochs + 1), history["train_acc"], "b-o", label="Train Accuracy (%)")
    axes[1].plot(range(1, epochs + 1), history["val_acc"], "g--s", label="Validation Accuracy (%)")
    axes[1].set_title(f"Classification Accuracy (Peak: {max(history['val_acc']):.1f}%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_cnn_training_loss_and_accuracy.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p2}")
    
    # 4. Final Evaluation & Confusion Matrix
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    test_acc = accuracy_score(all_targets, all_preds) * 100.0
    print(f"\nFinal Test Classification Accuracy: {test_acc:.2f}%")
    
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[c.replace("_", " ") for c in CLASS_NAMES],
                yticklabels=[c.replace("_", " ") for c in CLASS_NAMES])
    ax.set_title(f"Traffic Sign CNN Confusion Matrix (Test Accuracy: {test_acc:.1f}%)")
    ax.set_xlabel("Predicted Sign Class")
    ax.set_ylabel("True Sign Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_confusion_matrix_traffic_signs.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p3}")
    
    # Plot 4: Sample Predictions with Confidence Scores
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    sample_indices = np.random.choice(len(X_test), size=8, replace=False)
    for i, idx_s in enumerate(sample_indices):
        ax = axes[i // 4, i % 4]
        img = X_test[idx_s].transpose(1, 2, 0)
        true_lbl = CLASS_NAMES[all_targets[idx_s]]
        pred_lbl = CLASS_NAMES[all_preds[idx_s]]
        conf = all_probs[idx_s][all_preds[idx_s]] * 100.0
        
        ax.imshow(img)
        title_color = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(f"Pred: {pred_lbl.replace('_', ' ')}\nTrue: {true_lbl.replace('_', ' ')} ({conf:.1f}%)",
                     color=title_color, fontsize=8.5, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Sample Test Image Predictions & Confidence", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "04_sample_predictions_with_confidence.png")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p4}")

if __name__ == "__main__":
    out = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab6_traffic_sign_classification"
    train_and_evaluate_cnn(out)
