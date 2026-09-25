"""
sentiment_analysis_lstm.py
--------------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 8: Sentiment Analysis using LSTM
Analyze vehicle feedback using LSTM-based sentiment classification.

Curriculum Context:
- Unit 4: AI fundamentals & application development (RNN and LSTM overview,
  Text data preprocessing, Evaluation metrics: precision, recall, F1-score)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import re
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

SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def generate_vehicle_reviews_dataset(n_samples=1200, random_seed=42):
    """Generates realistic automotive customer feedback reviews across 3 sentiment categories."""
    np.random.seed(random_seed)
    
    positive_phrases = [
        "Outstanding fuel economy and exceptionally smooth automatic transmission on highway.",
        "The ride comfort and suspension absorption on potholes is truly best in class.",
        "Tata build quality feels sturdy solid and the 5 star crash safety gives peace of mind.",
        "Infotainment screen is responsive Apple CarPlay connects seamlessly and audio is crisp.",
        "Incredible electric acceleration instant torque and regenerative braking works flawlessly.",
        "Spacious cabin excellent legroom and the ventilated leather seats are fantastic.",
        "Dealership delivered car on time with flawless customer support and transparent pricing.",
        "Cruising at triple digits is effortless stability is rock solid and steering is precise.",
        "Best car in the compact SUV segment superb ground clearance and commanding road presence.",
        "Super quiet cabin NVH insulation is refined and highway mileage exceeded expectations."
    ]
    
    neutral_phrases = [
        "Average vehicle performance adequate for daily office commute but lacks highway punch.",
        "Fuel mileage is around 14 kmpl which is standard for a 1.2 liter petrol engine.",
        "The dashboard plastics are decent neither too premium nor terribly cheap.",
        "Suspension is slightly stiff over bumps but handles straight highway roads decently.",
        "Infotainment has basic features bluetooth works fine but screen brightness is average.",
        "Servicing cost was as quoted in the user manual no surprise charges but nothing special.",
        "Decent boot space for weekend luggage though three adults in the rear is a bit tight.",
        "Braking bite is acceptable pedal feel is progressive under normal city driving.",
        "Cabin acoustics are okay slight wind noise above 90 kmh but manageable with music.",
        "Styling is conventional blend in with traffic neither striking nor ugly."
    ]
    
    negative_phrases = [
        "Terrible lag in the AMT gearbox jerky gear shifts and constant hesitation when overtaking.",
        "Touchscreen freezes repeatedly reverse camera fails to display and Bluetooth disconnects.",
        "Shocking rattling noises from dashboard within two months and service center was unhelpful.",
        "Disastrous mileage barely gives 8 kmpl in city traffic company claims are completely false.",
        "Extremely harsh ride every road imperfection sends vibrations directly into the cabin spine.",
        "A/C cooling is ineffective takes over twenty minutes to chill cabin on sunny afternoons.",
        "Repeated battery drainage issues breakdown on highway stranded with family zero roadside assist.",
        "Poor fit and finish uneven panel gaps water ingress in tail lamp during monsoon rain.",
        "Steering feels dead vague centering and excessive body roll when cornering at speed.",
        "Worst customer service experience dealership kept car for five days without diagnosing error."
    ]
    
    reviews = []
    sentiments = []
    
    for _ in range(n_samples // 3):
        # Positive
        p_base = np.random.choice(positive_phrases)
        reviews.append(p_base)
        sentiments.append(2)
        
        # Neutral
        n_base = np.random.choice(neutral_phrases)
        reviews.append(n_base)
        sentiments.append(1)
        
        # Negative
        neg_base = np.random.choice(negative_phrases)
        reviews.append(neg_base)
        sentiments.append(0)
        
    df = pd.DataFrame({
        "review_id": [f"REV_{i+1:05d}" for i in range(len(reviews))],
        "review_text": reviews,
        "sentiment_label": sentiments,
        "sentiment_category": [SENTIMENT_MAP[s] for s in sentiments]
    })
    return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

# -------------------------------------------------------------
# Text Preprocessing & Vocabulary Tokenizer
# -------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def build_vocabulary(texts, max_vocab=1500):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    word_counts = {}
    for t in texts:
        for w in clean_text(t).split():
            word_counts[w] = word_counts.get(w, 0) + 1
            
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, _ in sorted_words[: max_vocab - 2]:
        vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=30):
    tokens = clean_text(text).split()
    indices = [vocab.get(w, vocab["<UNK>"]) for w in tokens][:max_len]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices

# -------------------------------------------------------------
# Bidirectional LSTM Classifier Architecture
# -------------------------------------------------------------
class VehicleReviewLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_classes=3, dropout=0.3):
        super(VehicleReviewLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, (hn, cn) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        hidden_cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        logits = self.fc(hidden_cat)
        return logits

def train_sentiment_lstm_pipeline(output_dir):
    print("=" * 80)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML")
    print("LAB STATEMENT 8: SENTIMENT ANALYSIS USING LSTM (VEHICLE FEEDBACK)")
    print("=" * 80)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Dataset Generation
    df = generate_vehicle_reviews_dataset(1200)
    csv_path = os.path.join(output_dir, "vehicle_reviews_sentiment.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} vehicle feedback reviews. Saved to: {csv_path}")
    print(df["sentiment_category"].value_counts())
    
    # Plot 1: Sentiment Distribution & Review Lengths
    df["word_count"] = df["review_text"].apply(lambda t: len(t.split()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.countplot(data=df, x="sentiment_category", ax=axes[0], palette="Set2")
    axes[0].set_title("Customer Review Sentiment Distribution")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Review Count")
    
    sns.histplot(data=df, x="word_count", hue="sentiment_category", ax=axes[1], kde=True, bins=15, palette="Set2")
    axes[1].set_title("Review Length Distribution (Word Count)")
    axes[1].set_xlabel("Words per Review")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_sentiment_class_distribution_and_lengths.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p1}")
    
    # 2. Tokenization & Sequence Encoding
    vocab = build_vocabulary(df["review_text"], max_vocab=1200)
    max_len = 30
    encoded_seqs = np.array([encode_text(t, vocab, max_len) for t in df["review_text"]], dtype=np.int64)
    labels = df["sentiment_label"].values.astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(encoded_seqs, labels, test_size=0.25, random_state=42, stratify=labels)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64, shuffle=False)
    
    # 3. Model Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VehicleReviewLSTM(vocab_size=len(vocab), embedding_dim=64, hidden_dim=64, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    epochs = 10
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print(f"\nTraining Bidirectional LSTM on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()
                
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
    axes[0].set_title("Bi-LSTM Cross-Entropy Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].plot(range(1, epochs + 1), history["train_acc"], "b-o", label="Train Accuracy (%)")
    axes[1].plot(range(1, epochs + 1), history["val_acc"], "g--s", label="Validation Accuracy (%)")
    axes[1].set_title(f"Sentiment Accuracy (Peak Val: {max(history['val_acc']):.1f}%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_lstm_training_loss_and_accuracy.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p2}")
    
    # 4. Confusion Matrix
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    ax.set_title(f"Sentiment Classification Confusion Matrix (Accuracy: {accuracy_score(all_targets, all_preds)*100:.1f}%)")
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("True Sentiment")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_confusion_matrix_sentiment.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p3}")
    
    # 5. Inference Testing on Unseen Automotive Reviews
    sample_tests = [
        "The electric vehicle acceleration is exhilarating and cabin comfort is sublime!",
        "Gear shifts are jerky and touchscreen constantly disconnects during driving.",
        "Average hatchback with adequate mileage for daily city commute."
    ]
    
    print("\n" + "=" * 70)
    print("INFERENCE VERIFICATION ON CUSTOM VEHICLE REVIEWS:")
    print("=" * 70)
    for text in sample_tests:
        seq = torch.tensor([encode_text(text, vocab, max_len)]).to(device)
        with torch.no_grad():
            logits = model(seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            pred_sent = SENTIMENT_MAP[pred_class]
            print(f"Review: \"{text}\"")
            print(f"--> Sentiment: {pred_sent.upper()} (Neg: {probs[0]*100:.1f}%, Neu: {probs[1]*100:.1f}%, Pos: {probs[2]*100:.1f}%)\n")
    print("=" * 70)

if __name__ == "__main__":
    out = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab8_sentiment_analysis_lstm"
    train_sentiment_lstm_pipeline(out)
