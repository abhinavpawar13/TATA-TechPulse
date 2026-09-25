[Uploading README.md…]()
# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 8: Sentiment Analysis using LSTM

**Curriculum Unit:** Unit 4 – AI fundamentals & application development (RNN and LSTM overview, Text data preprocessing, Sentiment classification)  
**Lab Statement 8:** *Analyze vehicle feedback using LSTM-based sentiment classification.*  
**Domain Focus:** Automotive Voice of Customer (VoC), Quality Engineering & Dealership Service Sentiment Mining  

---

## 📌 1. Overview & Objectives

Automotive manufacturers collect thousands of unstructured customer reviews from social channels, dealership feedback portals, and service surveys. Classifying sentiment automatically across domains like cabin NVH, infotainment reliability, engine performance, and dealership service enables proactive quality engineering and faster defect resolution.

This lab implements an end-to-end Recurrent Deep Learning pipeline using a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network in **PyTorch**.

### Key Learning Outcomes:
1. **Natural Language Processing (NLP) Preprocessing**: Clean, tokenize, build vocabulary mappings, and pad review sequences to uniform tensor lengths.
2. **Word Embedding Representation**: Learn low-dimensional continuous vector embeddings ($d=64$) preserving semantic relationships.
3. **Bidirectional LSTM Architecture**: Process textual sequence context from both forward and backward temporal directions, capturing nuanced dependencies.
4. **Classification & Multi-Class Inference**: Classify reviews into **Positive**, **Neutral**, and **Negative** sentiment classes with softmax probability scores.
5. **Model Evaluation**: Generate epoch loss curves, confusion matrices, and test inference on arbitrary vehicle reviews.

---

## 📁 2. Project Directory Structure

```text
lab8_sentiment_analysis_lstm/
│
├── README.md                                      # Lab manual & technical documentation
├── vehicle_reviews_sentiment.csv                  # Automotive customer review dataset (1,200 records)
├── sentiment_analysis_lstm.py                     # Complete PyTorch Bi-LSTM training & evaluation pipeline
├── lab8_sentiment_analysis_lstm.ipynb             # Interactive Jupyter Notebook
├── requirements.txt                               # Minimal dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_sentiment_class_distribution_and_lengths.png  # Review sentiment count & length histograms
    ├── 02_lstm_training_loss_and_accuracy.png     # Epoch-by-epoch loss and accuracy curves
    └── 03_confusion_matrix_sentiment.png          # 3-class sentiment confusion matrix
```

---

## 🔬 3. Bi-LSTM Architecture Specifications

```text
Input Tokens: [w_1, w_2, ..., w_30] (Batch Size x 30)
 │
 ├── Embedding Layer: Vocab Size (1,200) -> Embedding Dim (64)
 ├── Bidirectional LSTM (2 Layers, Hidden Dim: 64, Dropout: 0.3)
 │     ├── Forward LSTM:  h_t_fwd  (64)
 │     └── Backward LSTM: h_t_bwd  (64)
 ├── Concatenation: [h_final_fwd ; h_final_bwd] -> (128)
 ├── Dropout (p=0.30)
 ├── Dense (128 -> 64) + ReLU + Dropout(0.20)
 └── Output Linear (64 -> 3) -> Softmax [Negative, Neutral, Positive]
```

---

## 📊 4. Benchmark Performance & Inference

- **Training Epochs:** 10 Epochs (Adam, $\text{lr}=0.002$)
- **Peak Validation Accuracy:** **100.0%**
- **Test Sentiment Inferences:**
  - *"The electric vehicle acceleration is exhilarating and cabin comfort is sublime!"* $\to$ **POSITIVE (100.0%)**
  - *"Gear shifts are jerky and touchscreen constantly disconnects during driving."* $\to$ **NEGATIVE (99.9%)**
  - *"Average hatchback with adequate mileage for daily city commute."* $\to$ **NEUTRAL (100.0%)**

---

## 🚀 5. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Full Bi-LSTM Training Pipeline
```bash
python sentiment_analysis_lstm.py
```

### Step 3: Open Interactive Jupyter Notebook
```bash
jupyter notebook lab8_sentiment_analysis_lstm.ipynb
```
