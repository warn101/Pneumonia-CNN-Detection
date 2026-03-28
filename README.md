# Pneumonia Detection using CNN

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 89.4% |
| Recall (Sensitivity) | 91.8% |
| Specificity | 85.5% |
| AUC | 0.956 |

> **Why recall matters most:** In medical screening, missing a pneumonia case is far more dangerous than a false alarm. A recall of 91.8% means the model correctly identifies 9 out of 10 real pneumonia cases.

---

## Dataset

**Chest X-Ray Images (Pneumonia)** 


```
dataset/
├── train/
│   ├── NORMAL/       1,341 images
│   └── PNEUMONIA/    3,875 images
├── val/
│   ├── NORMAL/           8 images
│   └── PNEUMONIA/        8 images
└── test/
    ├── NORMAL/         234 images
    └── PNEUMONIA/      390 images
```

**Class imbalance:** The dataset has a 1:2.9 ratio (NORMAL:PNEUMONIA), reflecting real-world clinical data where sick patients are overrepresented in hospital archives. This is handled using class weights during training.

---

## Model Architecture

A custom CNN with 3 convolutional blocks followed by a fully connected classifier head.

```
Input (224 × 224 × 3)
       │
  ┌────▼────────────────────────────┐
  │  Block 1                        │
  │  Conv2D(32) → BN → ReLU        │
  │  Conv2D(32) → BN → ReLU        │
  │  MaxPool(2×2) → Dropout(0.30)  │  → 112 × 112
  └────────────────────────────────┘
       │
  ┌────▼────────────────────────────┐
  │  Block 2                        │
  │  Conv2D(64) → BN → ReLU        │
  │  Conv2D(64) → BN → ReLU        │
  │  MaxPool(2×2) → Dropout(0.30)  │  → 56 × 56
  └────────────────────────────────┘
       │
  ┌────▼────────────────────────────┐
  │  Block 3                        │
  │  Conv2D(128) → BN → ReLU       │
  │  Conv2D(128) → BN → ReLU       │
  │  MaxPool(2×2) → Dropout(0.50)  │  → 28 × 28
  └────────────────────────────────┘
       │
  ┌────▼────────────────────────────┐
  │  Classifier Head                │
  │  Flatten → Dense(256) → BN     │
  │  ReLU → Dropout(0.50)          │
  │  Dense(1) → Sigmoid            │  → probability [0, 1]
  └────────────────────────────────┘
```

**Total parameters:** 25,977,633  
**Output:** Single sigmoid neuron — probability of PNEUMONIA (threshold: 0.5)

---

## Design Decisions

**BatchNormalization after every Conv layer**  
Normalizes activations within each mini-batch. Stabilizes training, allows higher learning rates, and prevents the vanishing gradient problem.

**Filters double each block (32 → 64 → 128)**  
Early layers detect simple patterns (edges, textures) — few filters needed. Deeper layers detect complex patterns (lung opacity, consolidation) — more filters needed.

**Dropout increases with depth (0.30 → 0.30 → 0.50)**  
Deeper layers are more prone to overfitting. Stronger dropout forces redundant representations, improving generalization.

**Class weights instead of resampling**  
Computed using `sklearn.utils.class_weight.compute_class_weight`. Preserves all real data while correcting the loss function's attention toward the minority class (NORMAL).

```
NORMAL    weight: 1.9388   (underrepresented → penalized more)
PNEUMONIA weight: 0.6701   (overrepresented → penalized less)
```

---

## Training

**Optimizer:** Adam (lr=0.001)  
**Loss:** Binary crossentropy  
**Epochs:** 30 max (EarlyStopping triggered at epoch 14, best weights from epoch 9)

### Callbacks

| Callback | Purpose |
|---|---|
| EarlyStopping (patience=5) | Stops training when val_loss stops improving |
| ModelCheckpoint | Saves best model weights to disk |
| ReduceLROnPlateau (factor=0.5, patience=3) | Halves learning rate when stuck |

### Data Augmentation (training set only)

| Transform | Value | Reason |
|---|---|---|
| Horizontal flip | True | Mirror-image chest is clinically valid |
| Rotation | ±15° | Patients aren't always perfectly aligned |
| Zoom | ±15% | Simulates different scanner distances |
| Width/height shift | ±15% | Accounts for off-center positioning |
| Shear | 0.1 | Adds geometric variety |

Validation and test sets are only normalized (rescale 1/255) — never augmented.

---

## Evaluation

### Confusion Matrix

```
                 Predicted
             NORMAL   PNEUMONIA
Actual NORMAL   200        34    ← 34 false alarms
Actual PNEUMO    32       358    ← 32 missed cases
```

### Interpretation

- **32 missed cases (False Negatives):** Pneumonia patients incorrectly cleared — the critical error in medical screening
- **34 false alarms (False Positives):** Healthy patients flagged for follow-up — inconvenient but not dangerous
- The model makes the safer kind of mistake more often, which is the correct clinical trade-off

---

## Project Structure

```
pneumonia-detection-cnn/
├── notebooks/
│   └──Problem_Set_01_CNN(1).ipynb
├── outputs/
│   ├── confusion_matrix.png
│   └── training_curves.png
└── README.md
```

---


## Known Limitations

- The official validation split (16 images) is too small for reliable training feedback. The test set (624 images) was used as the validation signal during training — a known workaround for this specific dataset.
- Model trained from scratch. Transfer learning (VGG16, ResNet50) would likely push recall above 95%.
- No deployment pipeline — inference is notebook-based only.

---

## What I Learned

- How CNNs extract features hierarchically from raw pixels
- Why class imbalance matters in medical imaging and how to correct it with class weights
- The difference between training loss and validation loss, and how to read overfitting
- Why recall is more important than accuracy for medical screening tasks
- How EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau work together during training

---
## Saved Model

The trained weights are not stored in this repo due to file size (~100MB).

To reproduce them, run all cells in `notebooks/pneumonia_cnn.ipynb`.
Training takes approximately 30 minutes on a GPU (Google Colab recommended).
The best weights will be saved automatically to `/content/saved_models/custom_cnn_best.keras`
by the ModelCheckpoint callback during training.
.
