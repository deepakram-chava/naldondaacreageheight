import os
import sys
import json
import time
import random
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

print("=" * 65, flush=True)
print("PADDY DISEASE CLASSIFICATION: EFFICIENTNETB0 TRAINING PIPELINE", flush=True)
print(f"TensorFlow Version: {tf.__version__}", flush=True)
print("=" * 65, flush=True)

# 1. Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 10

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15

SPLIT_DIR = Path("split_dataset")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "paddy_disease_efficientnetb0.keras"
BEST_STAGE1_PATH = MODEL_DIR / "stage1_best_model.keras"
BEST_STAGE2_PATH = MODEL_DIR / "stage2_best_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# 2. Dataset Verification & Split Table
print("\n" + "=" * 65, flush=True)
print("STEP 6: DATASET SPLIT & ZERO-LEAKAGE VERIFICATION", flush=True)
print("=" * 65, flush=True)

def get_hashes_for_split(split_path):
    hashes = set()
    for p in split_path.glob("*/*.jpg"):
        with open(p, "rb") as f:
            hashes.add(hashlib.md5(f.read()).hexdigest())
    return hashes

train_hashes = get_hashes_for_split(TRAIN_DIR)
val_hashes = get_hashes_for_split(VAL_DIR)
test_hashes = get_hashes_for_split(TEST_DIR)

leak_tv = len(train_hashes & val_hashes)
leak_tt = len(train_hashes & test_hashes)
leak_vt = len(val_hashes & test_hashes)

print(f"Duplicate Leakage Check (MD5 Overlaps):", flush=True)
print(f"  • Train & Validation: {leak_tv} (Expected: 0)", flush=True)
print(f"  • Train & Test:       {leak_tt} (Expected: 0)", flush=True)
print(f"  • Val & Test:         {leak_vt} (Expected: 0)", flush=True)
assert leak_tv == 0 and leak_tt == 0 and leak_vt == 0, "CRITICAL ERROR: Data leakage detected between splits!"
print("  Zero duplicate leakage confirmed between Train, Validation, and Test splits.", flush=True)

# 3. Data Loading
print("\n[DATA LOADING]", flush=True)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print(f"Detected {len(class_names)} classes:", flush=True)
for i, c in enumerate(class_names):
    print(f"  [{i}]: {c}", flush=True)

# Save class_names.json immediately
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)
print(f"Saved class mapping to {CLASS_NAMES_PATH}", flush=True)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 4. Dataset Split Distribution & Class Weights strictly on TRAINING split
print("\n" + "=" * 65, flush=True)
print("DATASET SPLIT DISTRIBUTION TABLE")
print("=" * 65, flush=True)
print(f"{'Class':<28} {'Train':<7} {'Val':<7} {'Test':<7} {'Total':<7}", flush=True)
print("-" * 65, flush=True)

train_counts = {}
total_train = total_val = total_test = 0
for i, c in enumerate(class_names):
    n_tr = len(list((TRAIN_DIR / c).glob("*.jpg")))
    n_va = len(list((VAL_DIR / c).glob("*.jpg")))
    n_te = len(list((TEST_DIR / c).glob("*.jpg")))
    n_tot = n_tr + n_va + n_te
    train_counts[i] = n_tr
    total_train += n_tr
    total_val += n_va
    total_test += n_te
    print(f"{c:<28} {n_tr:<7} {n_va:<7} {n_te:<7} {n_tot:<7}", flush=True)

total_all = total_train + total_val + total_test
print("-" * 65, flush=True)
print(f"{'TOTAL':<28} {total_train:<7} {total_val:<7} {total_test:<7} {total_all:<7}", flush=True)
print(f"{'PERCENTAGE':<28} {total_train/total_all*100:5.2f}%  {total_val/total_all*100:5.2f}%  {total_test/total_all*100:5.2f}%  100.00%", flush=True)
print("=" * 65, flush=True)

print("\n[CLASS WEIGHTS CALCULATION (TRAINING SPLIT)]", flush=True)
class_weights = {}
for i in range(NUM_CLASSES):
    cnt = train_counts[i]
    class_weights[i] = float(total_train / (NUM_CLASSES * cnt))
    print(f"  Class {i} ({class_names[i]:<25}): {cnt:>5} samples -> weight: {class_weights[i]:.4f}", flush=True)

# 4. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# 5. Build Model Architecture
print("\n[BUILDING EFFICIENTNETB0 ARCHITECTURE]", flush=True)
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.BatchNormalization(name="batch_norm")(x)
x = layers.Dropout(0.4, name="top_dropout_1")(x)
x = layers.Dense(128, activation="relu", name="dense_128")(x)
x = layers.Dropout(0.3, name="top_dropout_2")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="paddy_disease_output")(x)

model = models.Model(inputs=inputs, outputs=outputs, name="PaddyDiseaseEfficientNetB0")

# 6. Stage 1 Compilation & Callbacks
print("\n[STAGE 1: TRAINING CLASSIFICATION HEAD (FROZEN BACKBONE)]", flush=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb_stage1 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint(BEST_STAGE1_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

t0 = time.time()
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weights,
    callbacks=cb_stage1,
    verbose=1
)
stage1_time = time.time() - t0
print(f"Stage 1 completed in {stage1_time:.1f}s", flush=True)

# 7. Stage 2 Fine-Tuning
print("\n[STAGE 2: FINE-TUNING FINAL 30% OF BACKBONE LAYERS]", flush=True)
# Load best Stage 1 weights
if BEST_STAGE1_PATH.exists():
    print(f"Loading best weights from {BEST_STAGE1_PATH}...", flush=True)
    model = tf.keras.models.load_model(BEST_STAGE1_PATH)

# Retrieve base_model inside model
effnet_layer = None
for layer in model.layers:
    if "efficientnetb0" in layer.name.lower():
        effnet_layer = layer
        break

if effnet_layer:
    effnet_layer.trainable = True
    fine_tune_from = int(len(effnet_layer.layers) * 0.70)
    for layer in effnet_layer.layers[:fine_tune_from]:
        layer.trainable = False
    trainable_count = sum(1 for l in effnet_layer.layers if l.trainable)
    print(f"Unfrozen {trainable_count} / {len(effnet_layer.layers)} EfficientNetB0 layers (top 30%).", flush=True)
else:
    print("Warning: Base model layer not found by name, fine-tuning top layers of model directly.", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb_stage2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint(BEST_STAGE2_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

t1 = time.time()
history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights,
    callbacks=cb_stage2,
    verbose=1
)
stage2_time = time.time() - t1
print(f"Stage 2 completed in {stage2_time:.1f}s", flush=True)

# Determine overall best model based purely on validation performance
val_loss_stage1 = min(history_stage1.history.get("val_loss", [float("inf")]))
val_loss_stage2 = min(history_stage2.history.get("val_loss", [float("inf")]))
print(f"\nStage 1 Best Validation Loss: {val_loss_stage1:.4f}", flush=True)
print(f"Stage 2 Best Validation Loss: {val_loss_stage2:.4f}", flush=True)

if val_loss_stage2 <= val_loss_stage1 and BEST_STAGE2_PATH.exists():
    print(f"Selecting Stage 2 model as overall best ({val_loss_stage2:.4f} <= {val_loss_stage1:.4f}). Loading from {BEST_STAGE2_PATH}...", flush=True)
    best_model = tf.keras.models.load_model(BEST_STAGE2_PATH)
elif BEST_STAGE1_PATH.exists():
    print(f"Selecting Stage 1 model as overall best ({val_loss_stage1:.4f} < {val_loss_stage2:.4f}). Loading from {BEST_STAGE1_PATH}...", flush=True)
    best_model = tf.keras.models.load_model(BEST_STAGE1_PATH)
else:
    best_model = model

# Save Final Production Model
best_model.save(FINAL_MODEL_PATH)
print(f"\nFinal Production Model saved to: {FINAL_MODEL_PATH}", flush=True)

# 8. Plot and Save Training Curves
print("\n[GENERATING TRAINING CURVES]", flush=True)
acc1 = history_stage1.history.get("accuracy", [])
val_acc1 = history_stage1.history.get("val_accuracy", [])
loss1 = history_stage1.history.get("loss", [])
val_loss1 = history_stage1.history.get("val_loss", [])

acc2 = history_stage2.history.get("accuracy", [])
val_acc2 = history_stage2.history.get("val_accuracy", [])
loss2 = history_stage2.history.get("loss", [])
val_loss2 = history_stage2.history.get("val_loss", [])

total_acc = acc1 + acc2
total_val_acc = val_acc1 + val_acc2
total_loss = loss1 + loss2
total_val_loss = val_loss1 + val_loss2
epochs_range = range(1, len(total_acc) + 1)
split_epoch = len(acc1)

# Accuracy Curve
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, total_acc, label="Training Accuracy", color="#3b82f6", linewidth=2)
plt.plot(epochs_range, total_val_acc, label="Validation Accuracy", color="#10b981", linewidth=2)
if split_epoch > 0 and len(acc2) > 0:
    plt.axvline(x=split_epoch, color="#f59e0b", linestyle="--", label="Start Fine-Tuning")
plt.title("Paddy Disease Classification — Accuracy Curves", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, total_loss, label="Training Loss", color="#ef4444", linewidth=2)
plt.plot(epochs_range, total_val_loss, label="Validation Loss", color="#8b5cf6", linewidth=2)
if split_epoch > 0 and len(loss2) > 0:
    plt.axvline(x=split_epoch, color="#f59e0b", linestyle="--", label="Start Fine-Tuning")
plt.title("Paddy Disease Classification — Loss Curves", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Categorical Crossentropy Loss")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# 9. Final Test Evaluation (Strictly ONCE on unseen test dataset)
print("\n" + "=" * 65, flush=True)
print("STEP 9: FINAL UNSEEN TEST EVALUATION", flush=True)
print("=" * 65, flush=True)
test_results = best_model.evaluate(test_ds, verbose=1)
test_loss, test_accuracy = test_results[0], test_results[1]
print(f"Test Loss     : {test_loss:.4f}", flush=True)
print(f"Test Accuracy : {test_accuracy * 100:.2f}%", flush=True)

# Generate predictions for all test samples
print("\nComputing predictions for classification report and confusion matrix...", flush=True)
y_pred_probs = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true_list = []
for _, labels in test_ds:
    y_true_list.append(np.argmax(labels.numpy(), axis=1))
y_true = np.concatenate(y_true_list, axis=0)

# Classification Report with zero_division=0
clf_report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
clf_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
print("\nCLASSIFICATION REPORT (UNSEEN TEST SET):", flush=True)
print(clf_report_str, flush=True)

with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(f"PADDY DISEASE EFFICIENTNETB0 — UNSEEN TEST EVALUATION\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    f.write(clf_report_str)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
plt.title("Paddy Disease Classification — Unseen Test Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# Identify Most Confused Disease Pairs
confused_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_names[i], class_names[j], int(cm[i, j])))

confused_pairs.sort(key=lambda x: x[2], reverse=True)
print("\nMOST CONFUSED DISEASE PAIRS:", flush=True)
for true_c, pred_c, count in confused_pairs[:5]:
    print(f"  • True '{true_c}' misclassified as '{pred_c}': {count} instances", flush=True)

# Per-class accuracy
print("\nPER-CLASS ACCURACY:", flush=True)
per_class_acc = {}
for i, c in enumerate(class_names):
    tot = int(cm[i].sum())
    cor = int(cm[i, i])
    acc = (cor / tot * 100) if tot > 0 else 0.0
    per_class_acc[c] = round(acc, 2)
    print(f"  {c:<30}: {acc:6.2f}% ({cor}/{tot})", flush=True)

# 10. Model Reloading & Verification
print("\n" + "=" * 65, flush=True)
print("MODEL PERSISTENCE & MAPPING VERIFICATION", flush=True)
print("=" * 65, flush=True)
print("Reloading saved .keras model from disk...", flush=True)
reloaded_model = tf.keras.models.load_model(FINAL_MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    reloaded_classes = json.load(f)

assert reloaded_classes == class_names, "CRITICAL: Reloaded class names do not match!"

# Test on 5 sample images
sample_batch = next(iter(test_ds.take(1)))
orig_preds = best_model.predict(sample_batch[0][:5], verbose=0)
reloaded_preds = reloaded_model.predict(sample_batch[0][:5], verbose=0)
diff = np.max(np.abs(orig_preds - reloaded_preds))
print(f"Maximum difference between original & reloaded model predictions: {diff:.6e}", flush=True)
if diff < 1e-4:
    print("✅ VERIFICATION PASSED: Reloaded model produces identical predictions.", flush=True)
else:
    print("❌ ERROR: Prediction discrepancy in reloaded model!", flush=True)

# Save Comprehensive Training Metadata
training_metadata = {
    "model_name": "paddy_disease_efficientnetb0",
    "architecture": "EfficientNetB0 Transfer Learning",
    "framework": f"TensorFlow {tf.__version__}",
    "python_version": "3.11",
    "random_seed": RANDOM_SEED,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
    "dataset_split": {
        "train": total_train,
        "validation": total_val,
        "test": total_test
    },
    "class_weights": {class_names[k]: round(v, 4) for k, v in class_weights.items()},
    "stage1_epochs_trained": len(acc1),
    "stage2_epochs_trained": len(acc2),
    "test_evaluation": {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "macro_precision": round(clf_report_dict["macro avg"]["precision"], 4),
        "macro_recall": round(clf_report_dict["macro avg"]["recall"], 4),
        "macro_f1": round(clf_report_dict["macro avg"]["f1-score"], 4),
        "weighted_precision": round(clf_report_dict["weighted avg"]["precision"], 4),
        "weighted_recall": round(clf_report_dict["weighted avg"]["recall"], 4),
        "weighted_f1": round(clf_report_dict["weighted avg"]["f1-score"], 4),
        "per_class_accuracy": per_class_acc,
        "most_confused_pairs": [(p[0], p[1], p[2]) for p in confused_pairs[:5]]
    }
}

with open(RESULTS_DIR / "training_metadata.json", "w") as f:
    json.dump(training_metadata, f, indent=2)

print(f"\nMetadata saved to {RESULTS_DIR / 'training_metadata.json'}", flush=True)
print("=" * 65, flush=True)
print("PADDY DISEASE TRAINING & EVALUATION COMPLETED SUCCESSFULLY!", flush=True)
print("=" * 65, flush=True)
