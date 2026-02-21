"""
Explainable AI Based Brain Tumor Detection and Prediction
=========================================================
Complete end-to-end TensorFlow/Keras pipeline for 4-class brain tumor MRI classification
using transfer learning (ResNet50) + Grad-CAM explainability.

Designed to run in Google Colab (GPU supported).

Dataset structure expected:
    dataset/
        train/
            glioma/
            meningioma/
            pituitary/
            no_tumor/
        test/
            glioma/
            meningioma/
            pituitary/
            no_tumor/
"""

# =========================
# imports
# =========================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras import mixed_precision

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import cv2


# =========================
# reproducibility + GPU check
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {gpus}")
else:
    print("No GPU found. Running on CPU.")

# Enable mixed precision on GPU for faster Colab training.
if gpus:
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled: mixed_float16")


# =========================
# configuration
# =========================
DATASET_DIR = "dataset"
# Support both common folder conventions: train/test and Training/Testing.
if os.path.isdir(os.path.join(DATASET_DIR, "train")) and os.path.isdir(os.path.join(DATASET_DIR, "test")):
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
elif os.path.isdir(os.path.join(DATASET_DIR, "Training")) and os.path.isdir(os.path.join(DATASET_DIR, "Testing")):
    TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
    TEST_DIR = os.path.join(DATASET_DIR, "Testing")
else:
    raise FileNotFoundError(
        "Could not find dataset folders. Expected either dataset/train+dataset/test or dataset/Training+dataset/Testing"
    )

if os.path.abspath(TRAIN_DIR) == os.path.abspath(TEST_DIR):
    raise ValueError("TRAIN_DIR and TEST_DIR point to the same path. Use a separate held-out test set.")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 20
FINE_TUNE_EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
UNFREEZE_LAYERS = 50

MODEL_PATH = "best_resnet50_brain_tumor_model.keras"
GRADCAM_OUTPUT = "gradcam_overlay.png"

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]


# ======================================================
# 1) data_preprocessing section
# ======================================================
print("\n[1/5] data_preprocessing section")

# Data augmentation for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.85, 1.15),
    horizontal_flip=True
)

# For validation/test: only preprocessing
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Class index mapping
print("Class mapping:", train_generator.class_indices)

# Class weights help improve minority/underperforming class recall.
class_ids = train_generator.classes
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_ids),
    y=class_ids
)
class_weights = {int(class_id): float(weight) for class_id, weight in zip(np.unique(class_ids), class_weights_values)}
print("Class weights:", class_weights)


# ======================================================
# 2) model_building section
# ======================================================
print("\n[2/5] model_building section")

# Build transfer learning model with ResNet50 base
input_tensor = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=input_tensor
)

# Freeze base layers initially
base_model.trainable = False

# Custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
)

model.summary()


# ======================================================
# 3) training section
# ======================================================
print("\n[3/5] training section")

checkpoint_cb = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=False,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history_stage1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    class_weight=class_weights
)

# Fine-tuning stage: unfreeze top layers of ResNet50 for better feature adaptation.
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

print(f"Fine-tuning top {UNFREEZE_LAYERS} layers of ResNet50")

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
)

history_stage2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_stage1.epoch[-1] + 1,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    class_weight=class_weights
)

# Merge histories for unified plots
history = {}
for key in history_stage1.history:
    history[key] = history_stage1.history[key] + history_stage2.history.get(key, [])


# ======================================================
# 4) evaluation section
# ======================================================
print("\n[4/5] evaluation section")

# Always load checkpoint-selected best model for reproducible evaluation
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Best checkpoint not found at {MODEL_PATH}. Training may have failed.")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded best model from {MODEL_PATH}")

# Evaluate accuracy on test set
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions for confusion matrix + classification report
test_generator.reset()
y_prob = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(test_generator.class_indices.keys()),
    yticklabels=list(test_generator.class_indices.keys())
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Classification report
class_labels_sorted = [k for k, _ in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels_sorted))

# Plot training history
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epoch_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epoch_range, acc, label='Training Accuracy')
plt.plot(epoch_range, val_acc, label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epoch_range, loss, label='Training Loss')
plt.plot(epoch_range, val_loss, label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# ======================================================
# 5) grad_cam section
# ======================================================
print("\n[5/5] grad_cam section")


def get_img_array(img_path, target_size):
    """Load and preprocess an image for ResNet50 input."""
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def make_gradcam_heatmap(img_array, model_obj, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image array."""
    grad_model = tf.keras.models.Model(
        [model_obj.inputs],
        [model_obj.get_layer(last_conv_layer_name).output, model_obj.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]).numpy())
        else:
            pred_index = int(pred_index)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="gradcam_overlay.png", alpha=0.4):
    """Overlay heatmap on original image and save the result."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Grad-CAM overlay saved at: {cam_path}")


# Auto-discover the last Conv2D layer for Grad-CAM robustness
last_conv_layer_name = None
for lyr in reversed(model.layers):
    if isinstance(lyr, tf.keras.layers.Conv2D):
        last_conv_layer_name = lyr.name
        break
if last_conv_layer_name is None:
    raise ValueError("Could not find a Conv2D layer for Grad-CAM.")
print("Using Grad-CAM layer:", last_conv_layer_name)

# Example prediction + Grad-CAM on one sample image from test set
sample_image_path = None
for class_name in CLASS_NAMES:
    class_dir = os.path.join(TEST_DIR, class_name)
    if os.path.isdir(class_dir):
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            sample_image_path = os.path.join(class_dir, files[0])
            break

if sample_image_path is None:
    raise FileNotFoundError(
        "No image found in test class folders. Please verify dataset/test/<class_name>/ contains images."
    )

print("Using sample image:", sample_image_path)

img_array = get_img_array(sample_image_path, IMG_SIZE)
preds = model.predict(img_array)
predicted_class_idx = np.argmax(preds[0])
predicted_class_name = class_labels_sorted[predicted_class_idx]
confidence = preds[0][predicted_class_idx]

print(f"Predicted Class: {predicted_class_name} | Confidence: {confidence:.4f}")

heatmap = make_gradcam_heatmap(
    img_array=img_array,
    model_obj=model,
    last_conv_layer_name=last_conv_layer_name,
    pred_index=predicted_class_idx
)
save_and_display_gradcam(sample_image_path, heatmap, cam_path=GRADCAM_OUTPUT, alpha=0.4)

print("\nPipeline completed successfully.")
