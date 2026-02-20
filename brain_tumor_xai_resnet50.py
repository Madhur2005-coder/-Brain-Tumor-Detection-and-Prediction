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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image

from sklearn.metrics import confusion_matrix, classification_report
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


# =========================
# configuration
# =========================
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

MODEL_PATH = "best_resnet50_brain_tumor_model.h5"
GRADCAM_OUTPUT = "gradcam_overlay.png"

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]


# ======================================================
# 1) data_preprocessing section
# ======================================================
print("\n[1/5] data_preprocessing section")

# Data augmentation for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,  # 0-1 normalization
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# For validation/test: only preprocessing (normalization via preprocess_input)
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0
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
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
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
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)


# ======================================================
# 4) evaluation section
# ======================================================
print("\n[4/5] evaluation section")

# Load best model if saved
if os.path.exists(MODEL_PATH):
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
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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
    arr = arr / 255.0
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
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
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


# Auto-select last conv layer from ResNet50 backbone
last_conv_layer_name = "conv5_block3_out"

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
