import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

# Directories and paths
train_dir = r"C:\Users\liyas\Downloads\Image Dataset of Indian Coins\Image Dataset of Indian Coins\Indian Coins Image Dataset\Indian Coins Image Dataset"
model_path = 'final_trained_coin_model.keras'
class_labels_path = 'class_labels.json'

def load_class_labels():
    """Load class labels from a JSON file."""
    with open(class_labels_path, 'r') as f:
        return json.load(f)

# Check if the model and class labels exist
if os.path.exists(model_path) and os.path.exists(class_labels_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    # Load class labels
    class_labels = load_class_labels()
    print("Model and class labels loaded from disk.")
else:
    raise FileNotFoundError("Model or class labels not found.")

# Create an ImageDataGenerator for augmenting and loading images
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Load the training data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load the validation data
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Compile the model (important to recompile before training)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Updated callbacks
checkpoint = ModelCheckpoint('improved_coin_model_v4.keras', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save class labels to a JSON file
with open(class_labels_path, 'w') as f:
    json.dump(train_data.class_indices, f)

print("Class labels saved to 'class_labels.json'")

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f'Validation Accuracy: {accuracy:.2f}')
