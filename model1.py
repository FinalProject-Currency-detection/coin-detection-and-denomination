import os
import json
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

train_dir = r"C:\Users\liyas\Downloads\Image Dataset of Indian Coins\Image Dataset of Indian Coins\Indian Coins Image Dataset\Indian Coins Image Dataset"
model_path = 'final_trained_coin_model.keras'
class_labels_path = 'class_labels.json'

def load_class_labels():
    """Load class labels from a JSON file."""
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels

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

val_data = datagen.flow_from_directory(
    train_dir,                    
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Check if the model and class labels already exist
if os.path.exists(model_path) and os.path.exists(class_labels_path):
    # Load the model from the checkpoint file
    model = tf.keras.models.load_model(model_path)
    # Load class labels
    class_labels = load_class_labels()
    print("Model and class labels loaded from disk.")
else:
    # Load the training data
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),      
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    print(train_data.class_indices)

    # Load EfficientNetB0 model with ImageNet weights
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Unfreeze more layers of the base model for fine-tuning
    for layer in base_model.layers:
        layer.trainable = True  # Set all layers to trainable

    # Rebuild the model
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)  # Increased units
    x = Dropout(0.5)(x)  # Increased dropout to 50%
    output = layers.Dense(train_data.num_classes, activation='softmax')(x)

    # Compile the model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=50,
        callbacks=[
            ModelCheckpoint('improved_coin_model_v3.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
    )
    # Save class labels to a JSON file
    class_labels = train_data.class_indices
    with open(class_labels_path, 'w') as f:
        json.dump(class_labels, f)
    print("Class labels saved to 'class_labels.json'")

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f'Validation Accuracy: {accuracy:.2f}')

# Making predictions on the validation set
val_data.reset()  # Reset the generator to ensure predictions are aligned with true labels
predictions = model.predict(val_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Calculate TP, TN, FP, FN
TP = np.diagonal(conf_matrix)
TN = conf_matrix.sum() - (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - TP)
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP

# Calculate accuracy
accuracy_custom = (TP.sum() + TN.sum()) / (TP.sum() + TN.sum() + FP.sum() + FN.sum())
print(f'Custom Accuracy: {accuracy_custom:.2f}')

# Visualizing the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
