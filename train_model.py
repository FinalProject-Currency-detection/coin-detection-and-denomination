

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
IMG_HEIGHT = 240
IMG_WIDTH = 320

# Step 1: Image Preprocessing
def preprocess_images(data_dir, batch_size, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize pixels to [0, 1]
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # Training set

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')  # Validation set
    
    return train_generator, validation_generator

# Step 2: Data Augmentation
def augment_images(train_generator):
    datagen_augmented = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Rotate images by up to 30 degrees
        brightness_range=[0.8, 1.2],  # Adjust brightness
        horizontal_flip=True,  # Flip horizontally
        validation_split=0.2)  # Reuse same validation split
    
    augmented_train_generator = datagen_augmented.flow_from_directory(
        train_generator.directory,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=train_generator.batch_size,
        class_mode='categorical',
        subset='training')
    
    return augmented_train_generator

# Step 3: CNN Architecture
def build_cnn_model():
    model = Sequential([
        # Convolutional layers with ReLU and max-pooling layers
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Global average pooling instead of flattening
        GlobalAveragePooling2D(),
        
        # Fully connected layers
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')  # 5 classes for coin denominations
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Training the CNN
def train_cnn_model(model, train_gen, val_gen, epochs=30):
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    
    return history

# Step 5: Evaluation
def evaluate_model(model, val_gen):
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f'Validation Accuracy: {val_accuracy*100:.2f}%')
    
    # Predict and calculate confusion matrix
    val_gen.reset()
    predictions = model.predict(val_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    # Classification Report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(f'Classification Report:\n{report}')
    
    return conf_matrix

# Plot training & validation accuracy/loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Main Function to Run the Full Program
def main():
    data_directory = r"D:\Final project\Indian Coins Image Dataset"  # Replace with your dataset directory
    batch_size = 32
    epochs = 30
    
    # Preprocess and augment data
    train_gen, val_gen = preprocess_images(data_directory, batch_size)
    augmented_train_gen = augment_images(train_gen)
    
    # Build and train model
    model = build_cnn_model()
    history = train_cnn_model(model, augmented_train_gen, val_gen, epochs)
    
    # Evaluate model
    evaluate_model(model, val_gen)
    
    # Plot accuracy and loss
    plot_training_history(history)

if __name__ == "__main__":
    main()
