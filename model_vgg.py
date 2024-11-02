import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Image dimensions
IMG_HEIGHT = 240
IMG_WIDTH = 320
NUM_CLASSES = 7  # Adjust according to your dataset

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

# Step 2: Build VGG19 Model
def build_vgg19_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(NUM_CLASSES, activation='softmax')  # Adjust number of classes
    ])

    # Freeze the layers of VGG19
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Training the CNN
def train_cnn_model(model, train_gen, val_gen, epochs=30, save_path="best_model.keras"):
    # Use ModelCheckpoint to save the best model
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint]  # Save the best model during training
    )

    return history

# Step 4: Evaluate the model on both training and validation sets
def evaluate_model(model, train_gen, val_gen):
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_gen)
    print(f'Validation Accuracy: {val_accuracy*100:.2f}%')

    # Evaluate on training set
    train_loss, train_accuracy = model.evaluate(train_gen)
    print(f'Training Accuracy: {train_accuracy*100:.2f}%')

    # Predict and calculate confusion matrix for validation set
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

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

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

# Step 5: Load the saved model and classify an input image
def classify_image(model_path, image_path, class_labels):
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Scale pixels to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Return the predicted class label
    print(f"Predicted class: {class_labels[predicted_class]}")
    return class_labels[predicted_class]

# Main Function to Run the Full Program
def main():
    data_directory = r"D:\Final project\Coin Denomination\Indian Coins Image Dataset"  # Replace with your dataset directory
      # Replace with your dataset directory
    batch_size = 32
    epochs = 30
    save_path = "best_model.keras"

    # Preprocess data
    train_gen, val_gen = preprocess_images(data_directory, batch_size)

    # Build and train model
    model = build_vgg19_model()
    history = train_cnn_model(model, train_gen, val_gen, epochs, save_path=save_path)

    # Evaluate model on both training and validation sets
    evaluate_model(model, train_gen, val_gen)

    # Plot accuracy and loss
    plot_training_history(history)

    # Example: Load the saved model and classify an image
    class_labels = list(train_gen.class_indices.keys())  # Get class labels from training generator
    image_path = "images.jpg"  # Replace with your image path
    classify_image(save_path, image_path, class_labels)

if __name__ == "__main__":
    main()
