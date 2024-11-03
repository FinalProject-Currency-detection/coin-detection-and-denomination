import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
NUM_CLASSES = 7  # Replace with actual number of classes
DATASET_DIR = "New-dataset"  # Update with your dataset path
MODEL_SAVE_PATH = "coin_classifier_resnet50.keras"

# Step 1: Data Preprocessing without Augmentation
def preprocess_data(dataset_dir, batch_size):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_gen, val_gen

# Step 2: Build ResNet Model with Transfer Learning
def build_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train Model with 35 Epochs and Early Stopping
def train_model(model, train_gen, val_gen, save_path, epochs=100):
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )
    return history

# Step 4: Model Evaluation
def evaluate_model(model, train_gen, val_gen):
    train_loss, train_acc = model.evaluate(train_gen)
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"Training Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Generate confusion matrix and classification report
    val_gen.reset()
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    conf_matrix = confusion_matrix(y_true, y_pred)
    class_labels = list(val_gen.class_indices.keys())
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)

# Step 5: Load and Test Model
def classify_image(model_path, image_path):
    model = load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Main Function
def main():
    train_gen, val_gen = preprocess_data(DATASET_DIR, BATCH_SIZE)
    model = build_resnet_model()
    history = train_model(model, train_gen, val_gen, MODEL_SAVE_PATH)
    evaluate_model(model, train_gen, val_gen)
    
    # Test the model with a new image
    image_path = "images.jpg"  # Update with your test image path
    class_labels = list(train_gen.class_indices.keys())
    predicted_class = classify_image(MODEL_SAVE_PATH, image_path)
    print(f"Predicted class for the input image: {class_labels[predicted_class]}")

if __name__ == "__main__":
    main()
