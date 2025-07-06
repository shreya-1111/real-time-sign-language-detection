# train_cnn_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# Path to dataset
DATASET_PATH = 'asl_alphabet_train'

# Image settings
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

print("📁 Preparing data generators...")
# Data generator with 80-20 train-val split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("✅ Data generators ready.")
print(f"🔤 Number of classes: {train_generator.num_classes}")
print("📊 Class indices:", train_generator.class_indices)

# CNN model
print("🧠 Building the CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile and train
print("⚙️ Compiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Starting training...")
model.fit(train_generator, validation_data=val_generator, epochs=10)
print("✅ Training complete.")

# Save the model and labels
print("💾 Saving the model to 'asl_cnn_model.h5'...")
model.save('asl_cnn_model.h5')

print("📝 Saving class labels to 'asl_labels.txt'...")
with open('asl_labels.txt', 'w') as f:
    for label in train_generator.class_indices:
        f.write(label + '\n')

print("🎉 Model and labels saved successfully!")
