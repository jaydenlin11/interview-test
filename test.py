# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import cv2

# --- Define Paths ---
DATASET_DIR = "correlation_assignment/images"  # Update with your dataset directory
CSV_FILE = "correlation_assignment/responses.csv"  # Update with your CSV file path

# Verify paths exist
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory {DATASET_DIR} does not exist.")
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file {CSV_FILE} does not exist.")

# --- Load Dataset ---
# Read CSV file with correlations
df = pd.read_csv(CSV_FILE)

# Add image paths to DataFrame
df['image_path'] = df['id'].apply(lambda x: os.path.join(DATASET_DIR, f"{x}.png"))

# Verify all images exist
df = df[df['image_path'].apply(os.path.exists)]
if len(df) == 0:
    raise ValueError("No valid images found in the dataset directory.")

# --- Split Data ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Data Generator ---
def create_data_generator(df, batch_size=32, target_size=(128, 128)):
    
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='corr',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',  # For regression
        shuffle=True
    )
    return generator

# Create generators
batch_size = 32
train_generator = create_data_generator(train_df, batch_size)
test_generator = create_data_generator(test_df, batch_size)

# --- Define Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

# --- Compile Model ---
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# --- Model Summary ---
model.summary()

# --- Train Model ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=test_generator,
    validation_steps=len(test_df) // batch_size,
    epochs=50,
    callbacks=[early_stop]
)

# --- Plot Training History ---
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('training_plot.png')

# --- Evaluate Model ---
loss, mae = model.evaluate(test_generator, steps=len(test_df) // batch_size)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# --- Save Model ---
model.save('scatter_correlation_model.h5')

# --- Predict Correlation ---
# def predict_correlation(image_path, model, target_size=(128, 128)):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image {image_path} does not exist.")
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Failed to load image {image_path}.")
#     img_resized = cv2.resize(img, target_size) / 255.0
#     img_expanded = np.expand_dims(img_resized, axis=0)
#     prediction = model.predict(img_expanded)
#     print(f"Predicted Correlation: {prediction[0][0]:.4f}")

# --- Test Prediction ---
# test_image_path = "path_to_test_image.png"  # Update with a test image path
# try:
#     predict_correlation(test_image_path, model)
# except Exception as e:
#     print(f"Error during prediction: {e}")