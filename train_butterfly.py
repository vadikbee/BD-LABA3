# train_butterfly.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score

# --- Параметры ---
train_dir = 'train_structured'
valid_dir = 'valid_structured'
height = 128
width = 128
channels = 3
epochs = 50
batch_size = 32

# --- 1. Генераторы данных с аугментацией ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    fill_mode="nearest"
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# --- Сохранение словаря классов ---
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print(f"Словарь классов сохранен в 'class_indices.json'. Всего классов: {len(class_indices)}")
num_classes = len(class_indices)

# --- 2. Архитектура модели ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(height, width, channels)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Коллбэки ---
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('butterfly_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# --- 4. Запуск обучения ---
print("\nНачинаю обучение модели...")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, model_checkpoint]
)
print("\nОбучение завершено!")

# --- 5. Финальная оценка и сохранение результатов ---
print("\n--- Финальная оценка по заданию ---")
best_model = tf.keras.models.load_model('butterfly_model.h5')
print(f"Получение предсказаний для {valid_generator.samples} изображений...")
start_time = time.time()
predictions = best_model.predict(valid_generator)
end_time = time.time()
total_prediction_time = end_time - start_time

y_true = valid_generator.classes
y_pred = np.argmax(predictions, axis=1)
final_accuracy = accuracy_score(y_true, y_pred)

print("\n--- Итоговые результаты ---")
print(f"1. Значение метрики Accuracy_score: {final_accuracy:.4f} ({final_accuracy:.2%})")
print(f"2. Общее время предсказания:     {total_prediction_time:.2f} секунд")
print("-----------------------------\n")

results_for_streamlit = {
    "accuracy": final_accuracy,
    "prediction_time": total_prediction_time
}
with open('evaluation_results.json', 'w') as f:
    json.dump(results_for_streamlit, f)
print("Результаты оценки сохранены в 'evaluation_results.json'.")