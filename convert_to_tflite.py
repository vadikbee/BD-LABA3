# convert_to_tflite.py
import tensorflow as tf

print("Загрузка обученной Keras модели 'butterfly_model.h5'...")
try:
    model = tf.keras.models.load_model('butterfly_model.h5')
except IOError:
    print("Ошибка: файл 'butterfly_model.h5' не найден.")
    print("Пожалуйста, сначала запустите скрипт 'train_butterfly.py'.")
    exit()

print("Конвертация модели в TensorFlow Lite формат...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open('butterfly_model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("\nГотово! Модель сохранена как 'butterfly_model_quant.tflite'.")