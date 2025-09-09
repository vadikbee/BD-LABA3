# lab3_streamlit.py
import streamlit as st
import numpy as np
import base64
import time
import os
import json
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ----- Конфигурация страницы и стили -----
st.set_page_config(
    page_title="Классификация бабочек",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Функции кеширования ---
@st.cache_data
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpg_file):
    try:
        bin_str = get_base64(jpg_file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Файл для фона '{jpg_file}' не найден.")

@st.cache_resource
def load_trained_model():
    try:
        interpreter = tflite.Interpreter(model_path='butterfly_model_quant.tflite')
        interpreter.allocate_tensors()
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        return interpreter, class_indices
    except Exception:
        return None, None

@st.cache_data
def load_data_info(train_path):
    try:
        if not os.path.exists(train_path):
            return None, None
        num_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        num_train_images = sum([len(files) for r, d, files in os.walk(train_path)])
        return num_classes, num_train_images
    except Exception:
        return None, None

@st.cache_data
def load_evaluation_results(file_path='evaluation_results.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# --- Основная часть приложения ---
set_background('sakura.jpg')
st.title('🦋 Классификация видов бабочек')
st.write('Лабораторная работа №3.')

interpreter, class_indices = load_trained_model()
height, width = 128, 128
st.sidebar.header("Панель управления")
app_mode = st.sidebar.selectbox("Выберите раздел:", ["Обзор данных", "Предсказание"])

if app_mode == "Обзор данных":
    st.header("1. Обзор данных")
    num_classes, num_train_images = load_data_info('train_structured')
    if num_classes is not None:
        st.metric("Количество классов (видов)", f"{num_classes} 🦋")

    st.divider()
    st.subheader("Результаты итоговой оценки модели")
    results = load_evaluation_results()
    if results:
        col1, col2 = st.columns(2)
        col1.metric("1. Точность на валидации (Accuracy)", f"{results.get('accuracy', 0):.2%}")
        col2.metric("2. Общее время предсказания", f"{results.get('prediction_time', 0):.2f} сек.")
    else:
        st.info("Результаты оценки не найдены. Сначала запустите `train_butterfly.py`.")

elif app_mode == "Предсказание":
    st.header("2. Предсказание по своему изображению")
    if interpreter is None or class_indices is None:
        st.error("Модель не загружена! Убедитесь, что файлы `butterfly_model_quant.tflite` и `class_indices.json` существуют.")
    else:
        uploaded_file = st.file_uploader("Загрузите изображение бабочки...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns([1, 1])
            col1.image(image, caption='Загруженное изображение', use_container_width=True)
            if col2.button("Классифицировать", use_container_width=True):
                with st.spinner("Анализ..."):
                    img_array = np.array(image.resize((width, height)), dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    
                    start_time = time.time()
                    interpreter.invoke()
                    end_time = time.time()

                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    pred_index = np.argmax(prediction)
                    class_labels = {v: k for k, v in class_indices.items()}
                    pred_name = class_labels[pred_index]
                    confidence = np.max(prediction)

                    col2.success(f"### Вид: **{pred_name}**")
                    col2.write(f"**Уверенность:** `{confidence:.2%}`")
                    col2.metric("Время предсказания", f"{(end_time - start_time) * 1000:.2f} мс")