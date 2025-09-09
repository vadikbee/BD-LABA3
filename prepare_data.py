# prepare_data.py
import os
import shutil
import re

def restructure_directory(source_dir, dest_dir):
    """
    Реструктурирует "плоскую" директорию с изображениями в директорию
    с подпапками для каждого класса.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Создана директория: {dest_dir}")

    files = os.listdir(source_dir)
    print(f"Найдено {len(files)} файлов в '{source_dir}'. Начинаю обработку...")

    for filename in files:
        if not (filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
            continue

        match = re.match(r"(.+?)\s*\(\d+\)", filename)
        if not match:
            print(f"Не удалось извлечь имя класса из файла: {filename}. Пропускаем.")
            continue

        class_name = match.group(1).strip()
        class_dir = os.path.join(dest_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(class_dir, filename)
        shutil.copy(source_path, dest_path)

    print(f"Реструктуризация директории '{source_dir}' завершена.")

if __name__ == "__main__":
    source_train_dir = 'train'
    source_valid_dir = 'test'
    dest_train_dir = 'train_structured'
    dest_valid_dir = 'valid_structured'

    if os.path.exists(source_train_dir):
        print("Начинаю обработку обучающей выборки...")
        restructure_directory(source_train_dir, dest_train_dir)
    else:
        print(f"Ошибка: Директория '{source_train_dir}' не найдена.")

    if os.path.exists(source_valid_dir):
        print("\nНачинаю обработку валидационной выборки...")
        restructure_directory(source_valid_dir, dest_valid_dir)
    else:
        print(f"Ошибка: Директория '{source_valid_dir}' не найдена.")

    print("\nВсе готово! Созданы папки 'train_structured' и 'valid_structured'.")