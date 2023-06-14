import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


def load_path(path, part):
    ds = []

    for wrapper in os.listdir(path):
        wrapper_path = os.path.join(path, wrapper)
        
        if not os.path.isdir(wrapper_path):
            continue
        
        for body in os.listdir(wrapper_path):
            if body != part:
                continue
            
            body_part = body
            body_path = os.path.join(wrapper_path, body)
            
            for id_s in os.listdir(body_path):
                sick = id_s
                dir = os.path.join(body_path, id_s)
                
                for sc in os.listdir(dir):
                    sc_path = os.path.join(dir, sc)
                    
                    if sc.split('_')[-1] == 'positive':
                        label = 'fractured'
                    elif sc.split('_')[-1] == 'negative':
                        label = 'normal'
                    
                    for img in os.listdir(sc_path):
                        img_path = os.path.join(sc_path, img)
                        ds.append({
                            'body_part': body_part,
                            'sick': sick,
                            'label': label,
                            'image_path': img_path
                        })

    return ds


# Функция знает какую часть тренировать, сохранять модель и графики 
def tp(part):
    
    wrapper = os.path.dirname(os.path.abspath(__file__))
    photodr = wrapper + '/SampleTest/'
    data = loading(photodr, part)
    lb = []
    filepaths = []

    # Добавление меток для фрейма данных для каждой категории
    for rw in data:
        lb.append(rw['label'])
        filepaths.append(rw['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    lb = pd.Series(lb, name='Label')

    img = pd.concat([filepaths, lb], axis=1)

    
    train_df, test_df = train_test_split(img, train_size=0.9, shuffle=True, random_state=1)

     # каждый генератор для обработки и преобразования путей к файлам в массивы изображений,
     # и метки в метки с горячим кодированием.
     # Полученные генераторы затем можно использовать для обучения и оценки модели глубокого обучения.

    tr_gnr = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=tf.keras.applications.densenet.preprocess_input, validation_split=0.2)


    test_gnr = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)

    tr_photo = tr_gnr.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )

    effect_photo = tr_gnr.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    trial_photo = test_gnr.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # Мы используем 3 канала rgb и изображения 224x224 пикселей, используем извлечение признаков и средний пул
    early = tf.keras.applications.densenet.DenseNet121(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    # Для более быстрой работы
    early.trainable = False

    inputs = early.input
    x = tf.keras.layers.Dense(128, activation='relu')(early.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)

    # выводит Dense '2' из-за 2 классов, fratured и normal
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    md = tf.keras.Model(inputs, outputs)
    

    # Оптимизатор Адама с низкой скоростью обучения для большей точности
    md.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Когда наша модель слишком подходит или исчезает градиент, с восстановлением лучших значений
    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = md.fit(tr_photo, validation_data=effect_photo, epochs=25, callbacks=[cb])

    # Сохранение модели
    md.save(wrapper + "/models" + part + "_frac.h5")
    res = md.evaluate(trial_photo, verbose=0)
    

    # Создание графиков и их сохранение
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Модель точности')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.legend(['training data', 'test data'], loc='upper left')
    
    figAcc = plt.gcf()
    mdata = os.path.join(wrapper, "./charts" + part + ".jpeg")
    figAcc.savefig(mdata)
    plt.clf()

   


# Запуск функцию и создание модели для каждой части в массиве
array = ["Elbow", "Hand", "Shoulder"]
for ct in array:
    tp(ct)
