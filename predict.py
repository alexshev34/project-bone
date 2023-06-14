import numpy as np
import tensorflow as tf
import keras.utils as image


cy_part = ["Elbow", "Hand", "Shoulder"]

ct_fr = ['fractured', 'normal']


# Загрузка моделей
elbow = tf.keras.models.load_model("models/Elbow.h5")
hand = tf.keras.models.load_model("models/Hand.h5")
shoulder = tf.keras.models.load_model("models/Shoulder.h5")

# Получение модели и предсказание из трех классов
# Parts - модель предсказания типа кости из 3-х классов
def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        result = model_parts
    else:
        if model == 'Elbow':
            result = elbow
        elif model == 'Hand':
            result = hand
        elif model == 'Shoulder':
            result = shoulder

    # Загрузка изображения
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    photo = np.vstack([x])
    value = np.argmax(result.predict(photo), axis=1)

    # Выбор категории
    if model == 'Parts':
        grade = cy_part[value.item()]
    else:
        grade = ct_fr[value.item()]

    return grade
