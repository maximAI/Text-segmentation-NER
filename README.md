# Text segmentation NER
Задача: Сделать сегментацию текстов, разметку данных сделать самостоятельно.

<a name="5"></a>
## [Оглавление:](#5)
1. [Callbacks](#1)
2. [Загрузка данных](#2)
3. [Создание сети](#3)
4. [Проверка работы сети](#4)

Импортируем нужные библиотеки.
```
import numpy as np                                              # Подключим numpy - библиотеку для работы с массивами данных
import pandas as pd                                             # Загружаем библиотеку Pandas
import matplotlib.pyplot as plt                                 # Подключим библиотеку для визуализации данных
import os                                                       # Импортируем модуль os для загрузки данных
import time                                                     # Импортируем модуль time
from google.colab import drive                                  # Подключим гугл диск
from tensorflow.keras.models import Model                       # Загружаем абстрактный класс базовой модели сети от кераса
# Подключим необходимые слои
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, \
                                    concatenate, Activation, MaxPooling2D, Conv2D, \
                                    BatchNormalization, Dropout, MaxPooling1D, UpSampling2D
from tensorflow.keras import backend as K                       # Подключим бэкэнд Керас
from tensorflow.keras.optimizers import Adam                    # Подключим оптимизатор
from tensorflow.keras import utils                              # Подключим utils
from tensorflow.keras.utils import plot_model                   # Подключим plot_model для отрисовки модели
from tensorflow.keras.preprocessing import image                # Подключим image для работы с изображениями
from PIL import Image                                           # Подключим Image для работы с изображениями
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback # 
import tensorflow as tf                                         # Импортируем tensorflow
import random                                                   # Импортируем библиотеку random
```
Объявим необходимые функции.
```
def plotImages(xTrain, pred, shape=(420, 540, 3)):
    '''
    Функция для вывода изображений
    '''
    n = 5  # количество картинок, которые хотим показать
    plt.figure(figsize=(18, 6)) # указываем размеры фигуры
    for i in range(n): # для каждой картинки из n(5)
        index = np.random.randint(0, pred.shape[0]) # startIndex - начиная с какого индекса хотим заплотить картинки
        # Показываем картинки из тестового набора
        ax = plt.subplot(2, n, i + 1) # выведем область рисования Axes
        plt.imshow(xTrain[index].reshape(shape)) # отрисуем правильные картинки
        ax.get_xaxis().set_visible(False) # скрываем вывод координатной оси x
        ax.get_yaxis().set_visible(False) # скрываем вывод координатной оси y

        # Показываем восстановленные картинки
        ax = plt.subplot(2, n, i + 1 + n) # выведем область рисования Axes 
        plt.imshow(pred[index].reshape(shape)) # отрисуем обработанные сеткой картинки 
        ax.get_xaxis().set_visible(False) # скрываем вывод координатной оси x
        ax.get_yaxis().set_visible(False) # скрываем вывод координатной оси y
    plt.show()
```
```
def getMSE(x1, x2):
    '''
    Функция среднеквадратичной ошибки
    
    Return:
        возвращаем сумму квадратов разницы, делённую на длину разницы
    '''
    x1 = x1.flatten() # сплющиваем в одномерный вектор
    x2 = x2.flatten() # сплющиваем в одномерный вектор
    delta = x1 - x2 # находим разницу
    return sum(delta ** 2) / len(delta)
```
```
def load_images(images_dir, img_height, img_width): 
    '''
    Функция загрузки изображений, на вход принемает имя папки с изображениями, 
    высоту и ширину к которой будут преобразованы загружаемые изображения

    Return:
        возвращаем numpy массив загруженных избражений
    '''
    list_images = [] # создаем пустой список в который будем загружать изображения
    for img in sorted(os.listdir(images_dir)): # получим список изображений и для каждого изображения
    # добавим в список изображение в виде массива, с заданными размерами
        list_images.append(image.img_to_array(image.load_img(os.path.join(images_dir, img), \
                                                            target_size=(img_height, img_width))))
    return np.array(list_images)
```
[:arrow_up:Оглавление](#5)
<a name="1"></a>
## Callbacks.
Объявим колбэки, которые будем применять для обучения сети.
```
# Остановит обучение, когда valloss не будет расти
earlyStopCB = EarlyStopping(
                    monitor='loss',
                    min_delta=0,
                    patience=8,
                    verbose=0,
                    mode='min',
                    baseline=None,
                    restore_best_weights=False,
                )
```
```
# Вывод шага обучения
def on_epoch_end(epoch, logs):
  lr = tf.keras.backend.get_value(model.optimizer.learning_rate)
  print(' Коэффициент обучения', lr)

lamCB = LambdaCallback(on_epoch_end=on_epoch_end)
```
```
# Меняет шаг обучения
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=5, min_lr=0.00000001)
```
[:arrow_up:Оглавление](#5)
<a name="2"></a>
## Загрузка данных
```
# Распаковка
!unzip -q '/content/drive/MyDrive/text_cleaning.zip' # Загрузка всех изображений

# Загрузка изображений с шумом
images_dir = '/content/text_cleaning/train_X'
# Зададим размеры изображений
img_height = 420
img_widht = 540

cur_time = time.time()
xTrain_imag = load_images(images_dir, img_height, img_widht)
print('Время загрузки: ', round(time.time() - cur_time, 2), 'с', sep = '')

# Нормируем
xTrain_img = xTrain_imag / 255
```
```
# Загрузка изображений с шумом
images_dir = '/content/text_cleaning/train_Y'

cur_time = time.time()
yTrain_imag = load_images(images_dir, img_height, img_widht)
print('Время загрузки: ', round(time.time() - cur_time, 2), 'с', sep = '')

# Нормируем
yTrain_img = yTrain_imag / 255
```
Отрисуем загруженные изображения
```
plotImages(xTrain_img, yTrain_img) # исходные и зашумленные варианты
```
[:arrow_up:Оглавление](#5)
<a name="3"></a>
## Создаем сеть.
```
def AE(shape=(420, 540, 3)):
    '''
    Функция создания базового автокодировщика
    '''
    img_input = Input((shape)) # Задаем входные размеры

    x = Conv2D(128, (3, 3), padding='same', activation='elu')(img_input) # Входные данные передаем на слой двумерной свертки
    x = BatchNormalization()(x) # Затем пропускаем через слой нормализации данных 
    x = Conv2D(128, (3, 3), padding='same', activation='elu')(x) # Далее снова слой двумерной свертки
    x = BatchNormalization()(x) # Слой нормализации данных
    x = MaxPooling2D()(x) # Передаем на слой подвыборки, снижающий размерность поступивших на него данных
    x = Dropout(0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='elu')(x) # Передаем на слой двумерной свертки
    x = BatchNormalization()(x) # Пропускаем через слой нормализации данных 
    x = Conv2D(256, (3, 3), padding='same', activation='elu')(x)  # Далее снова слой двумерной свертки
    x = BatchNormalization()(x) # Слой нормализации данных
    
    z = MaxPooling2D()(x) # Скрытое пространство
    
    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='elu')(z) 
    x = BatchNormalization()(x) # Слой нормализации данных   
    x = Conv2D(256, (3, 3), padding='same', activation='elu')(x) # Передаем на слой двумерной свертки
    x = BatchNormalization()(x) # Слой нормализации данных
    x = Conv2D(256, (3, 3), padding='same', activation='elu')(x) # Еще слой двумерной свертки
    x = BatchNormalization()(x) # Слой нормализации данных
    x = Dropout(0.2)(x)

    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='elu')(x) 
    x = BatchNormalization()(x) # Слой нормализации данных
    x = Conv2D(128, (3, 3), padding='same', activation='elu')(x) # Передаем на слой двумерной свертки
    x = BatchNormalization()(x) # Слой нормализации данных
    x = Conv2D(128, (3, 3), padding='same', activation='elu')(x) # Еще слой двумерной свертки
    x = BatchNormalization()(x) # Слой нормализации данных

    # Финальный слой двумерной свертки, выдающий итоговое изображение
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(img_input, x) # Указываем модель
    model.compile(optimizer=Adam(),
                  loss='mse') # Компилируем модель с оптимайзером Адам и среднеквадратичной ошибкой

    return model # Функция вернет заданную модель
```
Объявляем модель, запускаем обучение.
```
model = AE()

model.fit(xTrain_img[:130], yTrain_img[:130], 
          epochs=100, batch_size=5, 
          validation_data = (xTrain_img[130:], yTrain_img[130:]),
          callbacks=[earlyStopCB, reduce_lr, lamCB])
```
[:arrow_up:Оглавление](#5)
<a name="4"></a>
## Проверяем работу сети на тестовых данных.
```
images_dir = '/content/text_cleaning/test' # зададим имя папки в которую распоковали изображения
```
```
xTest_imag = load_images(images_dir, img_height, img_widht) # загрузим избражения
xTest_img = xTest_imag / 255 # отнормируем изображения от 0 до 1
# выведем изображение
plt.imshow(xTest_img[1])
plt.axis('off')
plt.show()
print('\nРазмерность', xTest_img.shape)
```
![Иллюстрация к проекту](https://github.com/maximAI/Autoencoder/blob/main/Screenshot_1.jpg)
```
def predictImg(image_my):
    '''
    Функция предикта только 1 изображения
    '''
    image_my = np.array(image_my)
    image_my = image_my / 255

    image_my_Denoise = model.predict(image_my.reshape(-1, 420, 540, 3))
    image_my_Denoise = image_my_Denoise * 255
    image_my_Denoise = image_my_Denoise.astype('uint8')

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title('Исходная картинка')
    plt.axis('off')
    plt.imshow(image_my)
    plt.subplot(1, 2, 2)
    plt.title('Обработанная картинка')
    plt.axis('off')
    plt.imshow(image_my_Denoise.reshape(420, 540, 3))
    plt.show()
```
```
image_my_1 = image.load_img('/content/text_cleaning/test/test_001.png', \
                          target_size = (420, 540))
predictImg(image_my_1)
```
![Иллюстрация к проекту](https://github.com/maximAI/Autoencoder/blob/main/Screenshot_2.jpg)
[:arrow_up:Оглавление](#5)

[Ноутбук](https://colab.research.google.com/drive/1F1XRgISbk0EaB-eZc-5kAXFfqTArxeAd?usp=sharing)
