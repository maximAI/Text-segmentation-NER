# Text segmentation NER
Задача: Сделать сегментацию текстов, разметку данных сделать самостоятельно.

<a name="5"></a>
## [Оглавление:](#5)
1. [Загрузка данных](#1)
2. [Преобразование данных](#2)
3. [Создание сети](#3)
4. [Проверка работы сети](#4)

Импортируем нужные библиотеки.
```
!pip install pymorphy2
```
```
import numpy as np                                              # Подключим numpy - библиотеку для работы с массивами данных
import re                                                       # Импортируем модуль re
import os                                                       # Импортируем модуль os для загрузки данных
import pandas as pd                                             # Загружаем библиотеку Pandas
import matplotlib.pyplot as plt                                 # Подключим библиотеку для визуализации данных
import time                                                     # Импортируем модуль time
import pymorphy2                                                # Импортируем pymorphy2
from tensorflow.keras.models import Model, load_model, Sequential   # Загружаем абстрактный класс базовой модели сети от кераса
from tensorflow.keras.preprocessing.text import Tokenizer       # Импортируем Tokenizer
from tensorflow.keras import utils                              # Подключим utils
from tensorflow.keras.utils import plot_model                   # Подключим plot_model для отрисовки модели
# Подключим необходимые слои
from tensorflow.keras.layers import Dense, Embedding, Input, concatenate, Activation, \
                                    MaxPooling1D, Conv1D, BatchNormalization, \
                                    Dropout, Conv2DTranspose, Conv1DTranspose, Lambda, \
                                    SpatialDropout1D, Flatten
from tensorflow.keras import backend as K                       # Подключим бэкэнд Керас
from tensorflow.keras.optimizers import Adam, Adadelta          # Подключим оптимизаторы
from google.colab import files                                  # Подключим гугл диск
from gensim.models import word2vec                              # Импортируем word2vec
from tensorflow.keras.callbacks import LambdaCallback           # Импортируем колбэк
```
Объявим необходимые функции.
```
def readText(fileName):
    '''
    Функция чтения файла текста из файла, очитска от знаков препинания
    '''
    f = open(fileName, 'r') # Открываем наш файл для чтения и считываем из него данные 
    text = f.read()         # Записываем прочитанный текст в переменную 
    # Определяем символы для удаления
    delSymbols = ['\n', "\t", "\ufeff", ".", "_", "-", ",", "!", "?", "–", "(", ")", "«", "»", "№", ";"]

    for dS in delSymbols:               # Каждый символ в списке символов для удаления
        text = text.replace(dS, " ")    # Удаляем, заменяя на пробел

    # Ищем шаблоны и символы в строке и меняем на указанную подстроку
    text = re.sub("[.]", " ", text) 
    text = re.sub(":", " ", text)
    text = re.sub("<", " <", text)
    text = re.sub(">", "> ", text)

    # Метод split разделит текст по пробелам (а их может быть несколько после удаления символов)
    # При помощи метода join запишем все разделенные слова снова в строку
    text = ' '.join(text.split()) 

    text = text.lower() # Переводим текст в нижний регистр
    return text         # Возвращаем тексты
```
```
def text2Words(text):
    '''
    Функция преобразования исходного текста в список из слов в нормальной форме
    '''
    morph = pymorphy2.MorphAnalyzer()   # Создаем экземпляр класса MorphAnalyzer
    words = text.split(' ')             # Разделяем текст на пробелы
    words = [morph.parse(word)[0].normal_form for word in words] #Переводим каждое слово в нормалную форму  
    return words                        # Возвращаем слова
```
```
def get01XSamples(tok_agreem, tags_index):
    '''
    Функция собирает список индексов и dummy encoded вектора
    '''  
    tags01 = []     # Список для тегов
    indexes = []    # Здесь будут лежать индексы
    
    for agreement in tok_agreem:                # Проходимся по каждому договору-списку
        tag_place = [0, 0, 0, 0, 0, 0, 0, 0]    # Создаем вектор [0,0,0,0,0,0,0,0]
        for ex in agreement:                    # Проходимся по каждому слову договора
            if ex in tags_index:                # Смотрим, если индекс оказался нашим тегом
            place = np.argwhere(tags_index==ex) # Записываем под каким местом лежит этот тег в своем списке
            if len(place)!=0:                   # Проверяем, чтобы тег действительно был
                if place[0][0] < 8:             # Первые шесть тегов в списке - открывающие
                tag_place[place[0][0]] = 1      # Поэтому ставим 1
                else: 
                tag_place[place[0][0] - 8] = 0  # Остальные теги закрывающие, так что меняем на ноль
            else:          
            tags01.append(tag_place.copy())     # Расширяем наш список с каждой итерацией. Получаем в конце длинный список из всех тегов в одном 
            indexes.append(ex)                  # Докидываем индекс слова в список индексов

    return indexes, tags01
```
```
def reverseIndex(clean_voc, x):
    '''
    Функция получения списка слов из индексов
    '''     
    reverse_word_map = dict(map(reversed, clean_voc.items()))   # Берем пары значений всего словаря и размечаем наоборот, т.е. value:key
    words = [reverse_word_map.get(letter) for letter in x]      # Вытаскиваем по каждому ключу в список
    return words # Возвращаем полученный текст
```
```
def getSetFromIndexes(wordIndexes, xLen, step):
    '''
    Функция формирования выборок из индексов
    '''
    xBatch = []                 # Лист для фрагментов текста
    wordsLen = len(wordIndexes) # Определяем длинну текста
    index = 0                   # Задаем стартовый индекс
    
    while (index + xLen <= wordsLen): # Пока сумма индекса с длинной фрагмента меньше или равна числу слов в выборке
        xBatch.append(wordIndexes[index:index+xLen]) # Добавляем X в лист фразментов текста
        index += step           # Сдвигаемся на step

    return xBatch               # Лист для фрагментов текста
```
```
def getSets(model, senI, tagI):
    '''
    Функция создания выборки
    '''
    xVector = [] # Здесь будет лежать embedding представление каждого из индексов
    tmp = [] # Временный список
    for text in senI: # Проходимся по каждому тексту-списку
        tmp=[]
        for word in text: # Проходимся по каждому слову в тексте-списке
        tmp.append(model[word]) 

        xVector.append(tmp)

    return np.array(xVector), np.array(tagI)
```
```
def dice_coef(y_true, y_pred):
    '''
    Функция, которая смотрит на пересечение областей. Нужна для accuracy
    '''
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
```
```
def reverseIndex(clean_voc, x):
    '''
    Функция получения списка слов из индексов
    ''' 
    reverse_word_map = dict(map(reversed, clean_voc.items())) # Берем пары значений всего словаря и размечаем наоборот, т.е. value:key
    words = [reverse_word_map.get(letter) for letter in x] # Вытаскиваем по каждому ключу в список
    return words # Возвращаем полученный текст
```
```
def recognizeSet(tagI, pred, tags, length, value):
    '''
    Функция, выводящая точность распознавания каждой категории отдельно
    '''     
    total = 0

    for j in range(8): # общее количество тегов
        correct = 0
        for i in range(len(tagI)): # проходимся по каждому списку списка тегов
        for k in range(length): # проходимся по каждому тегу
            if tagI[i][k][j]==(pred[i][k][j] > value).astype(int): # если соответствующие индексы совпадают, значит сеть распознала верно
            correct+=1 
        print("Сеть распознала категорию '{}' на {}%".format(tags[j], round(100 * correct / (len(tagI) * length), 2)))
        total += 100 * correct / (len(tagI) * length)
    print("Cредняя точность {}%".format(round(total/8, 2)))
```
[:arrow_up:Оглавление](#5)
<a name="1"></a>
## Загрузка данных.
```
directory = '/content/drive/MyDrive/База/'
os.listdir(directory)[30:35]
```
```
print('Всего', len(os.listdir(directory)), 'текстов.')
```
```
agreements = []                         # Список, в который запишем все наши договоры
curTime = time.time()                   # Засечем текущее время
for filename in os.listdir(directory):  # Проходим по всем файлам в директории договоров
  txt = readText(directory + filename)  # Читаем текст договора
  if txt != '':                         # Если текст не пустой
    agreements.append(readText(directory + filename)) # Преобразуем файл в одну строку и добавляем в agreements
print('На преобразование ушло:', round(time.time() - curTime, 2), ' с.')
```
```
words = []                              # Здесь будут храниться все тесты в виде списка слов
curTime = time.time()                   # Засечем текущее время
for i in range(len(agreements)):        # Проходимся по всем текстам
  words.append(text2Words(agreements[i])) # Преобразуем очередной текст в список слов и добавляем в words
print('На преобразование ушло:', round(time.time() - curTime, 2), ' с.')
```
```
wordsToTest = words[-7:]                # Возьмем 7 текстов для финальной проверки обученной нейронной сети 
words = words[:-7]                      # Для обученающей и проверочной выборок возьмем все тексты, кроме последних 7
```
[:arrow_up:Оглавление](#5)
<a name="2"></a>
## Преобразование данных.
```
# lower=True - приводим слова к нижнему регистру
# char_level=False - просим токенайзер учитывать слова, а не отдельные символы
tokenizer = Tokenizer(lower=True, filters = '', char_level=False)

tokenizer.fit_on_texts(words) # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности
clean_voc = {} 

for item in tokenizer.word_index.items(): # Преобразуем полученный список 
  clean_voc[item[0]] = item[1] # В словарь, меняя местами элементы полученного кортежа 
```
```
# Преобразовываем текст в последовательность индексов согласно частотному словарю
tok_agreem = tokenizer.texts_to_sequences(words) # Обучающие тесты в индексы
```
Описание тэгов.
```
# <s1> описание природы
# <s2> оружие
# <s3> растения
# <s4> ФИО
# <s5> даты
# <s6> адреса и геолокации
# <s7> ругательства
# <s8> звания

tags_index = ['<s' + str(i) + '>' for i in range(1, 9)] # Получаем список открывающих тегов
closetags = ['</s' + str(i) + '>' for i in range(1, 9)] # Получаем список закрывающих тегов
tags_index.extend(closetags) # Объединяем все теги

tags_index = np.array([clean_voc[i] for i in tags_index]) # Получаем из словаря частотности индексы всех тегов
print('Индексы тегов:', tags_index)
```
```
xData, yData = get01XSamples(tok_agreem,tags_index) # Распознаем теги и создаем список с ними, с индексами
decoded_text = reverseIndex(clean_voc, xData)       # Для создания списков с embedding-ами сначала преобразуем список индексов обратно в слова
```
Зададим параметры.
```
xLen = 256          # Длина окна
step = 30           # Шаг 
embeddingSize = 300 # Количество измерений для векторного пространства
```
Генерируем наборы с заданными параметрами окна.
```
xTrain = getSetFromIndexes(decoded_text, xLen, step)    # Последовательность из xLen слов
yTrain = getSetFromIndexes(yData, xLen, step)           # Последовательность из xLen-тегов
```
```
# Передаем в word2vec списки списков слов для обучения
# size = embeddingSize - размер эмбеддинга
# window = 10 - минимальное расстояние между словами в эмбеддинге 
# min_count = 1 - игнорирование всех слов с частотой, меньше, чем 1
# workers = 10 - число потоков обучения эмбеддинга
# iter = 10 - число эпох обучения эмбеддинга

modelGENSIM = word2vec.Word2Vec(xTrain, size = embeddingSize, 
                                window = 10, min_count = 1, 
                                workers = 10, iter = 10)
```
```
xTrainGENSIM, yTrainGENSIM = getSets(modelGENSIM, xTrain, yTrain)
```
Тестовая выборка.
```
# Преобразовываем текст в последовательность индексов согласно частотному словарю
tok_agreemTest = tokenizer.texts_to_sequences(wordsToTest) # Обучающие тесты в индексы
```
```
xDataTest, yDataTest = get01XSamples(tok_agreemTest, tags_index)    # Распознаем теги и создаем список с ними, с индексами
decoded_text = reverseIndex(clean_voc, xDataTest)                   # Для создания списков с embedding-ами сначала преобразуем список индексов обратно в слова
```
```
# Генерируем наборы с заданными параметрами окна
xTest = getSetFromIndexes(decoded_text, xLen, step) # Последовательность из xLen слов
yTest = getSetFromIndexes(yDataTest, xLen, step)    # Последовательность из xLen-тегов
```
```
# Передаем в word2vec списки списков слов для обучения
# size = embeddingSize - размер эмбеддинга
# window = 10 - расстояние между текущим и прогнозируемым словом в предложении
# min_count = 1 - игнорирование всех слов с частотой, меньше, чем 1
# workers = 10 - число потоков обучения эмбеддинга
# iter = 10 - число эпох обучения эмбеддинга

modelGENSIM = word2vec.Word2Vec(xTest, size = embeddingSize, 
                                window = 10, min_count = 1, 
                                workers = 10, iter = 10)
```
```
xTestGENSIM, yTestGENSIM = getSets(modelGENSIM, xTest, yTest)
```
Определим список тэгов.
```
tags = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
# <s1> описание природы
# <s2> оружие
# <s3> растения
# <s4> ФИО
# <s5> даты
# <s6> адреса и геолокации
# <s7> ругательства
# <s8> звания
```
[:arrow_up:Оглавление](#5)
<a name="3"></a>
## Создаем сеть.
```
def create_Conv1d(xLen, embeddingSize):
    '''
    Функция для создания Conv1d-сети
    '''
    text_input_layer = Input((xLen, embeddingSize)) 
    text_layer = Conv1D(16, 3, padding='same', activation='relu')(text_input_layer)
    text_layer = Conv1D(32, 3, padding='same', activation='relu')(text_layer)
    text_layer = Conv1D(64, 3,padding='same', activation='relu')(text_layer) 
    text_layer = Conv1D(yTrainGENSIM.shape[-1], 3, padding='same', activation='sigmoid')(text_layer)
    model = Model(text_input_layer, text_layer)
    model.compile(optimizer=Adam(),
                    loss='categorical_crossentropy',
                    metrics=[dice_coef])
    return model
```
Объявляем модель, запускаем обучение.
```
model_conv1d = create_Conv1d(xLen, embeddingSize)

history = model_conv1d.fit(xTrainGENSIM, yTrainGENSIM, 
                           epochs=20, 
                           batch_size=100, 
                           validation_split = 0.2)
```
[:arrow_up:Оглавление](#5)
<a name="4"></a>
## Проверяем работу сети на тестовых данных.
```
# <s1> описание природы
# <s2> оружие
# <s3> растения
# <s4> ФИО
# <s5> даты
# <s6> адреса и геолокации
# <s7> ругательства
# <s8> звания
pred = model_conv1d.predict(xTestGENSIM) # сделаем предсказание
recognizeSet(yTestGENSIM, pred, tags, xLen, 0.999)
```
![Иллюстрация к проекту](https://github.com/maximAI/Autoencoder/blob/main/Screenshot_1.jpg)
[:arrow_up:Оглавление](#5)

[Ноутбук](https://colab.research.google.com/drive/1mGO8sIbEQA8FNLk9Yz4lc_rVArD6LdJY?usp=sharing)
