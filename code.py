import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import PIL
import PIL.Image

import tensorflow as tf


from tensorflow import keras

import pathlib
import os
import warnings
warnings.filterwarnings('ignore')


##################ORGANIZANDO DATASET
##################################################
############################
df = pd.read_csv("/mnt/data/trainer/plants_diseases_classification/train.csv",index_col=0) # index_col = 0 (Para ignorar a primeira coluna index)
print(df.shape)
df.head()



import shutil
from shutil import copyfile

if os.path.exists('temp'): #Se existir o diretório
    shutil.rmtree('temp') #delete o mesmo

#Cria todas as pastas que serão necessárias para a aplicação
os.mkdir('temp')
os.mkdir('temp/images')
os.mkdir('temp/images/healthy')
os.mkdir('temp/images/multiple_diseases')
os.mkdir('temp/images/rust')
os.mkdir('temp/images/scab')

#Será pego as imagens que estão na pasta images convert
SOURCE = '//mnt/data/trainer/plants_diseases_classification/images convert'

#Serão divididas na pasta criada images
SPLIT_DIR = 'temp/images/'

# Copiando as imagens de treino para o diretório criado
for index, data in df.iterrows(): #iterrows vai varrer todas as linhas em df   #INDEX = NOME DAS IMAGES (TRAIN_1560) #DATA = DADOS DE DETERMINADO INDEX
    label = df.columns[np.argmax(data)] #Recebe o nome das colunas de cada data
    filepath = os.path.join(SOURCE, index + ".jpg") #caminho do arquivo = images do source e juntar com os nomes das images
    destination = os.path.join(SPLIT_DIR, label, index + ".jpg")#No destino, vai juntar no split_dir os labels e as images
    copyfile(filepath, destination)

   
    
for subdir in os.listdir(SPLIT_DIR):
    print(subdir, len(os.listdir(os.path.join(SPLIT_DIR, subdir))))
    
########################################################3
############################################
####################################
##################### 
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    'temp/images',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'temp/images',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(num_classes, class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[list(labels[i]).index(1)])
        plt.axis("off")
        
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([

  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
)

####
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = len(acc)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

###################3

from keras_preprocessing import image

test_set = pd.read_csv("/home/camila/Área de Trabalho/pibiti/test.csv", index_col=0)

X_test = []
for index, data in test_set.iterrows():
    filepath = os.path.join(SOURCE, index + ".jpg")
    img = image.load_img(filepath, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X_test.append(x)
    
X_test = np.vstack(X_test)


#####################
filepath = '//mnt/data/trainer/plants_diseases_classification/images convert/Test_3.jpg'
img = image.load_img(filepath, target_size=(img_height, img_width))
plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
predict = model.predict(x)
score = tf.nn.softmax(predict)
print(class_names)
print(np.array(score))



predictions = model.predict(X_test, batch_size= 10)
score = tf.nn.softmax(predictions)
score = np.array(score)
df_out = pd.concat([test_set.reset_index(), pd.DataFrame(score, columns = class_names)], axis=1).set_index("image_id")
df_out.to_csv('/mnt/data/trainer/plants_diseases_classification/submission.csv')
df_out.head()