import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import PIL
import PIL.Image

import tensorflow as tf


from tensorflow import keras
from keras_preprocessing import image
import pathlib
import os
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv("/home/camila/Área de Trabalho/pibiti/train.csv",index_col=0) # index_col = 0 (Para ignorar a primeira coluna index)
print(df.shape)
df.head()



#Será pego as imagens que estão na pasta images convert
SOURCE = '/home/camila/Área de Trabalho/pibiti/images convert'

#Serão divididas na pasta criada images
SPLIT_DIR = 'temp/images/'

    
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


model = keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32,(2,2),activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')])



model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
)

####



###################3



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

#Inferencia TF
predictions = model.predict(X_test, batch_size= 10)
score = tf.nn.softmax(predictions)
score = np.array(score)
df_out = pd.concat([test_set.reset_index(), pd.DataFrame(score, columns = class_names)], axis=1).set_index("image_id")
df_out.to_csv('/home/camila/Área de Trabalho/pibiti/inferenciaTF.csv')
df_out.head()

#Inferencia TFLITE
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
file = open( '/home/camila/Área de Trabalho/pibiti/TFLITE.tflite' , 'wb' ) #Criando e digitando um arquivo binário!
file.write( tflmodel )
file.close()

interpreter = tf.lite.Interpreter('/home/camila/Área de Trabalho/pibiti/TFLITE.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() #classes


interpreter.resize_tensor_input(input_details[0]['index'], (len(X_test), 180, 180, 3))
interpreter.resize_tensor_input(output_details[0]['index'], (len(X_test), 4))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


img = X_test
interpreter.set_tensor(input_details[0]['index'],img)
interpreter.invoke()

model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", model_predictions.shape)

tflite_predicted = np.argmax(model_predictions)

predictionsTF = model.predict(X_test, batch_size= 10)
score = tf.nn.softmax(predictionsTF)
score = np.array(score)
df_out = pd.concat([test_set.reset_index(), pd.DataFrame(score, columns = class_names)], axis=1).set_index("image_id")
df_out.to_csv('/home/camila/Área de Trabalho/pibiti/inferenciaTFLITE.csv')
df_out.head()


