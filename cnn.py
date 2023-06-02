import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import keras
from keras import regularizers
from keras.callbacks import CSVLogger

#Generación de DataFrame a partir del conjunto de datos
image_dir = Path('path_to_train')
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath',dtype='string').astype(str)
labels = pd.Series(labels, name='Label',dtype='string')
dataset_custom = pd.concat([filepaths, labels], axis=1)

#Division de datos de entrenamiento y de prueba
train_data, test_data = train_test_split(dataset_custom,test_size=0.1)

#Método de aumento de datos
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=90,
            brightness_range=[0.1, 0.7],
            width_shift_range=0.5,
            height_shift_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function = tf.keras.applications.vgg16.preprocess_input,
            validation_split=0.2

)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.vgg16.preprocess_input,
)

#Division entrenamiento - validación - evaluación
train_images = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='Filepath',
    y_col='Label',
	batch_size=32,
	target_size=(224,224),
    class_mode='categorical',
    seed=42,
    subset='training',
    shuffle=True,
)

test_images = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='Filepath',
        y_col='Label',
    	batch_size=32,
    	target_size=(224,224),
        seed=42,
        subset="validation",
        shuffle=True,
)
eval_images = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        x_col='Filepath',
        y_col='Label',
    	batch_size=32,
    	target_size=(224,224),
        seed=42,
        shuffle=False
)

#Modelo de red
vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', input_shape=(224,224,3), include_top=False)
fine_tune = 3
if fine_tune > 0:
    for layer in vgg16.layers[:-fine_tune]:
        layer.trainable = False
else:
    for layer in vgg16.layers:
        layer.trainable = False

model = tf.keras.models.Sequential([
        vgg16,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(20, activation='softmax')
])
#Compilación y función de optimización
model.compile(
    optimizer = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy','top_k_categorical_accuracy']
)

#Función de callback para almacenar resultados del entrenamiento y validación
csv_logger = CSVLogger('training.log',separator=',', append=False)

#Entrenamiento del modelo
history = model.fit(
    train_images,
    validation_data=test_images,
    epochs=50,
    verbose=1,
    callbacks=[
          tf.keras.callbacks.EarlyStopping(
                  monitor='loss',
                  patience=5,
                  restore_best_weights=True
              ),
          csv_logger
      ]
)

# Función de callback para almacenar resultados de la evaluación
CSVLogger.on_test_begin = CSVLogger.on_train_begin
CSVLogger.on_test_batch_end = CSVLogger.on_epoch_end
CSVLogger.on_test_end = CSVLogger.on_train_end
csv_logger = CSVLogger('evaluate.log',separator=',', append=False)

#Evaluacíón del modelo
model.evaluate(eval_images,callbacks=[csv_logger])

#Guardado del modelo
model.save('vgg16_20.h5')
