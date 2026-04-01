"""
GLOSARIO DE REDES NEURONALES CON KERAS Y TENSORFLOW
====================================================
Funciones para crear, entrenar y evaluar redes neuronales
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# 1. CONFIGURACIÓN INICIAL
# ============================================================================

# Versión de TensorFlow
print(tf.__version__)
print(keras.__version__)

# GPU disponible
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())

# Usar CPU (si tienes problemas con GPU)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Memoria GPU dinámica (evita que TensorFlow tome toda la GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ============================================================================
# 2. PREPARACIÓN DE DATOS
# ============================================================================

# Normalización/Escalado (IMPORTANTE para redes neuronales)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encoding para labels (clasificación multiclase)
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_train_cat = to_categorical(y_train, num_classes=10)

# Para clasificación binaria (0 o 1, no necesita to_categorical)
# y ya está bien como está

# Reshape para imágenes
X_train = X_train.reshape(-1, 28, 28, 1)  # (samples, height, width, channels)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalizar imágenes (0-255 a 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.astype('float32') / 255.0


# ============================================================================
# 3. CREAR MODELO SECUENCIAL (FORMA 1)
# ============================================================================

# Modelo más simple
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Modelo con más capas
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Con Dropout
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Con BatchNormalization
model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))


# ============================================================================
# 4. CREAR MODELO SECUENCIAL (FORMA 2 - MÁS COMPACTA)
# ============================================================================

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# ============================================================================
# 5. CREAR MODELO FUNCIONAL (MÁS FLEXIBLE)
# ============================================================================

from tensorflow.keras import Input, Model

# Modelo funcional básico
inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Modelo funcional con múltiples entradas/salidas
input1 = Input(shape=(10,), name='input1')
input2 = Input(shape=(5,), name='input2')

x1 = Dense(32, activation='relu')(input1)
x2 = Dense(16, activation='relu')(input2)

combined = layers.concatenate([x1, x2])
x = Dense(64, activation='relu')(combined)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input1, input2], outputs=outputs)


# ============================================================================
# 6. CAPAS DENSAS (FULLY CONNECTED)
# ============================================================================

# Capa básica
Dense(64)
Dense(units=64)

# Con activación
Dense(64, activation='relu')
Dense(32, activation='sigmoid')
Dense(10, activation='softmax')
Dense(1, activation='linear')

# Primera capa (especificar input)
Dense(128, input_shape=(784,))
Dense(128, input_dim=784)

# Con regularización
from tensorflow.keras import regularizers
Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))
Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))
Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))

# Con inicializadores
from tensorflow.keras import initializers
Dense(64, activation='relu', kernel_initializer='he_normal')
Dense(64, activation='relu', kernel_initializer='glorot_uniform')
Dense(64, kernel_initializer=initializers.RandomNormal(stddev=0.01))


# ============================================================================
# 7. ACTIVACIONES
# ============================================================================

# En la capa
Dense(64, activation='relu')
Dense(64, activation='sigmoid')
Dense(64, activation='tanh')
Dense(10, activation='softmax')
Dense(1, activation='linear')

# Como capa separada
Dense(64)
Activation('relu')

# Activaciones avanzadas
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU
Dense(64)
LeakyReLU(alpha=0.1)

Dense(64)
PReLU()

Dense(64, activation='elu')
ELU(alpha=1.0)

# ReLU con límite
Dense(64, activation='relu')
Dense(64, activation=tf.nn.relu6)


# ============================================================================
# 8. DROPOUT Y REGULARIZACIÓN
# ============================================================================

# Dropout
Dropout(0.2)  # desactiva 20% de neuronas
Dropout(0.3)
Dropout(0.5)
Dropout(rate=0.3)

# Spatial Dropout (para CNNs)
from tensorflow.keras.layers import SpatialDropout2D
SpatialDropout2D(0.2)

# Gaussian Dropout
from tensorflow.keras.layers import GaussianDropout
GaussianDropout(0.2)

# BatchNormalization
BatchNormalization()
BatchNormalization(momentum=0.99, epsilon=0.001)


# ============================================================================
# 9. COMPILAR MODELO
# ============================================================================

# Clasificación binaria
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Clasificación multiclase (one-hot encoded)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Clasificación multiclase (labels como enteros)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Regresión
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error', 'mse']
)

# Con learning rate personalizado
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Múltiples métricas
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)


# ============================================================================
# 10. FUNCIONES DE PÉRDIDA (LOSS)
# ============================================================================

# Clasificación
loss='binary_crossentropy'  # clasificación binaria
loss='categorical_crossentropy'  # multiclase (one-hot)
loss='sparse_categorical_crossentropy'  # multiclase (enteros)

# Regresión
loss='mse'  # mean squared error
loss='mae'  # mean absolute error
loss='mean_squared_error'
loss='mean_absolute_error'
loss='mean_absolute_percentage_error'
loss='mean_squared_logarithmic_error'
loss='huber'

# Otras
loss='hinge'
loss='squared_hinge'
loss='kullback_leibler_divergence'
loss='poisson'

# Custom loss
from tensorflow.keras import backend as K
def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

model.compile(optimizer='adam', loss=custom_loss)


# ============================================================================
# 11. OPTIMIZADORES
# ============================================================================

# Básicos (strings)
optimizer='adam'
optimizer='sgd'
optimizer='rmsprop'
optimizer='adagrad'
optimizer='adadelta'
optimizer='adamax'
optimizer='nadam'

# Con parámetros personalizados
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad

optimizer=Adam(learning_rate=0.001)
optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
optimizer=Adam(lr=0.001)  # lr deprecado, usar learning_rate

optimizer=SGD(learning_rate=0.01)
optimizer=SGD(learning_rate=0.01, momentum=0.9)
optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

optimizer=RMSprop(learning_rate=0.001)
optimizer=RMSprop(learning_rate=0.001, rho=0.9)

optimizer=Adagrad(learning_rate=0.01)


# ============================================================================
# 12. ENTRENAR MODELO
# ============================================================================

# Entrenamiento básico
history = model.fit(X_train, y_train, epochs=10)

# Con validation split
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2
)

# Con datos de validación separados
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Con batch size
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# Sin verbose
history = model.fit(X_train, y_train, epochs=10, verbose=0)
history = model.fit(X_train, y_train, epochs=10, verbose=1)  # progress bar
history = model.fit(X_train, y_train, epochs=10, verbose=2)  # una línea por época

# Shuffle
history = model.fit(X_train, y_train, epochs=10, shuffle=True)
history = model.fit(X_train, y_train, epochs=10, shuffle=False)


# ============================================================================
# 13. CALLBACKS
# ============================================================================

# Early Stopping (detener si no mejora)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max',
    restore_best_weights=True
)

# ModelCheckpoint (guardar mejor modelo)
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

checkpoint = ModelCheckpoint(
    'model_epoch_{epoch:02d}_acc_{val_accuracy:.2f}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# ReduceLROnPlateau (reducir learning rate)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard
from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs')

# CSVLogger
from tensorflow.keras.callbacks import CSVLogger
csv_logger = CSVLogger('training.log')

# Usar callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Custom callback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\nAlcanzado 95% de accuracy, deteniendo entrenamiento!")
            self.model.stop_training = True

custom_callback = CustomCallback()
history = model.fit(X_train, y_train, epochs=100, callbacks=[custom_callback])


# ============================================================================
# 14. EVALUAR MODELO
# ============================================================================

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

# Sin verbose
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Con batch size
loss, accuracy = model.evaluate(X_test, y_test, batch_size=128)


# ============================================================================
# 15. PREDICCIONES
# ============================================================================

# Predicción (devuelve probabilidades)
predictions = model.predict(X_test)
predictions = model.predict(X_test, batch_size=32)
predictions = model.predict(X_test, verbose=0)

# Para clasificación binaria (0-1)
predictions = model.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int)

# Para clasificación multiclase (argmax para obtener clase)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes = predictions.argmax(axis=1)

# Predicción para una sola muestra
sample = X_test[0:1]
prediction = model.predict(sample)

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

# Predict classes directamente (deprecado pero funciona)
predicted_classes = model.predict_classes(X_test)  # deprecado
# Alternativa:
predicted_classes = np.argmax(model.predict(X_test), axis=1)


# ============================================================================
# 16. VER ARQUITECTURA DEL MODELO
# ============================================================================

# Resumen del modelo
model.summary()

# Información detallada
for layer in model.layers:
    print(layer.name, layer.trainable, layer.output_shape)

# Número de parámetros
model.count_params()

# Configuración
model.get_config()

# Pesos de una capa
weights = model.layers[0].get_weights()
print(weights[0].shape)  # weights
print(weights[1].shape)  # biases


# ============================================================================
# 17. VISUALIZAR ENTRENAMIENTO
# ============================================================================

# Acceder al historial
history.history.keys()
history.history['loss']
history.history['accuracy']
history.history['val_loss']
history.history['val_accuracy']

# Graficar pérdida
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Graficar accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Ambas en subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Validation')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Validation')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()


# ============================================================================
# 18. GUARDAR Y CARGAR MODELOS
# ============================================================================

# Guardar modelo completo (arquitectura + pesos)
model.save('my_model.h5')
model.save('my_model.keras')
model.save('my_model')  # formato SavedModel

# Cargar modelo
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
model = load_model('my_model.keras')
model = load_model('my_model')

# Guardar solo pesos
model.save_weights('model_weights.h5')

# Cargar solo pesos (modelo debe existir)
model.load_weights('model_weights.h5')

# Guardar arquitectura como JSON
json_config = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(json_config)

# Cargar arquitectura
from tensorflow.keras.models import model_from_json
with open('model_architecture.json', 'r') as f:
    json_config = f.read()
model = model_from_json(json_config)


# ============================================================================
# 19. REDES NEURONALES CONVOLUCIONALES (CNN)
# ============================================================================

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D

# CNN básica
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# CNN más compleja
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Con BatchNormalization
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation='softmax')
])

# Capas convolucionales
Conv2D(32, (3, 3))  # 32 filtros, kernel 3x3
Conv2D(64, kernel_size=(5, 5))
Conv2D(128, (3, 3), strides=(1, 1))
Conv2D(64, (3, 3), padding='same')  # mantiene dimensiones
Conv2D(64, (3, 3), padding='valid')  # default

# Pooling
MaxPooling2D((2, 2))
MaxPooling2D(pool_size=(2, 2))
MaxPooling2D((3, 3), strides=(2, 2))
AveragePooling2D((2, 2))

# Global pooling
GlobalMaxPooling2D()
GlobalAveragePooling2D()

# Flatten
Flatten()


# ============================================================================
# 20. DATA AUGMENTATION (AUMENTO DE DATOS)
# ============================================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator (método clásico)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ajustar a los datos
datagen.fit(X_train)

# Entrenar con data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test)
)

# Keras layers para augmentation (más moderno)
from tensorflow.keras.layers.experimental import preprocessing

data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomZoom(0.1),
])

# Incluir en el modelo
model = Sequential([
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # ... resto del modelo
])


# ============================================================================
# 21. TRANSFER LEARNING
# ============================================================================

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, InceptionV3

# Cargar modelo pre-entrenado (sin capa final)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar capas base
base_model.trainable = False

# Añadir capas propias
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Otros modelos
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Descongelar algunas capas para fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:  # congelar todas menos las últimas 4
    layer.trainable = False


# ============================================================================
# 22. REDES RECURRENTES (RNN, LSTM, GRU)
# ============================================================================

from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional

# LSTM básico
model = Sequential([
    LSTM(128, input_shape=(timesteps, features)),
    Dense(1)
])

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(32),
    Dense(1)
])

# GRU
model = Sequential([
    GRU(128, input_shape=(timesteps, features)),
    Dense(1)
])

# SimpleRNN
model = Sequential([
    SimpleRNN(64, input_shape=(timesteps, features)),
    Dense(1)
])

# Bidirectional
model = Sequential([
    Bidirectional(LSTM(64), input_shape=(timesteps, features)),
    Dense(1)
])

# Stacked LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])


# ============================================================================
# 23. MÉTRICAS PERSONALIZADAS
# ============================================================================

from tensorflow.keras import metrics

# Métricas disponibles
metrics=['accuracy']
metrics=['binary_accuracy']
metrics=['categorical_accuracy']
metrics=['sparse_categorical_accuracy']
metrics=['mse', 'mae']

# Múltiples métricas
metrics=['accuracy', 'Precision', 'Recall']
metrics=['accuracy', metrics.Precision(), metrics.Recall()]

# AUC
metrics=[metrics.AUC()]
metrics=[metrics.AUC(name='auc')]

# Precision y Recall
metrics=[metrics.Precision(), metrics.Recall()]

# F1 Score (custom)
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])


# ============================================================================
# 24. EJEMPLO COMPLETO - CLASIFICACIÓN
# ============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Preparar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Crear modelo
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 5. Entrenar
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 7. Predecir
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)


# ============================================================================
# 25. EJEMPLO COMPLETO - REGRESIÓN
# ============================================================================

# 1. Preparar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Crear modelo
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Sin activación para regresión
])

# 3. Compilar
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 4. Entrenar
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 5. Evaluar
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

# 6. Predecir
predictions = model.predict(X_test)
