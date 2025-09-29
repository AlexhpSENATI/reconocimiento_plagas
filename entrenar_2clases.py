import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
#  Data Augmentation
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    validation_split=0.2  
)

# Dataset de entrenamiento
train_ds = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',  
    subset='training'
)

# Dataset de validación
val_ds = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ---------------------------
#  Cargar MobileNetV2 preentrenado
# ---------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,    
    weights='imagenet'
)
base_model.trainable = False  

# ---------------------------
#  Construir el modelo final
# ---------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  
])

# ---------------------------
#  Compilar modelo
# ---------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
#  Entrenar modelo
# ---------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# ---------------------------
#  Guardar modelo
# ---------------------------
model.save("modelo_3clases.h5")
print("✅ Modelo entrenado y guardado como modelo_3clases.h5")
