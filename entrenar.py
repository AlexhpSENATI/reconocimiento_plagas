import os
import tensorflow as tf
from tensorflow.keras import layers, models


dataset_path = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

if not os.path.exists("modelo_plagas.h5"):
    print("⚙️ Entrenando nuevo modelo...")

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save("modelo_plagas.h5")
    print("✅ Modelo entrenado y guardado como modelo_plagas.h5")

else:
    print("📁 Cargando modelo ya entrenado...")
    model = tf.keras.models.load_model("modelo_plagas.h5")
    print("✅ Modelo cargado correctamente.")
