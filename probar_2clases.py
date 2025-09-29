import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("modelo_3clases.h5")

clases = ["pulgon", "mosca_blanca", "trips"]

# Cargar imagen
img = image.load_img("trips.png", target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predicciones = model.predict(img_array)[0]

print("🔎 Probabilidades:")
for i, clase in enumerate(clases):
    print(f" - {clase}: {predicciones[i]*100:.2f}%")

resultado = clases[np.argmax(predicciones)]
print("✅ Resultado final:", resultado)
