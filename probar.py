import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

model = tf.keras.models.load_model("modelo_plagas.h5")

img = load_img("model03.png", target_size=(128, 128))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predicciones = model.predict(img_array)
clases = ["pulgon", "mosca_blanca"]  
resultado = clases[np.argmax(predicciones)]

print("✅ Resultado:", resultado)
for clase, prob in zip(clases, predicciones[0]):
    print(f"{clase}: {prob * 100:.2f}%")
