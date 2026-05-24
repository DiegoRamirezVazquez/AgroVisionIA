import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# =========================
# CARGAR MODELO
# =========================
model = tf.keras.models.load_model(
    "../modelos/modelo_plantas.keras"
)

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# =========================
# RUTA DE IMAGEN
# =========================
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "../test_images/strawberry_LS.JPG"

if not os.path.exists(img_path):
    print(f"Error: No se encontró la imagen '{img_path}'")
    sys.exit(1)

# =========================
# CONFIGURACIÓN
# =========================
IMG_SIZE = 224

# =========================
# CARGAR IMAGEN
# =========================
img = image.load_img(
    img_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

# =========================
# CONVERTIR A ARRAY
# =========================
img_array = image.img_to_array(img)

# NORMALIZAR
img_array = img_array / 255.0

# AGREGAR DIMENSIÓN
img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICCIÓN
# =========================
prediction = model.predict(img_array)

# OBTENER ÍNDICE
predicted_class = np.argmax(prediction)

# CONFIANZA
confidence = np.max(prediction)

# NOMBRE REAL DE LA CLASE
predicted_label = class_names[predicted_class]

# =========================
# RESULTADOS
# =========================
print("Clase predicha:", predicted_label)
print("Confianza:", confidence)

# =========================
# MOSTRAR IMAGEN
# =========================
plt.imshow(img)
plt.title(
    f"{predicted_label} ({confidence:.2f})"
)

plt.axis("off")
plt.show()