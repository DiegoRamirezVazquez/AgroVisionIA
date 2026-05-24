import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

from frontend.data.class_names import class_names
from frontend.data.translations import translations


class Predictor:

    IMG_SIZE = 224

    def __init__(self):

        self.model = tf.keras.models.load_model(
            "modelos/modelo_plantas.keras"
        )

    def is_loaded(self):

        return self.model is not None

    def predict(self, img_path):

        img = image.load_img(
            img_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE)
        )

        img_array = image.img_to_array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = self.model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = float(np.max(prediction))

        predicted_label = class_names[predicted_class]

        translated = translations.get(
            predicted_label,
            predicted_label
        )


        return {
            "label": translated,
            "raw_label": predicted_label,
            "confidence": confidence,
        }