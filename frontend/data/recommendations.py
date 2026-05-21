class RecommendationEngine:
    DISEASE_DATA = {
        # APPLE
        "Apple___Apple_scab": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida foliar",
            "humidity": True,
            "isolation": False
        },

        "Apple___Black_rot": {

            "type": "hongo",
            "spread": "alta",
            "treatment": "eliminación de frutos infectados",
            "humidity": True,
            "isolation": True
        },

        "Apple___Cedar_apple_rust": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida especializado",
            "humidity": True,
            "isolation": False
        },

        # CHERRY
        "Cherry_(including_sour)___Powdery_mildew": {

            "type": "hongo",
            "spread": "media",
            "treatment": "tratamiento antifúngico",
            "humidity": True,
            "isolation": False
        },

        # CORN
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida foliar",
            "humidity": True,
            "isolation": False
        },

        "Corn_(maize)___Common_rust_": {

            "type": "hongo",
            "spread": "media",
            "treatment": "control preventivo",
            "humidity": True,
            "isolation": False
        },

        "Corn_(maize)___Northern_Leaf_Blight": {

            "type": "hongo",
            "spread": "alta",
            "treatment": "tratamiento foliar",
            "humidity": True,
            "isolation": False
        },

        # GRAPE
        "Grape___Black_rot": {

            "type": "hongo",
            "spread": "alta",
            "treatment": "fungicida sistémico",
            "humidity": True,
            "isolation": False
        },

        "Grape___Esca_(Black_Measles)": {

            "type": "hongo",
            "spread": "media",
            "treatment": "poda de zonas afectadas",
            "humidity": False,
            "isolation": False
        },

        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {

            "type": "hongo",
            "spread": "media",
            "treatment": "tratamiento foliar",
            "humidity": True,
            "isolation": False
        },

        # ORANGE
        "Orange___Haunglongbing_(Citrus_greening)": {

            "type": "bacteria",
            "spread": "alta",
            "treatment": "control de vectores",
            "humidity": False,
            "isolation": True
        },


        # PEACH
        "Peach___Bacterial_spot": {

            "type": "bacteria",
            "spread": "media",
            "treatment": "bactericida preventivo",
            "humidity": True,
            "isolation": False
        },

        # PEPPER
        "Pepper,_bell___Bacterial_spot": {

            "type": "bacteria",
            "spread": "media",
            "treatment": "bactericida agrícola",
            "humidity": True,
            "isolation": False
        },

        # POTATO
        "Potato___Early_blight": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida preventivo",
            "humidity": True,
            "isolation": False
        },

        "Potato___Late_blight": {

            "type": "hongo",
            "spread": "alta",
            "treatment": "fungicida sistémico",
            "humidity": True,
            "isolation": True
        },

        # SQUASH
        "Squash___Powdery_mildew": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida antifúngico",
            "humidity": True,
            "isolation": False
        },

        # STRAWBERRY
        "Strawberry___Leaf_scorch": {

            "type": "hongo",
            "spread": "media",
            "treatment": "control foliar",
            "humidity": True,
            "isolation": False
        },

        # TOMATE
        "Tomato___Bacterial_spot": {

            "type": "bacteria",
            "spread": "media",
            "treatment": "bactericida agrícola",
            "humidity": True,
            "isolation": False
        },

        "Tomato___Early_blight": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida preventivo",
            "humidity": True,
            "isolation": False
        },

        "Tomato___Late_blight": {

            "type": "hongo",
            "spread": "alta",
            "treatment": "fungicida sistémico",
            "humidity": True,
            "isolation": True
        },

        "Tomato___Leaf_Mold": {

            "type": "hongo",
            "spread": "media",
            "treatment": "control antifúngico",
            "humidity": True,
            "isolation": False
        },

        "Tomato___Septoria_leaf_spot": {

            "type": "hongo",
            "spread": "media",
            "treatment": "tratamiento foliar",
            "humidity": True,
            "isolation": False
        },

        "Tomato___Spider_mites Two-spotted_spider_mite": {

            "type": "ácaro",
            "spread": "media",
            "treatment": "acaricida agrícola",
            "humidity": False,
            "isolation": False
        },

        "Tomato___Target_Spot": {

            "type": "hongo",
            "spread": "media",
            "treatment": "fungicida foliar",
            "humidity": True,
            "isolation": False
        },

        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {

            "type": "virus",
            "spread": "alta",
            "treatment": "control de insectos vectores",
            "humidity": False,
            "isolation": True
        },

        "Tomato___Tomato_mosaic_virus": {

            "type": "virus",
            "spread": "alta",
            "treatment": "eliminación de plantas infectadas",
            "humidity": False,
            "isolation": True
        }
    }

    @staticmethod
    def generate(

        predicted_label,
        confidence,
        severity

    ):

        if "healthy" in predicted_label.lower():

            return (
                "El cultivo presenta un estado saludable. "
                "Se recomienda mantener monitoreo preventivo, "
                "control adecuado de humedad y buenas prácticas agrícolas."
            )

        disease = (
            RecommendationEngine.DISEASE_DATA.get(
                predicted_label
            )
        )

        if not disease:

            return (
                "No se encontró información específica "
                "para la enfermedad detectada. "
                "Se recomienda monitoreo preventivo."
            )

        disease_type = disease["type"]

        spread = disease["spread"]

        treatment = disease["treatment"]

        humidity = disease["humidity"]

        isolation = disease["isolation"]

        message = (
            f"La IA detectó una posible enfermedad "
            f"de tipo {disease_type}. "
        )

        if spread == "alta":

            message += (
                "La enfermedad presenta alta "
                "capacidad de propagación. "
            )

        elif spread == "media":

            message += (
                "La enfermedad presenta propagación moderada. "
            )

        else:

            message += (
                "La propagación estimada es baja. "
            )

        if confidence >= 0.90:

            message += (
                "La IA presenta alta seguridad "
                "en el diagnóstico. "
            )

        elif confidence >= 0.70:

            message += (
                "La IA presenta confianza moderada "
                "en el diagnóstico. "
            )

        else:

            message += (
                "Se recomienda validación adicional "
                "del cultivo. "
            )

        if severity == "CRÍTICO":

            message += (
                "El cultivo presenta un estado "
                "fitosanitario crítico. "
            )

        elif severity == "ALTO":

            message += (
                "El cultivo presenta un nivel "
                "fitosanitario elevado. "
            )

        elif severity == "MEDIO":

            message += (
                "La enfermedad parece encontrarse "
                "en una etapa moderada. "
            )

        else:

            message += (
                "La enfermedad parece encontrarse "
                "en una etapa inicial. "
            )

        if humidity:

            message += (
                "Se recomienda reducir niveles "
                "de humedad y mejorar ventilación. "
            )

        if isolation:
            message += (
                "Se recomienda aislar plantas afectadas "
                "para evitar propagación. "
            )

        message += (
            f"Tratamiento sugerido: {treatment}."
        )

        return message