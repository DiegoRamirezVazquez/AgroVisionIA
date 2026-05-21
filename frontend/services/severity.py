class SeverityAnalyzer:

    @staticmethod
    def calculate(predicted_label, confidence):

        critical_diseases = [

            "Tomato___Late_blight",

            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

            "Tomato___Tomato_mosaic_virus",

            "Potato___Late_blight",

            "Orange___Haunglongbing_(Citrus_greening)"
        ]

        moderate_diseases = [

            "Tomato___Early_blight",

            "Tomato___Leaf_Mold",

            "Tomato___Septoria_leaf_spot",

            "Corn_(maize)___Northern_Leaf_Blight",

            "Grape___Black_rot",

            "Apple___Black_rot"
        ]

        if "healthy" in predicted_label.lower():

            return (
                "SALUDABLE",
                "#22c55e",
                0
            )

        if predicted_label in critical_diseases:

            if confidence >= 0.90:

                return (
                    "CRÍTICO",
                    "#dc2626",
                    95
                )

            elif confidence >= 0.75:

                return (
                    "ALTO",
                    "#f97316",
                    75
                )

            else:

                return (
                    "MEDIO",
                    "#facc15",
                    50
                )

        elif predicted_label in moderate_diseases:

            if confidence >= 0.90:

                return (
                    "ALTO",
                    "#f97316",
                    70
                )

            elif confidence >= 0.70:

                return (
                    "MEDIO",
                    "#facc15",
                    45
                )

            else:

                return (
                    "LEVE",
                    "#84cc16",
                    20
                )

        else:

            if confidence >= 0.85:

                return (
                    "MEDIO",
                    "#facc15",
                    40
                )

            elif confidence >= 0.65:

                return (
                    "LEVE",
                    "#84cc16",
                    15
                )

            else:

                return (
                    "BAJO",
                    "#22c55e",
                    5
                )