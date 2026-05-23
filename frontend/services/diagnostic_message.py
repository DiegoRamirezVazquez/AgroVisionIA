class DiagnosticMessage:

    @staticmethod
    def generate(confidence, severity, damage):

        # ============================================
        # CONFIANZA
        # ============================================
        if confidence >= 0.90:

            confidence_text = (

                "El modelo presenta un nivel "
                "elevado de confianza en el "
                "diagnóstico realizado."
            )

        elif confidence >= 0.70:

            confidence_text = (

                "El diagnóstico presenta un "
                "nivel de confianza favorable "
                "según el análisis realizado."
            )

        elif confidence >= 0.50:

            confidence_text = (

                "El resultado obtenido presenta "
                "una confianza moderada, por lo "
                "que se recomienda monitoreo adicional."
            )

        else:

            confidence_text = (

                "El nivel de confianza del modelo "
                "es reducido, por lo que se recomienda "
                "validar el diagnóstico manualmente."
            )

        # ============================================
        # SEVERIDAD
        # ============================================
        if severity == "SALUDABLE":

            severity_text = (

                "No se observan signos visuales "
                "importantes de afectación en la hoja analizada."
            )

        elif severity == "LEVE":

            severity_text = (

                f"Se observan indicios leves "
                f"de afectación foliar "
                f"aproximados del {damage:.1f}%."
            )

        elif severity == "MEDIO":

            severity_text = (

                f"La hoja presenta síntomas "
                f"moderados de afectación visual "
                f"aproximados del {damage:.1f}%."
            )

        elif severity == "ALTO":

            severity_text = (

                f"Se identifican daños importantes "
                f"en la superficie foliar "
                f"aproximados del {damage:.1f}%."
            )

        else:

            severity_text = (

                f"La hoja presenta una afectación "
                f"visual crítica aproximada "
                f"del {damage:.1f}%."
            )

        # ============================================
        # MENSAJE FINAL
        # ============================================
        return (

            confidence_text
            + "\n\n"
            + severity_text
        )