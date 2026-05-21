class DiagnosticMessage:

    @staticmethod
    def generate(confidence, severity, damage):

        if confidence >= 0.90:
            confidence_text = (
                "La IA está muy segura "
                "del diagnóstico realizado."
            )

        elif confidence >= 0.70:
            confidence_text = (
                "La IA presenta una buena "
                "confianza en el diagnóstico."
            )

        elif confidence >= 0.50:
            confidence_text = (
                "La IA tiene una confianza "
                "moderada en el resultado."
            )

        else:
            confidence_text = (
                "La IA presenta baja confianza "
                "en el diagnóstico."
            )
#SEVERIDAD
        if severity == "SALUDABLE":

            severity_text = (
                "La hoja analizada no presenta "
                "daños significativos."
            )

        elif severity == "LEVE":

            severity_text = (
                f"La hoja presenta un daño leve "
                f"aproximado del {damage:.1f}%."
            )

        elif severity == "MEDIO":

            severity_text = (
                f"La hoja presenta un daño moderado "
                f"aproximado del {damage:.1f}%."
            )

        elif severity == "ALTO":

            severity_text = (
                f"La hoja presenta daños importantes "
                f"aproximados del {damage:.1f}%."
            )

        else:

            severity_text = (
                f"La hoja presenta daños críticos "
                f"aproximados del {damage:.1f}%."
            )

# MENSAJE FINAL
        return (
            confidence_text
            + "\n\n"
            + severity_text
        )