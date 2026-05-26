"""
AgroVisionIA — Servicio Climático
==================================
Obtiene ubicación aproximada por IP (nivel de ciudad) y consulta
el clima actual en Open-Meteo (sin API key requerida).

No se almacena ninguna coordenada ni dato personal.

Flujo:
    1. get_location_by_ip()        → lat, lon, city, country
    2. fetch_weather(lat, lon)     → temperatura, humedad, precipitación, viento
    3. calculate_climate_risk(...) → risk_level, factors, conclusion
"""

import urllib.request
import json


class WeatherService:

    # ── URLs externas ─────────────────────────────────────────────────────────
    OPENMETEO_URL = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m"
        ",precipitation,wind_speed_10m"
        "&wind_speed_unit=kmh"
        "&forecast_days=1"
    )

    IPAPI_URL = "http://ip-api.com/json/?fields=lat,lon,city,country,status"

    # =========================================================================
    # GEOLOCALIZACIÓN POR IP
    # =========================================================================

    @staticmethod
    def get_location_by_ip(timeout=6):
        """
        Devuelve ubicación aproximada (nivel ciudad) a partir de la IP pública.
        No requiere permiso del usuario; la precisión es de ciudad, no de GPS.

        Retorna:
            {"lat": float, "lon": float, "city": str, "country": str}

        Lanza:
            RuntimeError si la consulta falla.
        """
        try:
            req  = urllib.request.urlopen(WeatherService.IPAPI_URL, timeout=timeout)
            data = json.loads(req.read().decode("utf-8"))
            if data.get("status") == "success":
                return {
                    "lat":     float(data["lat"]),
                    "lon":     float(data["lon"]),
                    "city":    data.get("city",    "—"),
                    "country": data.get("country", "—"),
                }
            raise RuntimeError("La geolocalización por IP no retornó resultados válidos.")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"No se pudo determinar la ubicación: {exc}") from exc

    # =========================================================================
    # CONSULTA DE CLIMA — OPEN-METEO
    # =========================================================================

    @staticmethod
    def fetch_weather(lat, lon, timeout=8):
        """
        Consulta Open-Meteo (sin API key requerida) y retorna condiciones actuales.

        Retorna:
            {
                "temperature_2m":       float | None,
                "relative_humidity_2m": float | None,
                "precipitation":        float,
                "wind_speed_10m":       float,
            }

        Lanza:
            RuntimeError si la consulta falla.
        """
        url = WeatherService.OPENMETEO_URL.format(lat=lat, lon=lon)
        try:
            req     = urllib.request.urlopen(url, timeout=timeout)
            data    = json.loads(req.read().decode("utf-8"))
            current = data.get("current", {})
            return {
                "temperature_2m":       current.get("temperature_2m"),
                "relative_humidity_2m": current.get("relative_humidity_2m"),
                "precipitation":        current.get("precipitation",  0.0) or 0.0,
                "wind_speed_10m":       current.get("wind_speed_10m", 0.0) or 0.0,
            }
        except Exception as exc:
            raise RuntimeError(f"No se pudo obtener el clima: {exc}") from exc

    # =========================================================================
    # LÓGICA DE RIESGO CLIMÁTICO
    # =========================================================================

    @staticmethod
    def calculate_climate_risk(weather, raw_label):
        """
        Evalúa el riesgo de propagación de la enfermedad detectada
        en función de las condiciones climáticas actuales.

        Parámetros:
            weather   : dict de fetch_weather()
            raw_label : etiqueta cruda del modelo (ej. "Tomato___Late_blight")

        Retorna:
            {
                "risk_level": "Bajo" | "Medio" | "Alto" | "—",
                "factors":    [str, ...],
                "conclusion": str,
            }
        """
        temp   = weather.get("temperature_2m")
        hum    = weather.get("relative_humidity_2m")
        precip = weather.get("precipitation",  0.0) or 0.0
        wind   = weather.get("wind_speed_10m", 0.0) or 0.0

        if temp is None or hum is None:
            return {
                "risk_level": "—",
                "factors":    [],
                "conclusion": (
                    "No hay suficientes datos climáticos disponibles "
                    "para estimar el nivel de riesgo de propagación."
                ),
            }

        lbl          = raw_label.lower()
        is_fungal    = any(k in lbl for k in [
            "blight", "mildew", "spot", "rust", "mold",
            "rot", "esca", "scorch", "measles", "haunglongbing",
        ])
        is_bacterial = "bacterial" in lbl
        is_viral     = any(k in lbl for k in ["virus", "mosaic", "curl"])
        is_healthy   = "healthy" in lbl

        factors    = []
        risk_level = "Bajo"

        # ── Reglas de riesgo ──────────────────────────────────────────────────

        if hum >= 80 and 18 <= temp <= 28:
            risk_level = "Alto"
            factors.append(
                f"Humedad muy alta ({int(hum)}%) dentro del rango térmico "
                f"óptimo para hongos ({temp:.0f}°C)"
            )
        elif hum >= 70 and precip > 0:
            risk_level = "Alto"
            factors.append(
                f"Humedad elevada ({int(hum)}%) con precipitación activa "
                f"({precip:.1f} mm)"
            )
        elif temp > 30 and hum < 60:
            risk_level = "Medio"
            factors.append(
                f"Temperatura alta ({temp:.0f}°C) con baja humedad — "
                "posible estrés hídrico en la planta"
            )
        elif hum >= 70:
            risk_level = "Medio"
            factors.append(f"Humedad moderadamente alta ({int(hum)}%)")

        if wind > 20:
            factors.append(
                f"Viento considerable ({wind:.0f} km/h) — "
                "puede dispersar esporas o patógenos entre plantas"
            )

        # ── Conclusión adaptada a la enfermedad ───────────────────────────────

        if is_healthy:
            if risk_level == "Alto":
                conclusion = (
                    "Aunque la planta aparece saludable, la alta humedad actual "
                    "crea condiciones favorables para el desarrollo de hongos y bacterias. "
                    "Se recomienda realizar inspecciones preventivas con mayor frecuencia "
                    "y evitar exceso de humedad en el follaje."
                )
            else:
                conclusion = (
                    "La planta parece saludable y las condiciones climáticas actuales "
                    "son relativamente seguras. Mantén un monitoreo preventivo regular "
                    "y asegura buena ventilación entre plantas."
                )

        elif is_fungal:
            if risk_level == "Alto":
                conclusion = (
                    "Las condiciones actuales de alta humedad y temperatura son ideales "
                    "para la propagación de enfermedades fúngicas. Se recomienda revisar "
                    "plantas cercanas, mejorar la ventilación del cultivo y considerar "
                    "la aplicación de fungicidas preventivos si hay más plantas afectadas."
                )
            elif risk_level == "Medio":
                conclusion = (
                    "El clima presenta riesgo moderado para la propagación de hongos. "
                    "Monitorea otras plantas del cultivo, evita riegos nocturnos y reduce "
                    "la humedad del follaje para limitar la dispersión."
                )
            else:
                conclusion = (
                    "El clima actual no favorece especialmente la dispersión de hongos. "
                    "Aplica el tratamiento correspondiente en la planta afectada y "
                    "mantén vigilancia sobre el resto del cultivo."
                )

        elif is_bacterial:
            if precip > 0 or hum >= 70:
                conclusion = (
                    "La lluvia y la alta humedad pueden dispersar bacterias mediante "
                    "salpicaduras de agua entre plantas. Evita manipular el cultivo "
                    "durante lluvias, revisa el sistema de drenaje y considera "
                    "bactericidas cúpricos si la enfermedad continúa avanzando."
                )
            else:
                conclusion = (
                    "Las condiciones actuales presentan riesgo moderado para enfermedades "
                    "bacterianas. Trata la planta afectada, evita mojar el follaje al "
                    "regar y elimina hojas infectadas para reducir el inóculo."
                )

        elif is_viral:
            conclusion = (
                "Los virus en plantas generalmente se transmiten por insectos vectores "
                "como pulgones o mosca blanca. Monitorea la presencia de estos insectos "
                "y considera tratamientos insecticidas si la presión de plagas es alta. "
                "El clima cálido y húmedo puede favorecer la reproducción de vectores."
            )

        else:
            conclusion = (
                "Considera las condiciones climáticas actuales al planificar el tratamiento. "
                "La temperatura y humedad pueden influir en la efectividad de los productos "
                "aplicados y en la velocidad de propagación de la enfermedad detectada."
            )

        return {
            "risk_level": risk_level,
            "factors":    factors,
            "conclusion": conclusion,
        }
