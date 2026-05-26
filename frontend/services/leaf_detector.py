import cv2
import numpy as np


class LeafDetector:

    @staticmethod
    def is_leaf(img_path):

        # =====================================================
        # CARGAR IMAGEN
        # =====================================================
        img = cv2.imread(img_path)

        if img is None:
            return False

        # =====================================================
        # RESIZE
        # =====================================================
        img = cv2.resize(
            img,
            (400, 400)
        )

        # =====================================================
        # HSV
        # =====================================================
        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        # =====================================================
        # RANGO VERDE
        # =====================================================
        lower_green = np.array(
            [25, 40, 40]
        )

        upper_green = np.array(
            [95, 255, 255]
        )

        # =====================================================
        # MASCARA VERDE
        # =====================================================
        mask_green = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # =====================================================
        # RANGO AMARILLO (hojas enfermas/secas)
        # =====================================================
        lower_yellow = np.array(
            [15, 40, 40]
        )

        upper_yellow = np.array(
            [25, 255, 255]
        )

        mask_yellow = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        # =====================================================
        # RANGO MARRÓN (hojas muy afectadas)
        # =====================================================
        lower_brown = np.array(
            [8, 30, 30]
        )

        upper_brown = np.array(
            [20, 200, 180]
        )

        mask_brown = cv2.inRange(
            hsv,
            lower_brown,
            upper_brown
        )

        # =====================================================
        # COMBINAR MASCARAS
        # =====================================================
        mask = cv2.bitwise_or(mask_green, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_brown)

        # =====================================================
        # PORCENTAJE FOLIAR
        # =====================================================
        leaf_pixels = cv2.countNonZero(
            mask
        )

        total_pixels = (
            img.shape[0]
            * img.shape[1]
        )

        leaf_percentage = (
            leaf_pixels / total_pixels
        ) * 100

        # =====================================================
        # SI HAY MUY POCO COLOR FOLIAR
        # =====================================================
        if leaf_percentage < 12:

            return False

        # =====================================================
        # SUAVIZADO
        # =====================================================
        blurred = cv2.GaussianBlur(
            mask,
            (7, 7),
            0
        )

        # =====================================================
        # CONTORNOS
        # =====================================================
        contours, _ = cv2.findContours(
            blurred,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # =====================================================
        # VALIDAR CONTORNOS
        # =====================================================
        large_contours = 0

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            # ================================================
            # CONTORNO GRANDE
            # ================================================
            if area > 5000:

                # ============================================
                # RECTANGULO
                # ============================================
                x, y, w, h = cv2.boundingRect(
                    contour
                )

                # ============================================
                # RELACION ASPECTO
                # ============================================
                aspect_ratio = w / h

                # ============================================
                # FORMAS TIPO HOJA
                # ============================================
                if 0.3 < aspect_ratio < 3.0:

                    large_contours += 1

        # =====================================================
        # DECISION FINAL
        # =====================================================
        if large_contours == 0:

            return False

        return True

    @staticmethod
    def extract_visual_features(img_path):
        """
        Extrae características visuales HSV de la imagen.
        Devuelve un dict con ratios de color y métricas de calidad.

        Retorna:
        {
            "leaf_area_percent": float,  # % de píxeles detectados como hoja
            "green_ratio":       float,  # tejido sano
            "yellow_ratio":      float,  # clorosis / estrés
            "brown_ratio":       float,  # necrosis / daño
            "damage_percent":    float,  # amarillo + café (relativo a hoja)
            "image_quality":     float,  # nitidez estimada 0-1
        }
        """
        _empty = {
            "leaf_area_percent": 0.0,
            "green_ratio":       0.0,
            "yellow_ratio":      0.0,
            "brown_ratio":       0.0,
            "damage_percent":    0.0,
            "image_quality":     0.0,
        }

        img = cv2.imread(img_path)
        if img is None:
            return _empty

        img = cv2.resize(img, (400, 400))
        total_pixels = img.shape[0] * img.shape[1]   # 160 000

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ── Máscaras de color ──────────────────────────────────────────────
        mask_green  = cv2.inRange(hsv, np.array([25, 40,  40]),
                                       np.array([95, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([15, 40,  40]),
                                       np.array([25, 255, 255]))
        mask_brown  = cv2.inRange(hsv, np.array([ 8, 30,  30]),
                                       np.array([20, 200, 180]))

        mask_leaf = cv2.bitwise_or(mask_green,  mask_yellow)
        mask_leaf = cv2.bitwise_or(mask_leaf,   mask_brown)

        leaf_px   = cv2.countNonZero(mask_leaf)
        green_px  = cv2.countNonZero(mask_green)
        yellow_px = cv2.countNonZero(mask_yellow)
        brown_px  = cv2.countNonZero(mask_brown)

        leaf_area_pct = (leaf_px / total_pixels) * 100.0

        # Ratios relativos al área foliar detectada
        base = max(leaf_px, 1)
        green_ratio  = green_px  / base
        yellow_ratio = yellow_px / base
        brown_ratio  = brown_px  / base
        damage_pct   = ((yellow_px + brown_px) / base) * 100.0

        # ── Nitidez: varianza del Laplaciano ──────────────────────────────
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality = min(1.0, lap_var / 500.0)

        return {
            "leaf_area_percent": round(leaf_area_pct,  2),
            "green_ratio":       round(green_ratio,    3),
            "yellow_ratio":      round(yellow_ratio,   3),
            "brown_ratio":       round(brown_ratio,    3),
            "damage_percent":    round(damage_pct,     2),
            "image_quality":     round(quality,        3),
        }
