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
        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # =====================================================
        # PORCENTAJE VERDE
        # =====================================================
        green_pixels = cv2.countNonZero(
            mask
        )

        total_pixels = (
            img.shape[0]
            * img.shape[1]
        )

        green_percentage = (
            green_pixels / total_pixels
        ) * 100

        # =====================================================
        # SI HAY MUY POCO VERDE
        # =====================================================
        if green_percentage < 12:

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
