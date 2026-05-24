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
