import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import threading
import time

from frontend.services.predictor import Predictor
from frontend.services.severity import SeverityAnalyzer
from frontend.services.diagnostic_message import DiagnosticMessage
from frontend.data.recommendations import RecommendationEngine
from frontend.services.leaf_detector import LeafDetector

class AgroVisionUI:

    def __init__(self, root):
        self.root = root
        self.predictor = Predictor()
        self.img_path = None
        self.create_interface()

    def create_interface(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.root.after(0, lambda: self.root.state("zoomed"))
        self.root.title("AgroVision IA")


# SPLASH
        self.splash = ctk.CTkFrame(self.root, fg_color="#07130d")
        self.splash.pack(fill="both", expand=True)

        logo_image = ctk.CTkImage(
            light_image=Image.open("frontend/assets/logo.png"),
            dark_image=Image.open("frontend/assets/logo.png"),
            size=(380, 380)
        )

        self.logo_label = ctk.CTkLabel(self.splash, image=logo_image, text="")
        self.logo_label.pack(pady=70)

        splash_title = ctk.CTkLabel(
            self.splash,
            text="AgroVision IA",
            font=("Arial", 42, "bold"),
            text_color="#8bffb0"
        )
        splash_title.pack()

        subtitle = ctk.CTkLabel(
            self.splash,
            text="Sistema Inteligente de Diagnóstico Agrícola",
            font=("Arial", 18),
            text_color="white"
        )
        subtitle.pack(pady=10)

        self.main_frame = ctk.CTkFrame(self.root, fg_color="#020617")

# HEADER
        header = ctk.CTkFrame(self.main_frame, fg_color="#0f172a", height=90, corner_radius=20)
        header.pack(fill="x", padx=20, pady=(20, 10))
        header.pack_propagate(False)

        title = ctk.CTkLabel(
            header,
            text="AgroVision IA",
            font=("Arial", 38, "bold"),
            text_color="#8bffb0"
        )
        title.place(relx=0.5, rely=0.5, anchor="center")

# DASHBOARD CONTENEDOR
        dashboard = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        dashboard.pack(fill="both", expand=True, padx=20, pady=(0, 20))


# PANEL IZQUIERDO
        left_column = ctk.CTkFrame(dashboard, fg_color="transparent")
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Imagen
        left_panel = ctk.CTkFrame(left_column, fg_color="#0f172a", corner_radius=25)
        left_panel.pack(fill="both", expand=True, pady=(0, 15))

        left_title = ctk.CTkLabel(
            left_panel,
            text="Imagen Analizada",
            font=("Arial", 26, "bold"),
            text_color="white"
        )
        left_title.pack(pady=(20, 10))

        image_container = ctk.CTkFrame(left_panel, width=460, height=440, fg_color="#1e293b", corner_radius=20)
        image_container.pack(pady=10, expand=True)
        image_container.pack_propagate(False)

        self.image_label = ctk.CTkLabel(
            image_container,
            text="Sin imagen",
            font=("Arial", 22),
            text_color="gray"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        # Contenedor de Botones
        button_frame = ctk.CTkFrame(left_column, fg_color="#0f172a", height=90, corner_radius=25)
        button_frame.pack(fill="x")
        button_frame.pack_propagate(False)

        # Botón Subir
        upload_button = ctk.CTkButton(
            button_frame,
            text="Subir Imagen",
            width=200,
            height=50,
            corner_radius=15,
            font=("Arial", 16, "bold"),
            fg_color="#16a34a",
            hover_color="#15803d",
            command=self.upload_image
        )
        upload_button.place(relx=0.28, rely=0.5, anchor="center")

        # Botón Analizar
        predict_button = ctk.CTkButton(
            button_frame,
            text="Analizar Enfermedad",
            width=220,
            height=50,
            corner_radius=15,
            font=("Arial", 16, "bold"),
            fg_color="#22c55e",
            hover_color="#16a34a",
            command=self.predict_image
        )
        predict_button.place(relx=0.72, rely=0.5, anchor="center")

# PANEL DERECHO
        right_panel = ctk.CTkFrame(dashboard, fg_color="#0f172a", corner_radius=25)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.result_title = ctk.CTkLabel(
            right_panel,
            text="Esperando análisis...",
            font=("Arial", 28, "bold"),
            text_color="#8bffb0"
        )
        self.result_title.pack(pady=(30, 15))

        confidence_text = ctk.CTkLabel(
            right_panel,
            text="Nivel de confianza de AgroVisionIA",
            font=("Arial", 16, "bold"),
            text_color="#94a3b8"
        )
        confidence_text.pack(pady=(0, 5))

        # Contenedor Barra de Confianza
        bar_container = ctk.CTkFrame(right_panel, fg_color="transparent")
        bar_container.pack(fill="x", padx=50, pady=(0, 20))

        self.confidence_bar = ctk.CTkProgressBar(
            bar_container,
            height=14,
            progress_color="#22c55e",
            fg_color="#1e293b"
        )
        self.confidence_bar.pack(side="left", fill="x", expand=True, padx=(0, 15))
        self.confidence_bar.set(0)

        self.confidence_label = ctk.CTkLabel(
            bar_container,
            text="0%",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        self.confidence_label.pack(side="right")

        # Estado
        self.severity_label = ctk.CTkLabel(
            right_panel,
            text="Estado",
            font=("Arial", 22, "bold"),
            text_color="#22c55e"
        )
        self.severity_label.pack(pady=(0, 25))

        # Sección: Recomendaciones
        recommendation_title = ctk.CTkLabel(
            right_panel,
            text="Recomendación de AgroVisionIA",
            font=("Arial", 18, "bold"),
            text_color="#8bffb0"
        )
        recommendation_title.pack(anchor="w", padx=50, pady=(0, 8))

        self.recommendation_box = ctk.CTkTextbox(
            right_panel,
            height=110,
            corner_radius=15,
            fg_color="#0b132b",
            text_color="white",
            font=("Arial", 15),
            wrap="word",
            border_width=0
        )
        self.recommendation_box.pack(fill="x", padx=50, pady=(0, 20))
        self.recommendation_box.insert("0.0", "Las recomendaciones aparecerán aquí...")

        # Sección: Interpretación
        message_title = ctk.CTkLabel(
            right_panel,
            text="Interpretación Inteligente",
            font=("Arial", 18, "bold"),
            text_color="#8bffb0"
        )
        message_title.pack(anchor="w", padx=50, pady=(0, 8))

        self.message_box = ctk.CTkTextbox(
            right_panel,
            height=110,
            corner_radius=15,
            fg_color="#0b132b",
            text_color="#d1fae5",
            font=("Arial", 15),
            wrap="word",
            border_width=0
        )
        self.message_box.pack(fill="x", padx=50, pady=(0, 15))
        self.message_box.insert("0.0", "La interpretación inteligente aparecerá aquí...")

        # Loading Label
        self.loading_label = ctk.CTkLabel(
            right_panel,
            text="",
            font=("Arial", 15),
            text_color="#8bffb0"
        )
        self.loading_label.pack(pady=(0, 10))

        # MOSTRAR APP
        self.root.after(3000, self.show_main)

    # MOSTRAR MAIN
    def show_main(self):
        self.splash.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    # SUBIR IMAGEN
    def upload_image(self):
        self.img_path = filedialog.askopenfilename()
        if not self.img_path:
            return

        img = Image.open(self.img_path)
        img = img.resize((440, 420))

        img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(440, 420))
        self.image_label.configure(image=img_tk, text="")

    # ANALISIS
    def analyze(self):
        if not self.img_path:
            return

        self.loading_label.configure(text="Analizando imagen con IA...")
        time.sleep(1.5)

        if not LeafDetector.is_leaf(self.img_path):
            self.result_title.configure(
                text="Imagen no válida",
                text_color="#ef4444"
            )

            self.confidence_bar.set(0)

            self.confidence_label.configure(
                text="0%"
            )

            self.severity_label.configure(
                text="Sin diagnóstico",
                text_color="#ef4444"
            )

            self.recommendation_box.delete(
                "0.0",
                "end"
            )

            self.recommendation_box.insert(
                "0.0",
                "La imagen analizada no parece "
                "corresponder a una hoja vegetal "
                "compatible con AgroVisionIA."
            )

            self.message_box.delete(
                "0.0",
                "end"
            )

            self.message_box.insert(
                "0.0",
                "No se detectaron patrones "
                "visuales compatibles con hojas "
                "utilizadas durante el entrenamiento "
                "del modelo."
            )

            self.loading_label.configure(
                text=""
            )

            return


        result = self.predictor.predict(self.img_path)
        level, color, damage_percentage = SeverityAnalyzer.calculate(
            result["raw_label"], result["confidence"]
        )
        diagnostic_message = DiagnosticMessage.generate(
            result["confidence"], level, damage_percentage
        )

        # RESULTADOS
        self.result_title.configure(text=result["label"])
        self.confidence_bar.set(result["confidence"])
        self.confidence_label.configure(text=f'{result["confidence"]:.2%}')
        self.severity_label.configure(text=f"RIESGO {level}", text_color=color)

        # RECOMENDACIONES
        recommendation = RecommendationEngine.generate(
            result["raw_label"], result["confidence"], level
        )
        self.recommendation_box.delete("0.0", "end")
        self.recommendation_box.insert("0.0", recommendation)

        # INTERPRETACION
        self.message_box.delete("0.0", "end")
        self.message_box.insert("0.0", diagnostic_message)
        self.loading_label.configure(text="")

    def predict_image(self):
        threading.Thread(target=self.analyze).start()