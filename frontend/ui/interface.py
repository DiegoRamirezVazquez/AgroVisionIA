"""
AgroVisionIA — Interfaz de usuario
Dashboard moderno CustomTkinter
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import os

from frontend.services.predictor import Predictor
from frontend.services.severity import SeverityAnalyzer
from frontend.services.diagnostic_message import DiagnosticMessage
from frontend.data.recommendations import RecommendationEngine
from frontend.services.leaf_detector import LeafDetector
from frontend.services.fuzzy_corrector import FuzzyCorrector
from frontend.services.weather_service import WeatherService

# ── Paleta de colores ─────────────────────────────────────────────────────────
BG_MAIN   = "#020617"
BG_CARD   = "#0f172a"
BG_CARD2  = "#111827"
BG_INSET  = "#1e293b"
TEXT_MAIN = "#f8fafc"
TEXT_SEC  = "#94a3b8"
ACCENT    = "#8bffb0"
GREEN     = "#22c55e"
YELLOW    = "#facc15"
ORANGE    = "#f97316"
BLUE      = "#60a5fa"
RED       = "#ef4444"

DECISION_COLOR = {
    "CONFIRMADO":     GREEN,
    "POSIBLE":        YELLOW,
    "CORREGIDO":      BLUE,
    "NO_CONCLUYENTE": ORANGE,
}

DECISION_BG = {
    "CONFIRMADO":     "#052e16",
    "POSIBLE":        "#451a03",
    "CORREGIDO":      "#172554",
    "NO_CONCLUYENTE": "#431407",
}

DECISION_LABEL = {
    "CONFIRMADO":     "✓  CONFIRMADO",
    "POSIBLE":        "~  POSIBLE",
    "CORREGIDO":      "↻  AJUSTADO",
    "NO_CONCLUYENTE": "?  NO CONCLUYENTE",
}

# Metadatos para las barras visuales: (clave, nombre UI, escala_max)
VISUAL_FEATURES_META = [
    ("leaf_area_percent", "Área foliar",   100),
    ("green_ratio",       "Verde",           1),
    ("yellow_ratio",      "Amarillo",        1),
    ("brown_ratio",       "Café / Marrón",   1),
    ("damage_percent",    "Daño estimado", 100),
    ("image_quality",     "Calidad imagen",  1),
]

VIS_BAR_COLOR = {
    "leaf_area_percent": BLUE,
    "green_ratio":       GREEN,
    "yellow_ratio":      YELLOW,
    "brown_ratio":       "#a16207",
    "damage_percent":    ORANGE,
    "image_quality":     TEXT_SEC,
}

# Colores del indicador de riesgo climático
RISK_COLOR = {"Bajo": GREEN,  "Medio": YELLOW, "Alto": RED,      "—": TEXT_SEC}
RISK_BG    = {"Bajo": "#052e16", "Medio": "#451a03", "Alto": "#450a0a", "—": BG_INSET}


class AgroVisionUI:

    def __init__(self, root):
        self.root      = root
        self.img_path  = None
        self.img_tk    = None
        self.analyzing = False

        # Estado del último análisis
        self.last_top_predictions = None
        self.last_visual_features = None
        self.last_correction      = None

        # Estado climático (no se persiste entre sesiones)
        self.weather_data      = None
        self.weather_location  = None
        self.climate_analyzing = False

        # Servicios
        try:
            self.predictor = Predictor()
        except Exception as e:
            self.predictor = None
            root.after(100, lambda: messagebox.showerror(
                "Error de modelo",
                f"No se pudo cargar el modelo de IA.\n\n"
                f"Verifica que el archivo 'modelos/modelo_plantas.keras' "
                f"exista y no esté corrupto.\n\nDetalle: {e}"
            ))

        self.fuzzy_corrector = FuzzyCorrector()
        self.create_interface()

    # =========================================================================
    # INTERFAZ PRINCIPAL
    # =========================================================================

    def create_interface(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.root.after(0, lambda: self.root.state("zoomed"))
        self.root.title("AgroVision IA")
        self.root.configure(fg_color=BG_MAIN)

        self._build_splash()

        self.main_frame = ctk.CTkFrame(self.root, fg_color=BG_MAIN)
        self._build_dashboard()

        self.root.after(3000, self._show_main)

    def _build_splash(self):
        self.splash = ctk.CTkFrame(self.root, fg_color="#07130d")
        self.splash.pack(fill="both", expand=True)

        try:
            logo = ctk.CTkImage(
                light_image=Image.open("frontend/assets/logo.png"),
                dark_image=Image.open("frontend/assets/logo.png"),
                size=(310, 310)
            )
            ctk.CTkLabel(self.splash, image=logo, text="").pack(pady=55)
        except Exception:
            ctk.CTkLabel(
                self.splash, text="🌿",
                font=("Arial", 80)
            ).pack(pady=55)

        ctk.CTkLabel(
            self.splash, text="AgroVision IA",
            font=("Arial", 44, "bold"), text_color=ACCENT
        ).pack()
        ctk.CTkLabel(
            self.splash,
            text="Sistema Inteligente de Diagnóstico Agrícola",
            font=("Arial", 17), text_color="#b0c4b1"
        ).pack(pady=10)

    def _show_main(self):
        self.splash.pack_forget()
        self.main_frame.pack(fill="both", expand=True)

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def _build_dashboard(self):
        mf = self.main_frame

        # ── Header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(mf, fg_color=BG_CARD, height=68, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🌿  AgroVision IA",
            font=("Arial", 28, "bold"), text_color=ACCENT
        ).place(relx=0.5, rely=0.5, anchor="center")

        # ── Body ──────────────────────────────────────────────────────────────
        body = ctk.CTkFrame(mf, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left = ctk.CTkFrame(body, fg_color="transparent", width=455)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        right = ctk.CTkScrollableFrame(
            body, fg_color="transparent",
            scrollbar_button_color=BG_INSET,
            scrollbar_button_hover_color="#334155",
        )
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    # ── Panel izquierdo ───────────────────────────────────────────────────────

    def _build_left(self, parent):
        # Card: imagen
        img_card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=20)
        img_card.pack(fill="both", expand=True, pady=(0, 10))

        ctk.CTkLabel(
            img_card, text="Imagen analizada",
            font=("Arial", 15, "bold"), text_color=TEXT_SEC
        ).pack(pady=(14, 6))

        img_box = ctk.CTkFrame(
            img_card, fg_color=BG_INSET,
            width=415, height=385, corner_radius=16
        )
        img_box.pack(padx=12, pady=(0, 12), expand=True)
        img_box.pack_propagate(False)

        self.image_label = ctk.CTkLabel(
            img_box, text="Sin imagen",
            font=("Arial", 16), text_color="#475569"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        # Card: botones
        btn_card = ctk.CTkFrame(parent, fg_color=BG_CARD, height=72, corner_radius=16)
        btn_card.pack(fill="x")
        btn_card.pack_propagate(False)

        ctk.CTkButton(
            btn_card, text="📁  Subir imagen",
            width=190, height=44, corner_radius=12,
            font=("Arial", 13, "bold"),
            fg_color="#166534", hover_color="#14532d",
            command=self.upload_image
        ).place(relx=0.27, rely=0.5, anchor="center")

        ctk.CTkButton(
            btn_card, text="🔍  Analizar",
            width=205, height=44, corner_radius=12,
            font=("Arial", 13, "bold"),
            fg_color=GREEN, hover_color="#16a34a",
            text_color="#021a0a",
            command=self.predict_image
        ).place(relx=0.73, rely=0.5, anchor="center")

    # ── Panel derecho (scrollable) ────────────────────────────────────────────

    def _build_right(self, parent):

        # ── Card 1: Diagnóstico principal ─────────────────────────────────────
        c1 = self._card(parent, "Diagnóstico principal")

        # Badge de decisión
        badge_wrap = ctk.CTkFrame(c1, fg_color="transparent")
        badge_wrap.pack(fill="x", padx=20, pady=(0, 8))

        self.badge_frame = ctk.CTkFrame(
            badge_wrap, fg_color=BG_INSET,
            corner_radius=20, height=30
        )
        self.badge_frame.pack(side="left")
        self.badge_frame.pack_propagate(False)

        self.badge_label = ctk.CTkLabel(
            self.badge_frame, text="—  Esperando",
            font=("Arial", 11, "bold"), text_color=TEXT_SEC,
            padx=14
        )
        self.badge_label.pack(side="left", fill="y")

        # Loading spinner text
        self.loading_label = ctk.CTkLabel(
            badge_wrap, text="",
            font=("Arial", 11), text_color=ACCENT
        )
        self.loading_label.pack(side="right", padx=(0, 4))

        # Nombre del diagnóstico
        self.result_title = ctk.CTkLabel(
            c1, text="Esperando análisis...",
            font=("Arial", 22, "bold"), text_color=ACCENT,
            wraplength=580, anchor="w"
        )
        self.result_title.pack(fill="x", padx=20, pady=(0, 6))

        # Fila confianza
        conf_row = ctk.CTkFrame(c1, fg_color="transparent")
        conf_row.pack(fill="x", padx=20, pady=(0, 4))

        ctk.CTkLabel(
            conf_row, text="Confianza híbrida",
            font=("Arial", 12), text_color=TEXT_SEC
        ).pack(side="left")

        self.confidence_label = ctk.CTkLabel(
            conf_row, text="—",
            font=("Arial", 13, "bold"), text_color=TEXT_MAIN
        )
        self.confidence_label.pack(side="right")

        self.confidence_bar = ctk.CTkProgressBar(
            c1, height=10,
            progress_color=GREEN, fg_color=BG_INSET
        )
        self.confidence_bar.pack(fill="x", padx=20, pady=(0, 8))
        self.confidence_bar.set(0)

        # Nivel de riesgo
        self.severity_label = ctk.CTkLabel(
            c1, text="Estado: —",
            font=("Arial", 15, "bold"), text_color=TEXT_SEC
        )
        self.severity_label.pack(padx=20, pady=(0, 14), anchor="w")

        # ── Card 2: Top-3 CNN ─────────────────────────────────────────────────
        c2 = self._card(parent, "🧠  Top 3 — Red neuronal")

        TOP3_COLORS = [GREEN, YELLOW, TEXT_SEC]
        self.top3_name_labels = []
        self.top3_bars        = []
        self.top3_pct_labels  = []

        for i in range(3):
            name_lbl = ctk.CTkLabel(
                c2, text=f"{i+1}.  —",
                font=("Arial", 12), text_color=TOP3_COLORS[i],
                anchor="w"
            )
            name_lbl.pack(fill="x", padx=20, pady=(4, 0))
            self.top3_name_labels.append(name_lbl)

            row = ctk.CTkFrame(c2, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=(2, 8))

            bar = ctk.CTkProgressBar(
                row, height=8,
                progress_color=BG_INSET, fg_color=BG_INSET
            )
            bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
            bar.set(0)
            self.top3_bars.append(bar)

            pct = ctk.CTkLabel(
                row, text="—",
                font=("Arial", 12, "bold"), text_color=TOP3_COLORS[i],
                width=50, anchor="e"
            )
            pct.pack(side="right")
            self.top3_pct_labels.append(pct)

        # ── Card 3: Análisis visual ───────────────────────────────────────────
        c3 = self._card(parent, "🔬  Análisis visual de la hoja")

        self.vis_bars   = []
        self.vis_labels = []

        for key, name, scale in VISUAL_FEATURES_META:
            lbl_row = ctk.CTkFrame(c3, fg_color="transparent")
            lbl_row.pack(fill="x", padx=20, pady=(4, 0))

            ctk.CTkLabel(
                lbl_row, text=name,
                font=("Arial", 12), text_color=TEXT_SEC,
                anchor="w", width=130
            ).pack(side="left")

            val_lbl = ctk.CTkLabel(
                lbl_row, text="—",
                font=("Arial", 12, "bold"), text_color=TEXT_MAIN,
                anchor="e"
            )
            val_lbl.pack(side="right")
            self.vis_labels.append(val_lbl)

            bar = ctk.CTkProgressBar(
                c3, height=7,
                progress_color=BG_INSET, fg_color=BG_INSET
            )
            bar.pack(fill="x", padx=20, pady=(2, 6))
            bar.set(0)
            self.vis_bars.append(bar)

        # ── Card 4: Segunda opinión difusa ────────────────────────────────────
        c4 = self._card(parent, "🔮  Segunda opinión — Sistema difuso")

        self.fuzzy_box = ctk.CTkTextbox(
            c4, height=82,
            fg_color=BG_INSET, text_color=ACCENT,
            font=("Arial", 13), wrap="word",
            corner_radius=12, border_width=0
        )
        self.fuzzy_box.pack(fill="x", padx=20, pady=(0, 14))
        self.fuzzy_box.insert("0.0", "La segunda opinión aparecerá aquí tras el análisis.")

        # ── Card 5: Recomendación ─────────────────────────────────────────────
        c5 = self._card(parent, "📋  Recomendación de AgroVisionIA")

        self.recommendation_box = ctk.CTkTextbox(
            c5, height=90,
            fg_color=BG_INSET, text_color=TEXT_MAIN,
            font=("Arial", 13), wrap="word",
            corner_radius=12, border_width=0
        )
        self.recommendation_box.pack(fill="x", padx=20, pady=(0, 14))
        self.recommendation_box.insert("0.0", "Las recomendaciones aparecerán aquí...")

        # ── Card 6: Interpretación ────────────────────────────────────────────
        c6 = self._card(parent, "📊  Interpretación inteligente")

        self.message_box = ctk.CTkTextbox(
            c6, height=90,
            fg_color=BG_INSET, text_color="#d1fae5",
            font=("Arial", 13), wrap="word",
            corner_radius=12, border_width=0
        )
        self.message_box.pack(fill="x", padx=20, pady=(0, 14))
        self.message_box.insert("0.0", "La interpretación aparecerá aquí...")

        # ── Card 7: Contexto climático ────────────────────────────────────────
        self._build_climate_card(parent)

    # ── Helper: card con título y separador ───────────────────────────────────

    def _card(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=18)
        frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            frame, text=title,
            font=("Arial", 13, "bold"), text_color=ACCENT,
            anchor="w"
        ).pack(fill="x", padx=20, pady=(14, 6))

        ctk.CTkFrame(frame, fg_color=BG_INSET, height=1).pack(
            fill="x", padx=20, pady=(0, 10)
        )
        return frame

    # =========================================================================
    # HELPERS DE ACTUALIZACIÓN UI
    # =========================================================================

    def _set_badge(self, decision):
        bg    = DECISION_BG.get(decision, BG_INSET)
        color = DECISION_COLOR.get(decision, TEXT_SEC)
        label = DECISION_LABEL.get(decision, decision)
        self.badge_frame.configure(fg_color=bg)
        self.badge_label.configure(text=label, text_color=color)

    def _update_top3(self, top_predictions):
        colors = [GREEN, YELLOW, TEXT_SEC]
        for i, pred in enumerate(top_predictions[:3]):
            name = pred.get("label", "—")
            conf = max(0.0, min(1.0, pred.get("confidence", 0)))
            self.top3_name_labels[i].configure(
                text=f"{i+1}.  {name}",
                text_color=colors[i]
            )
            self.top3_bars[i].configure(progress_color=colors[i])
            self.top3_bars[i].set(conf)
            self.top3_pct_labels[i].configure(
                text=f"{conf:.1%}", text_color=colors[i]
            )

    def _update_visual_features(self, features):
        for i, (key, name, scale) in enumerate(VISUAL_FEATURES_META):
            raw = features.get(key, 0)
            norm = max(0.0, min(1.0, raw / scale))
            pct  = f"{raw:.1f}%" if scale == 100 else f"{raw * 100:.1f}%"
            col  = VIS_BAR_COLOR.get(key, TEXT_SEC)
            self.vis_bars[i].configure(progress_color=col)
            self.vis_bars[i].set(norm)
            self.vis_labels[i].configure(text=pct, text_color=col)

    def _clear_top3(self):
        for i in range(3):
            self.top3_name_labels[i].configure(
                text=f"{i+1}.  —", text_color=TEXT_SEC)
            self.top3_bars[i].configure(progress_color=BG_INSET)
            self.top3_bars[i].set(0)
            self.top3_pct_labels[i].configure(text="—", text_color=TEXT_SEC)

    def _clear_vis_bars(self):
        for i in range(len(VISUAL_FEATURES_META)):
            self.vis_bars[i].configure(progress_color=BG_INSET)
            self.vis_bars[i].set(0)
            self.vis_labels[i].configure(text="—", text_color=TEXT_SEC)

    def _reset_confidence_bar(self, value=0.0):
        self.confidence_bar.stop()
        self.confidence_bar.configure(mode="determinate")
        self.confidence_bar.set(max(0.0, min(1.0, value)))

    # =========================================================================
    # CARD 7 — CONTEXTO CLIMÁTICO
    # =========================================================================

    def _build_climate_card(self, parent):
        c7 = self._card(parent, "🌦️  Contexto climático de propagación")

        # Aviso de privacidad
        ctk.CTkLabel(
            c7,
            text=(
                "ℹ️  La ubicación se estima por IP (precisión de ciudad). "
                "No se almacena ningún dato personal ni se comparte tu posición."
            ),
            font=("Arial", 11), text_color=TEXT_SEC,
            wraplength=560, anchor="w", justify="left",
        ).pack(fill="x", padx=20, pady=(0, 10))

        # Fila: botón + ubicación detectada
        btn_row = ctk.CTkFrame(c7, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 12))

        self.climate_btn = ctk.CTkButton(
            btn_row, text="🌦️  Analizar clima local",
            width=210, height=38, corner_radius=10,
            font=("Arial", 12, "bold"),
            fg_color="#164e63", hover_color="#0e7490",
            state="disabled",
            command=self.analyze_climate,
        )
        self.climate_btn.pack(side="left")

        self.climate_location_label = ctk.CTkLabel(
            btn_row, text="📍  —",
            font=("Arial", 12), text_color=TEXT_SEC,
        )
        self.climate_location_label.pack(side="right")

        # 4 cajas de métricas: Temp | Humedad | Lluvia | Viento
        metrics_frame = ctk.CTkFrame(c7, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=20, pady=(0, 12))

        self.climate_metric_labels = []
        metrics_cfg = [
            ("🌡️", "Temperatura"),
            ("💧", "Humedad"),
            ("🌧️", "Lluvia"),
            ("🌬️", "Viento"),
        ]

        for icon, name in metrics_cfg:
            box = ctk.CTkFrame(metrics_frame, fg_color=BG_INSET, corner_radius=12)
            box.pack(side="left", fill="x", expand=True, padx=(0, 6))

            ctk.CTkLabel(
                box, text=icon, font=("Arial", 20)
            ).pack(pady=(10, 2))

            ctk.CTkLabel(
                box, text=name,
                font=("Arial", 10), text_color=TEXT_SEC,
            ).pack()

            val_lbl = ctk.CTkLabel(
                box, text="—",
                font=("Arial", 13, "bold"), text_color=TEXT_MAIN,
            )
            val_lbl.pack(pady=(2, 10))
            self.climate_metric_labels.append(val_lbl)

        # Fila: nivel de riesgo
        risk_row = ctk.CTkFrame(c7, fg_color="transparent")
        risk_row.pack(fill="x", padx=20, pady=(0, 6))

        ctk.CTkLabel(
            risk_row, text="Nivel de riesgo climático:",
            font=("Arial", 13), text_color=TEXT_SEC,
        ).pack(side="left")

        self.climate_risk_frame = ctk.CTkFrame(
            risk_row, fg_color=BG_INSET, corner_radius=14, height=26,
        )
        self.climate_risk_frame.pack(side="left", padx=(10, 0))
        self.climate_risk_frame.pack_propagate(False)

        self.climate_risk_label = ctk.CTkLabel(
            self.climate_risk_frame, text="  —  ",
            font=("Arial", 11, "bold"), text_color=TEXT_SEC,
        )
        self.climate_risk_label.pack(side="left", fill="y")

        # Factores de riesgo detectados
        self.climate_factors_label = ctk.CTkLabel(
            c7, text="",
            font=("Arial", 11), text_color=TEXT_SEC,
            wraplength=560, anchor="w", justify="left",
        )
        self.climate_factors_label.pack(fill="x", padx=20, pady=(4, 6))

        # Conclusión generada
        self.climate_conclusion_box = ctk.CTkTextbox(
            c7, height=82,
            fg_color=BG_INSET, text_color="#bae6fd",
            font=("Arial", 12), wrap="word",
            corner_radius=12, border_width=0,
        )
        self.climate_conclusion_box.pack(fill="x", padx=20, pady=(0, 14))
        self.climate_conclusion_box.insert(
            "0.0",
            "Presiona «Analizar clima local» después de obtener un diagnóstico "
            "para ver el análisis de riesgo de propagación según el clima actual.",
        )

    def _reset_climate_card(self):
        """Restaura la card climática a su estado inicial (sin datos)."""
        self.weather_data     = None
        self.weather_location = None
        self.climate_location_label.configure(text="📍  —")
        for lbl in self.climate_metric_labels:
            lbl.configure(text="—", text_color=TEXT_MAIN)
        self.climate_risk_frame.configure(fg_color=BG_INSET)
        self.climate_risk_label.configure(text="  —  ", text_color=TEXT_SEC)
        self.climate_factors_label.configure(text="")
        self.climate_conclusion_box.delete("0.0", "end")
        self.climate_conclusion_box.insert(
            "0.0",
            "Presiona «Analizar clima local» después de obtener un diagnóstico "
            "para ver el análisis de riesgo de propagación según el clima actual.",
        )
        self.climate_btn.configure(
            text="🌦️  Analizar clima local", state="disabled")

    def _update_climate_ui(self, location, weather, risk):
        """Actualiza todos los widgets climáticos con datos reales."""
        temp   = weather.get("temperature_2m")
        hum    = weather.get("relative_humidity_2m")
        precip = weather.get("precipitation",  0.0) or 0.0
        wind   = weather.get("wind_speed_10m", 0.0) or 0.0

        temp_str   = f"{temp:.0f} °C"   if temp is not None else "—"
        hum_str    = f"{int(hum)} %"    if hum  is not None else "—"
        precip_str = f"{precip:.1f} mm"
        wind_str   = f"{wind:.0f} km/h"

        for i, val in enumerate([temp_str, hum_str, precip_str, wind_str]):
            self.climate_metric_labels[i].configure(text=val)

        # Badge de riesgo
        rl  = risk.get("risk_level", "—")
        rc  = RISK_COLOR.get(rl, TEXT_SEC)
        rb  = RISK_BG.get(rl, BG_INSET)
        self.climate_risk_frame.configure(fg_color=rb)
        self.climate_risk_label.configure(text=f"  {rl}  ", text_color=rc)

        # Factores
        factors = risk.get("factors", [])
        self.climate_factors_label.configure(
            text="  •  ".join(factors) if factors else "")

        # Conclusión
        self.climate_conclusion_box.delete("0.0", "end")
        self.climate_conclusion_box.insert(
            "0.0", risk.get("conclusion", "—"))

    # =========================================================================
    # ANÁLISIS CLIMÁTICO — acción del botón
    # =========================================================================

    def analyze_climate(self):
        """Punto de entrada del botón: lanza hilo de análisis climático."""
        if not self.last_correction or self.climate_analyzing:
            return
        threading.Thread(target=self._do_climate_analysis, daemon=True).start()

    def _do_climate_analysis(self):
        """Worker en hilo secundario: obtiene ubicación, clima y riesgo."""
        self.climate_analyzing = True

        # Estado de carga
        self._update_ui(lambda: self.climate_btn.configure(
            text="⏳  Consultando...", state="disabled"))
        self._update_ui(lambda: self.climate_location_label.configure(
            text="📍  Detectando ubicación..."))

        try:
            # ── Paso 1: ubicación por IP ──────────────────────────────────────
            location = WeatherService.get_location_by_ip()
            city     = location["city"]
            country  = location["country"]
            lat      = location["lat"]
            lon      = location["lon"]

            self._update_ui(lambda: self.climate_location_label.configure(
                text=f"📍  {city}, {country}"))

            # ── Paso 2: datos climáticos ──────────────────────────────────────
            weather = WeatherService.fetch_weather(lat, lon)

            # ── Paso 3: riesgo de propagación ─────────────────────────────────
            raw_label = self.last_correction.get("final_raw_label", "")
            risk      = WeatherService.calculate_climate_risk(weather, raw_label)

            self.weather_data     = weather
            self.weather_location = location

            def apply_result():
                self._update_climate_ui(location, weather, risk)
                self.climate_btn.configure(
                    text="🔄  Actualizar clima", state="normal")

            self._update_ui(apply_result)

        except Exception as exc:
            err = str(exc)

            def show_climate_error():
                self.climate_location_label.configure(text="📍  Error")
                self.climate_conclusion_box.delete("0.0", "end")
                self.climate_conclusion_box.insert(
                    "0.0",
                    f"No se pudo consultar el clima actual. "
                    f"La predicción de la enfermedad no fue afectada.\n\n"
                    f"Detalle: {err}",
                )
                self.climate_btn.configure(
                    text="🌦️  Reintentar", state="normal")

            self._update_ui(show_climate_error)

        finally:
            self.climate_analyzing = False

    # =========================================================================
    # ACCIONES
    # =========================================================================

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img = img.resize((415, 385))
            self.img_tk = ctk.CTkImage(
                light_image=img, dark_image=img, size=(415, 385))
            self.img_path = path
            self.image_label.configure(image=self.img_tk, text="")
        except Exception as e:
            self.img_path = None
            messagebox.showerror(
                "Error al abrir imagen",
                f"No se pudo abrir el archivo seleccionado.\n\nDetalle: {e}"
            )

    def _update_ui(self, callback):
        self.root.after(0, callback)

    def predict_image(self):
        threading.Thread(target=self.analyze, daemon=True).start()

    # =========================================================================
    # ANÁLISIS — lógica completa
    # =========================================================================

    def analyze(self):
        if not self.img_path or self.analyzing:
            return

        self.analyzing = True

        if self.predictor is None:
            self._update_ui(lambda: messagebox.showerror(
                "Modelo no disponible",
                "El modelo de IA no se cargó correctamente.\n\n"
                "Reinicia la aplicación y verifica que el archivo "
                "'modelos/modelo_plantas.keras' exista."
            ))
            self.analyzing = False
            return

        # Activar estado de carga
        self._update_ui(lambda: self.loading_label.configure(
            text="⏳  Analizando..."))
        self._update_ui(lambda: self.confidence_bar.configure(
            mode="indeterminate"))
        self._update_ui(lambda: self.confidence_bar.start())

        try:

            # ── Detección de hoja ─────────────────────────────────────────────
            if not LeafDetector.is_leaf(self.img_path):
                def show_invalid():
                    self._reset_confidence_bar(0)
                    self._set_badge("NO_CONCLUYENTE")
                    self.result_title.configure(
                        text="Imagen no válida", text_color=RED)
                    self.confidence_label.configure(text="—")
                    self.severity_label.configure(
                        text="Sin diagnóstico", text_color=RED)
                    self.fuzzy_box.delete("0.0", "end")
                    self.fuzzy_box.insert(
                        "0.0",
                        "No se detectaron patrones visuales compatibles con "
                        "hojas vegetales. Usa una foto clara de una hoja."
                    )
                    self.recommendation_box.delete("0.0", "end")
                    self.recommendation_box.insert(
                        "0.0",
                        "La imagen analizada no parece corresponder a una "
                        "hoja vegetal compatible con AgroVisionIA."
                    )
                    self.message_box.delete("0.0", "end")
                    self.message_box.insert("0.0", "—")
                    self.loading_label.configure(text="")
                    self._clear_top3()
                    self._clear_vis_bars()
                    self.climate_btn.configure(state="disabled")
                    self._reset_climate_card()
                self._update_ui(show_invalid)
                return

            # ── Corrector difuso nivel 3 ──────────────────────────────────────
            top_predictions = self.predictor.predict_top_k(self.img_path, k=3)
            visual_features = LeafDetector.extract_visual_features(self.img_path)
            correction      = self.fuzzy_corrector.correct(
                top_predictions, visual_features)

            self.last_top_predictions = top_predictions
            self.last_visual_features = visual_features
            self.last_correction      = correction

            decision   = correction["decision"]
            hyb_conf   = max(0.0, min(1.0, correction["hybrid_confidence"]))
            orig_conf  = correction.get("original_confidence", 0.0)

            # ── CONFIRMADO ────────────────────────────────────────────────────
            if decision == "CONFIRMADO":
                raw_label  = correction["final_raw_label"]
                confidence = orig_conf

                level, color, damage_pct = SeverityAnalyzer.calculate(
                    raw_label, confidence)
                diag_msg = DiagnosticMessage.generate(
                    confidence, level, damage_pct)
                rec = RecommendationEngine.generate(
                    raw_label, confidence, level)

                def show_confirmed():
                    self._reset_confidence_bar(hyb_conf)
                    self._set_badge("CONFIRMADO")
                    self.result_title.configure(
                        text=correction["final_label"], text_color=ACCENT)
                    self.confidence_label.configure(text=f"{hyb_conf:.1%}")
                    self.confidence_bar.configure(progress_color=GREEN)
                    self.severity_label.configure(
                        text=f"RIESGO  {level}", text_color=color)
                    self.fuzzy_box.delete("0.0", "end")
                    self.fuzzy_box.insert("0.0", correction["explanation"])
                    self.recommendation_box.delete("0.0", "end")
                    self.recommendation_box.insert("0.0", rec)
                    self.message_box.delete("0.0", "end")
                    self.message_box.insert("0.0", diag_msg)
                    self.loading_label.configure(text="")
                    self._update_top3(top_predictions)
                    self._update_visual_features(visual_features)
                    self.climate_btn.configure(state="normal")

                self._update_ui(show_confirmed)

            # ── POSIBLE ───────────────────────────────────────────────────────
            elif decision == "POSIBLE":
                def show_possible():
                    self._reset_confidence_bar(hyb_conf)
                    self._set_badge("POSIBLE")
                    self.result_title.configure(
                        text=f"Posible: {correction['final_label']}",
                        text_color=YELLOW)
                    self.confidence_label.configure(text=f"{hyb_conf:.1%}")
                    self.confidence_bar.configure(progress_color=YELLOW)
                    self.severity_label.configure(
                        text="REVISIÓN RECOMENDADA", text_color=YELLOW)
                    self.fuzzy_box.delete("0.0", "end")
                    self.fuzzy_box.insert("0.0", correction["explanation"])
                    self.recommendation_box.delete("0.0", "end")
                    self.recommendation_box.insert(
                        "0.0",
                        "El modelo detectó una posible coincidencia pero la "
                        "confianza es moderada. Confirma con una imagen más "
                        "nítida y enfocada en la hoja."
                    )
                    self.message_box.delete("0.0", "end")
                    self.message_box.insert(
                        "0.0",
                        "Se recomienda validación adicional. El diagnóstico "
                        "puede mejorar con una foto mejor iluminada."
                    )
                    self.loading_label.configure(text="")
                    self._update_top3(top_predictions)
                    self._update_visual_features(visual_features)
                    self.climate_btn.configure(state="normal")

                self._update_ui(show_possible)

            # ── CORREGIDO ─────────────────────────────────────────────────────
            elif decision == "CORREGIDO":
                def show_corrected():
                    self._reset_confidence_bar(hyb_conf)
                    self._set_badge("CORREGIDO")
                    self.result_title.configure(
                        text=f"Resultado ajustado: {correction['final_label']}",
                        text_color=BLUE)
                    self.confidence_label.configure(text=f"{hyb_conf:.1%}")
                    self.confidence_bar.configure(progress_color=BLUE)
                    self.severity_label.configure(
                        text="REVISIÓN RECOMENDADA", text_color=YELLOW)
                    self.fuzzy_box.delete("0.0", "end")
                    self.fuzzy_box.insert("0.0", correction["explanation"])
                    self.recommendation_box.delete("0.0", "end")
                    self.recommendation_box.insert(
                        "0.0",
                        "El sistema difuso ajustó la predicción inicial basándose "
                        "en las características visuales de la imagen. "
                        "Se recomienda validación con un experto agrícola."
                    )
                    self.message_box.delete("0.0", "end")
                    self.message_box.insert(
                        "0.0",
                        "Este es un resultado ajustado — no un diagnóstico definitivo. "
                        "El sistema encontró mayor compatibilidad visual con esta clase."
                    )
                    self.loading_label.configure(text="")
                    self._update_top3(top_predictions)
                    self._update_visual_features(visual_features)
                    self.climate_btn.configure(state="normal")

                self._update_ui(show_corrected)

            # ── NO_CONCLUYENTE ────────────────────────────────────────────────
            else:
                def show_inconclusive():
                    self._reset_confidence_bar(hyb_conf)
                    self._set_badge("NO_CONCLUYENTE")
                    self.result_title.configure(
                        text="Resultado no concluyente",
                        text_color=ORANGE)
                    self.confidence_label.configure(text=f"{orig_conf:.1%}")
                    self.confidence_bar.configure(progress_color=ORANGE)
                    self.severity_label.configure(
                        text="NO CONCLUYENTE", text_color=ORANGE)
                    self.fuzzy_box.delete("0.0", "end")
                    self.fuzzy_box.insert("0.0", correction["explanation"])
                    self.recommendation_box.delete("0.0", "end")
                    self.recommendation_box.insert(
                        "0.0",
                        "Intenta con una foto bien iluminada, enfocada en la "
                        "hoja y sin fondo complejo."
                    )
                    self.message_box.delete("0.0", "end")
                    self.message_box.insert(
                        "0.0",
                        "El sistema no pudo emitir un diagnóstico confiable "
                        "con la imagen actual."
                    )
                    self.loading_label.configure(text="")
                    self._update_top3(top_predictions)
                    self._update_visual_features(visual_features)
                    self.climate_btn.configure(state="disabled")
                    self._reset_climate_card()

                self._update_ui(show_inconclusive)

            # Habilitar el botón climático para toda predicción válida de hoja
            # (FIFO: corre después de cualquier show_ closure que lo haya deshabilitado)
            self._update_ui(lambda: self.climate_btn.configure(state="normal"))

        except Exception as e:
            def show_error():
                self._reset_confidence_bar(0)
                self._set_badge("NO_CONCLUYENTE")
                self.result_title.configure(
                    text="Error al procesar", text_color=RED)
                self.confidence_label.configure(text="—")
                self.severity_label.configure(
                    text="Sin diagnóstico", text_color=RED)
                self.fuzzy_box.delete("0.0", "end")
                self.fuzzy_box.insert("0.0", "—")
                self.recommendation_box.delete("0.0", "end")
                self.recommendation_box.insert(
                    "0.0",
                    f"Ocurrió un error durante el análisis.\n\nDetalle: {e}"
                )
                self.message_box.delete("0.0", "end")
                self.message_box.insert(
                    "0.0",
                    "Intenta subir una imagen diferente o reinicia la aplicación."
                )
                self.loading_label.configure(text="")
                self._clear_top3()
                self._clear_vis_bars()
                self.climate_btn.configure(state="disabled")
                self._reset_climate_card()

            self._update_ui(show_error)

        finally:
            self.analyzing = False
