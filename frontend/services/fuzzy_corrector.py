"""
AgroVisionIA — Corrector Difuso Nivel 3
========================================
Sistema híbrido CNN + lógica difusa para segunda opinión
en clasificación de enfermedades de plantas.

Score híbrido = 0.65 * CNN_confidence + 0.35 * visual_compatibility

Decisiones posibles:
  CONFIRMADO     — confianza alta + respaldo visual
  POSIBLE        — evidencia parcial, monitoreo recomendado
  CORREGIDO      — otra clase del top-3 tiene mayor score híbrido (margen ≥ 0.08)
  NO_CONCLUYENTE — evidencia insuficiente
"""


class FuzzyCorrector:

    CNN_WEIGHT        = 0.65
    VISUAL_WEIGHT     = 0.35
    CORRECTION_MARGIN = 0.08   # Margen mínimo para activar CORREGIDO

    # =========================================================================
    # FUNCIONES DE MEMBRESÍA (0 → 1)
    # =========================================================================

    @staticmethod
    def _confidence_low(conf):
        if conf <= 0.40: return 1.0
        if conf >= 0.60: return 0.0
        return (0.60 - conf) / 0.20

    @staticmethod
    def _confidence_medium(conf):
        if conf <= 0.45 or conf >= 0.85: return 0.0
        if conf <= 0.65: return (conf - 0.45) / 0.20
        return (0.85 - conf) / 0.20

    @staticmethod
    def _confidence_high(conf):
        if conf <= 0.60: return 0.0
        if conf >= 0.90: return 1.0
        return (conf - 0.60) / 0.30

    @staticmethod
    def _damage_low(damage):
        if damage <= 10: return 1.0
        if damage >= 35: return 0.0
        return (35 - damage) / 25

    @staticmethod
    def _damage_medium(damage):
        if damage <= 10 or damage >= 65: return 0.0
        if damage <= 35: return (damage - 10) / 25
        return (65 - damage) / 30

    @staticmethod
    def _damage_high(damage):
        if damage <= 35: return 0.0
        if damage >= 70: return 1.0
        return (damage - 35) / 35

    @staticmethod
    def _leaf_area_low(area):
        if area <= 10: return 1.0
        if area >= 30: return 0.0
        return (30 - area) / 20

    @staticmethod
    def _leaf_area_good(area):
        if area <= 12: return 0.0
        if area >= 35: return 1.0
        return (area - 12) / 23

    # =========================================================================
    # COMPATIBILIDAD VISUAL POR TIPO DE CLASE
    # =========================================================================

    def _visual_compatibility(self, raw_label, features):
        """
        Score 0–1: qué tan compatible es raw_label con las características
        visuales HSV observadas en la imagen.
        """
        g   = features["green_ratio"]
        y   = features["yellow_ratio"]
        b   = features["brown_ratio"]
        d   = features["damage_percent"]
        lbl = raw_label.lower()

        # ── SALUDABLE ─────────────────────────────────────────────────────────
        if "healthy" in lbl:
            compat = g * 0.50 + (1 - min(d / 50.0, 1.0)) * 0.30 + (1 - y) * 0.20
            if d > 30:
                compat -= 0.30
            return max(0.0, min(1.0, compat))

        # ── BLIGHT (tizón temprano / tardío) ──────────────────────────────────
        if "blight" in lbl:
            compat = y * 0.35 + b * 0.35 + min(d / 60.0, 1.0) * 0.30
            return max(0.0, min(1.0, compat))

        # ── SPOT (septoria, target_spot, leaf_spot) ───────────────────────────
        if "spot" in lbl:
            compat = b * 0.45 + min(d / 50.0, 1.0) * 0.35 + y * 0.20
            if g > 0.80 and d < 10:
                compat -= 0.25   # Muy verde → poco compatible con spot
            return max(0.0, min(1.0, compat))

        # ── RUST (roya) ───────────────────────────────────────────────────────
        if "rust" in lbl:
            compat = b * 0.50 + y * 0.30 + min(d / 50.0, 1.0) * 0.20
            return max(0.0, min(1.0, compat))

        # ── SCORCH (quemado foliar) ───────────────────────────────────────────
        if "scorch" in lbl:
            compat = y * 0.40 + b * 0.30 + min(d / 60.0, 1.0) * 0.30
            return max(0.0, min(1.0, compat))

        # ── POWDERY MILDEW (oidio) — conservador: no detectamos blanco fácil ──
        if "powdery_mildew" in lbl or "powdery mildew" in lbl:
            compat = 0.35 + y * 0.10
            return max(0.0, min(1.0, compat))

        # ── VIRUS / MOSAIC / YELLOW LEAF CURL ────────────────────────────────
        if any(k in lbl for k in ["virus", "mosaic", "yellow_leaf", "curl"]):
            compat = y * 0.50 + (1 - g) * 0.30 + min(d / 40.0, 1.0) * 0.20
            return max(0.0, min(1.0, compat))

        # ── MOLD (moho foliar) ────────────────────────────────────────────────
        if "mold" in lbl:
            compat = b * 0.45 + y * 0.25 + min(d / 50.0, 1.0) * 0.30
            return max(0.0, min(1.0, compat))

        # ── ROT / BLACK ROT ───────────────────────────────────────────────────
        if "rot" in lbl:
            compat = b * 0.60 + min(d / 70.0, 1.0) * 0.40
            return max(0.0, min(1.0, compat))

        # ── BACTERIAL SPOT / BACTERIAL ────────────────────────────────────────
        if "bacterial" in lbl:
            compat = b * 0.40 + y * 0.30 + min(d / 50.0, 1.0) * 0.30
            return max(0.0, min(1.0, compat))

        # ── HAUNGLONGBING / GREENING ──────────────────────────────────────────
        if "haunglongbing" in lbl or "greening" in lbl:
            compat = y * 0.50 + (1 - g) * 0.30 + min(d / 40.0, 1.0) * 0.20
            return max(0.0, min(1.0, compat))

        # ── SPIDER MITES ──────────────────────────────────────────────────────
        if "spider" in lbl or "mite" in lbl:
            compat = y * 0.40 + b * 0.30 + min(d / 40.0, 1.0) * 0.30
            return max(0.0, min(1.0, compat))

        # ── ESCA / BLACK MEASLES ──────────────────────────────────────────────
        if "esca" in lbl or "measles" in lbl:
            compat = b * 0.40 + y * 0.35 + min(d / 50.0, 1.0) * 0.25
            return max(0.0, min(1.0, compat))

        # ── GENÉRICO (fallback) ───────────────────────────────────────────────
        compat = (y + b) * 0.40 + min(d / 50.0, 1.0) * 0.20 + 0.20
        return max(0.0, min(1.0, compat))

    # =========================================================================
    # CORRECTOR PRINCIPAL
    # =========================================================================

    def correct(self, top_predictions, visual_features):
        """
        Combina top-k predicciones CNN con características visuales.

        Parámetros:
            top_predictions : lista de dicts de predict_top_k()
            visual_features : dict de LeafDetector.extract_visual_features()

        Retorna dict con:
            decision, final_label, final_raw_label,
            original_label, original_raw_label, original_confidence,
            hybrid_confidence, risk, explanation, should_show_disease,
            visual_features
        """
        vf   = visual_features
        top1 = top_predictions[0]

        # ── Score híbrido para cada predicción ───────────────────────────────
        scored = []
        for pred in top_predictions:
            vc = self._visual_compatibility(pred["raw_label"], vf)
            hs = self.CNN_WEIGHT * pred["confidence"] + self.VISUAL_WEIGHT * vc
            scored.append({
                **pred,
                "visual_compat": round(vc, 3),
                "hybrid_score":  round(hs, 4),
            })

        scored_sorted = sorted(scored, key=lambda x: x["hybrid_score"], reverse=True)
        best = scored_sorted[0]

        # ── Debug consola ─────────────────────────────────────────────────────
        print("\n" + "─" * 58)
        print("  [CORRECTOR DIFUSO — DEBUG]")
        print("─" * 58)
        print("  Top-3 CNN:")
        for p in scored:
            print(f"    {p['rank']}. {p['raw_label'][:45]:<45} "
                  f"conf={p['confidence']:.3f}")
        print("\n  Características visuales:")
        for k, v in vf.items():
            unit = "%" if "percent" in k else ""
            print(f"    {k:<22}: {v}{unit}")
        print("\n  Scores híbridos:")
        for p in scored:
            print(f"    {p['raw_label'][:40]:<40} "
                  f"cnn={p['confidence']:.3f}  "
                  f"vis={p['visual_compat']:.3f}  "
                  f"hyb={p['hybrid_score']:.4f}")
        print("─" * 58)

        conf_top1 = top1["confidence"]
        vc_top1   = scored[0]["visual_compat"]
        hs_top1   = scored[0]["hybrid_score"]
        hs_best   = best["hybrid_score"]

        # ── Área foliar insuficiente ──────────────────────────────────────────
        if vf["leaf_area_percent"] < 7.0:
            print("  Decisión: NO_CONCLUYENTE (área foliar insuficiente)")
            print("─" * 58 + "\n")
            return self._build(
                decision="NO_CONCLUYENTE",
                final=top1, original=top1,
                hybrid_confidence=conf_top1 * 0.5,
                risk="NO CONCLUYENTE",
                explanation=(
                    "El área foliar detectada en la imagen es insuficiente. "
                    "Asegúrate de que la hoja sea visible y ocupe la mayor "
                    "parte de la imagen antes de analizar."
                ),
                show=False, vf=vf,
            )

        # ── Plantas completamente distintas + confianza baja ─────────────────
        plants      = set(p["raw_label"].split("___")[0] for p in top_predictions)
        mixed_plants = len(plants) >= 3 and conf_top1 < 0.70

        if mixed_plants:
            print("  Decisión: NO_CONCLUYENTE (plantas mixtas, confianza baja)")
            print("─" * 58 + "\n")
            return self._build(
                decision="NO_CONCLUYENTE",
                final=top1, original=top1,
                hybrid_confidence=hs_top1,
                risk="NO CONCLUYENTE",
                explanation=(
                    "Las predicciones del modelo muestran clases de plantas "
                    "completamente diferentes sin evidencia visual concluyente. "
                    "Intenta con una imagen más clara y enfocada en la hoja."
                ),
                show=False, vf=vf,
            )

        # ── CORREGIDO: otra clase tiene mayor hybrid_score (margen ≥ 0.08) ──
        if best["rank"] != 1 and (hs_best - hs_top1) >= self.CORRECTION_MARGIN:
            print(f"  Decisión: CORREGIDO → {best['raw_label']}")
            print("─" * 58 + "\n")
            return self._build(
                decision="CORREGIDO",
                final=best, original=top1,
                hybrid_confidence=hs_best,
                risk="REVISIÓN RECOMENDADA",
                explanation=(
                    f"La red sugirió inicialmente «{top1['label']}», "
                    f"pero las características visuales detectadas "
                    f"(verde {vf['green_ratio']:.0%} · "
                    f"amarillo {vf['yellow_ratio']:.0%} · "
                    f"café {vf['brown_ratio']:.0%}) "
                    f"respaldan más «{best['label']}» dentro de las "
                    f"tres predicciones principales del modelo."
                ),
                show=True, vf=vf,
            )

        # ── CONFIRMADO: confianza alta + compatibilidad visual aceptable ──────
        if conf_top1 >= 0.75 and vc_top1 >= 0.25:
            print(f"  Decisión: CONFIRMADO → {top1['raw_label']}")
            print("─" * 58 + "\n")
            return self._build(
                decision="CONFIRMADO",
                final=top1, original=top1,
                hybrid_confidence=hs_top1,
                risk=self._infer_risk(top1["raw_label"], conf_top1),
                explanation=(
                    "El diagnóstico cuenta con respaldo visual "
                    "y una confianza sólida del modelo."
                ),
                show=True, vf=vf,
            )

        # ── POSIBLE: evidencia parcial ────────────────────────────────────────
        if conf_top1 >= 0.45 and vc_top1 >= 0.15:
            print(f"  Decisión: POSIBLE → {top1['raw_label']}")
            print("─" * 58 + "\n")
            return self._build(
                decision="POSIBLE",
                final=top1, original=top1,
                hybrid_confidence=hs_top1,
                risk="REVISIÓN RECOMENDADA",
                explanation=(
                    f"El modelo detectó «{top1['label']}» con confianza moderada. "
                    f"Las características visuales son parcialmente compatibles. "
                    f"Usa una imagen más nítida con la hoja bien enfocada "
                    f"para obtener un diagnóstico más preciso."
                ),
                show=True, vf=vf,
            )

        # ── NO_CONCLUYENTE (fallback) ─────────────────────────────────────────
        print("  Decisión: NO_CONCLUYENTE (confianza e indicios visuales insuficientes)")
        print("─" * 58 + "\n")
        return self._build(
            decision="NO_CONCLUYENTE",
            final=top1, original=top1,
            hybrid_confidence=hs_top1,
            risk="NO CONCLUYENTE",
            explanation=(
                "Ni la confianza del modelo ni las características visuales "
                "son suficientes para emitir un diagnóstico confiable. "
                "Intenta con una foto bien iluminada y enfocada en la hoja."
            ),
            show=False, vf=vf,
        )

    # =========================================================================
    # HELPERS PRIVADOS
    # =========================================================================

    @staticmethod
    def _infer_risk(raw_label, confidence):
        """Risk string simple para el caso CONFIRMADO."""
        lbl = raw_label.lower()
        if "healthy" in lbl:
            return "SALUDABLE"
        if confidence >= 0.90:
            return "RIESGO ALTO"
        if confidence >= 0.75:
            return "RIESGO MODERADO"
        return "RIESGO LEVE"

    @staticmethod
    def _build(decision, final, original, hybrid_confidence,
               risk, explanation, show, vf):
        return {
            "decision":            decision,
            "final_label":         final["label"],
            "final_raw_label":     final["raw_label"],
            "original_label":      original["label"],
            "original_raw_label":  original["raw_label"],
            "original_confidence": original["confidence"],
            "hybrid_confidence":   round(max(0.0, min(1.0, hybrid_confidence)), 4),
            "risk":                risk,
            "explanation":         explanation,
            "should_show_disease": show,
            "visual_features":     vf,
        }
