# 🌿 AgroVision IA

**Sistema Inteligente de Diagnóstico Agrícola**

AgroVision IA es una aplicación de escritorio que utiliza inteligencia artificial para diagnosticar enfermedades en plantas a partir de fotografías de hojas. El sistema analiza la imagen, identifica la enfermedad, evalúa la severidad y genera recomendaciones de tratamiento fitosanitario.

---

## 📋 Requisitos

- **Python 3.10 o 3.11** (recomendado). TensorFlow puede tener restricciones de compatibilidad con versiones más recientes de Python.
- **pip** (gestor de paquetes de Python)
- **Sistema operativo**: Windows 10/11

---

## 🚀 Instalación

1. **Clonar o descargar** el repositorio:

```bash
git clone <url-del-repositorio>
cd AgroVisionIA
```

2. **Crear un entorno virtual** (recomendado):

```bash
python -m venv .venv
```

3. **Activar el entorno virtual**:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

4. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

**Importante**: La aplicación debe ejecutarse siempre desde la **raíz del proyecto**:

```bash
python main.py
```

Al iniciar se mostrará un splash screen de 3 segundos seguido del dashboard principal.

---

## 🔬 Cómo Usar

1. Haz clic en **"Subir Imagen"** para seleccionar una fotografía de una hoja.
2. Haz clic en **"Analizar Enfermedad"** para ejecutar el diagnóstico con IA.
3. El panel derecho mostrará:
   - **Diagnóstico** (nombre de la enfermedad en español)
   - **Nivel de confianza** del modelo (barra y porcentaje)
   - **Severidad estimada** (SALUDABLE / LEVE / MEDIO / ALTO / CRÍTICO)
   - **Recomendación de tratamiento** contextualizada
   - **Interpretación inteligente** del resultado

---

## 🌱 Cultivos y Enfermedades Soportadas (38 clases)

| Cultivo | Enfermedades |
|---------|-------------|
| Manzano | Sarna, Podredumbre negra, Roya, Saludable |
| Arándano | Saludable |
| Cerezo | Oídio, Saludable |
| Maíz | Mancha gris, Roya común, Tizón del norte, Saludable |
| Uva | Podredumbre negra, Esca, Tizón foliar, Saludable |
| Naranja | Enverdecimiento de cítricos (HLB) |
| Durazno | Mancha bacteriana, Saludable |
| Pimiento | Mancha bacteriana, Saludable |
| Papa | Tizón temprano, Tizón tardío, Saludable |
| Frambuesa | Saludable |
| Soya | Saludable |
| Calabaza | Oídio |
| Fresa | Quemadura foliar, Saludable |
| Tomate | Mancha bacteriana, Tizón temprano/tardío, Moho foliar, Septoria, Ácaros, Target Spot, TYLCV, Virus mosaico, Saludable |

---

## 🧠 Modelo de IA

- **Arquitectura**: MobileNetV2 (transfer learning desde ImageNet)
- **Dataset**: PlantVillage
- **Input**: Imágenes de 224×224 px, normalizadas (÷255)
- **Formato**: Keras v3 (`.keras`)
- **Ubicación**: `modelos/modelo_plantas.keras`

---

## 📁 Estructura del Proyecto

```
AgroVisionIA/
├── main.py                      # Punto de entrada
├── requirements.txt             # Dependencias
├── README.md                    # Este archivo
├── modelos/
│   └── modelo_plantas.keras     # Modelo entrenado
├── dataset/
│   └── PlantVillage/            # Dataset de entrenamiento
├── entrenamiento/
│   ├── train_model.py           # Script de entrenamiento
│   └── predict_image.py         # Script de prueba CLI
├── test_images/                 # Imágenes de prueba
└── frontend/
    ├── assets/                  # Logo y recursos visuales
    ├── ui/                      # Interfaz gráfica (CustomTkinter)
    ├── services/                # Lógica de negocio (predicción, severidad, diagnóstico)
    └── data/                    # Datos estáticos (clases, traducciones, recomendaciones)
```

---

## 🧪 Prueba CLI de Inferencia

Para probar el modelo desde línea de comandos (desde el directorio `entrenamiento/`):

```bash
cd entrenamiento
python predict_image.py ../test_images/strawberry_LS.JPG
```

O sin argumentos (usa imagen por defecto):

```bash
cd entrenamiento
python predict_image.py
```

---

## 🛠️ Stack Tecnológico

| Componente | Librería |
|---|---|
| GUI | CustomTkinter |
| Deep Learning | TensorFlow / Keras |
| Preprocesamiento | OpenCV, NumPy |
| Imágenes en UI | Pillow (PIL) |
| Visualización | Matplotlib |

---

## 📝 Notas Importantes

- La app debe ejecutarse **siempre desde la raíz del proyecto** (`python main.py`).
- El modelo `modelo_plantas.keras` debe existir en `modelos/` para que la app funcione.
- Las imágenes subidas deben ser fotografías de hojas (JPG, PNG, BMP, WEBP).
- Los porcentajes de daño estimado son **valores referenciales por categoría**, no calculados por píxel.
