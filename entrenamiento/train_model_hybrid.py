"""
AgroVisionIA - Script de Entrenamiento Híbrido v2
==================================================
MobileNetV2 + transfer learning + fine-tuning opcional
Clasificación multiclase - PlantVillage Dataset (38 clases)

Novedades v2:
  · Top-3 accuracy metric
  · Evaluación del train set SIN augmentation
  · Fase de fine-tuning opcional (ENABLE_FINE_TUNING)
  · Matriz de confusión → PNG + CSV
  · Reporte de las 10 clases con peor F1
  · Gráficas separadas: accuracy, loss, top3, fine-tuning
"""

import os
import sys
import json
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    from sklearn.metrics import classification_report, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[AVISO] scikit-learn no instalado. Instala con: pip install scikit-learn\n")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from frontend.data.class_names import class_names as APP_CLASS_NAMES
    HAS_APP_CLASSES = True
except ImportError:
    HAS_APP_CLASSES = False
    APP_CLASS_NAMES = []

# =============================================================================
# PARÁMETROS CONFIGURABLES
# =============================================================================
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 20
LEARNING_RATE   = 0.001

# ── Fine-tuning ───────────────────────────────────────────────────────────────
ENABLE_FINE_TUNING   = True   # Cambiar a False para saltarlo
FINE_TUNE_LAYERS     = 30     # Últimas N capas a descongelar
FINE_TUNE_LR         = 1e-5
FINE_TUNE_EPOCHS     = 10

# ── Rutas ─────────────────────────────────────────────────────────────────────
TRAIN_DIR      = "../dataset/PlantVillage/train"
VAL_DIR        = "../dataset/PlantVillage/val"
TEST_DIR       = "../dataset/PlantVillage/test"
MODEL_OUTPUT   = "../modelos/modelo_plantas.keras"
INDICES_OUTPUT = "../modelos/class_indices.json"

PLOT_ACC_PATH   = "training_accuracy.png"
PLOT_LOSS_PATH  = "training_loss.png"
PLOT_TOP3_PATH  = "training_top3_accuracy.png"
PLOT_FT_ACC     = "fine_tuning_accuracy.png"
PLOT_FT_LOSS    = "fine_tuning_loss.png"
CONFUSION_PNG   = "confusion_matrix.png"
CONFUSION_CSV   = "confusion_matrix.csv"


# =============================================================================
# VALIDACIÓN DE RUTAS
# =============================================================================
def validate_paths():
    base = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.normpath(os.path.join(base, TRAIN_DIR))
    val_path   = os.path.normpath(os.path.join(base, VAL_DIR))
    test_path  = os.path.normpath(os.path.join(base, TEST_DIR))
    model_dir  = os.path.normpath(os.path.join(base, "../modelos"))

    print(f"\n{'='*62}")
    print("   AgroVisionIA — Entrenamiento Híbrido v2")
    print(f"{'='*62}")
    print(f"   TensorFlow : {tf.__version__}")
    print(f"   Keras      : {keras.__version__}")
    print(f"   Fine-tuning: {'ACTIVADO' if ENABLE_FINE_TUNING else 'DESACTIVADO'}")
    print(f"{'='*62}\n")

    for path, name in [(train_path, "train/"), (val_path, "val/")]:
        if not os.path.exists(path):
            print(f"[ERROR] No existe {name}: {path}")
            sys.exit(1)

    os.makedirs(model_dir, exist_ok=True)

    has_test  = os.path.exists(test_path)
    eval_path = test_path if has_test else val_path

    if has_test:
        print("[INFO] Carpeta test/ encontrada. Se usará para evaluación final.")
    else:
        print("[AVISO] No existe test/. Se usará val/ para evaluación final.")
        print("        LIMITACIÓN: métricas finales pueden estar optimistas.\n")

    return train_path, val_path, eval_path, has_test


# =============================================================================
# VALIDACIÓN DE CLASES (PRE-VUELO)
# =============================================================================
def validate_classes(train_path, val_path):
    print(f"\n{'='*62}")
    print("  Validación de clases (pre-vuelo)")
    print(f"{'='*62}")

    def get_classes(path):
        return sorted([
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ])

    train_classes = get_classes(train_path)
    val_classes   = get_classes(val_path)

    print(f"[INFO] Clases en train/         : {len(train_classes)}")
    print(f"[INFO] Clases en val/           : {len(val_classes)}")
    if HAS_APP_CLASSES:
        print(f"[INFO] Clases en class_names.py : {len(APP_CLASS_NAMES)}")

    errors = False

    only_in_train = set(train_classes) - set(val_classes)
    only_in_val   = set(val_classes)   - set(train_classes)

    if only_in_train:
        errors = True
        print(f"\n[ERROR] En train/ pero NO en val/:")
        for c in sorted(only_in_train):
            print(f"   ✗  {c}")
        print("\n  SOLUCIÓN: Crea la carpeta en val/ y agrega imágenes.")

    if only_in_val:
        errors = True
        print(f"\n[ERROR] En val/ pero NO en train/:")
        for c in sorted(only_in_val):
            print(f"   ✗  {c}")

    if HAS_APP_CLASSES:
        dataset_set        = set(train_classes)
        missing_in_dataset = set(APP_CLASS_NAMES) - dataset_set
        extra_in_dataset   = dataset_set - set(APP_CLASS_NAMES)

        if missing_in_dataset:
            errors = True
            print(f"\n[ERROR] En class_names.py pero NO en el dataset:")
            for c in sorted(missing_in_dataset):
                print(f"   ✗  {c}")

        if extra_in_dataset:
            errors = True
            print(f"\n[ERROR] En el dataset pero NO en class_names.py:")
            for c in sorted(extra_in_dataset):
                print(f"   ✗  {c}")

    if errors:
        print(f"\n{'='*62}")
        print("  ENTRENAMIENTO BLOQUEADO — mismatch de clases")
        print(f"{'='*62}")
        print("  Opciones:")
        print("  A) Agrega la carpeta faltante en train/ y val/ con imágenes.")
        print("  B) Si trabajarás con menos clases, actualiza también:")
        print("     · frontend/data/class_names.py")
        print("     · frontend/data/translations.py")
        print("     · frontend/data/recommendations.py")
        print(f"{'='*62}\n")
        sys.exit(1)

    print(f"[OK] {len(train_classes)} clases — train/ val/ y class_names.py coinciden.\n")
    return train_classes


# =============================================================================
# DATA AUGMENTATION
# =============================================================================
def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(factor=0.10),
    ], name="data_augmentation")


# =============================================================================
# CARGA DE DATASETS
# =============================================================================
def load_datasets(train_path, val_path, eval_path):
    print(f"[INFO] Cargando datasets...")

    common = dict(image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                  label_mode="categorical")

    train_ds = keras.utils.image_dataset_from_directory(
        train_path, shuffle=True, seed=42, **common)

    val_ds = keras.utils.image_dataset_from_directory(
        val_path, shuffle=False, **common)

    eval_ds = keras.utils.image_dataset_from_directory(
        eval_path, shuffle=False, **common)

    # Dataset de train SIN augmentation (para diagnóstico real de accuracy)
    train_eval_ds = keras.utils.image_dataset_from_directory(
        train_path, shuffle=False, **common)

    class_names_list = train_ds.class_names
    num_classes      = len(class_names_list)
    print(f"[INFO] Clases cargadas: {num_classes}\n")

    return train_ds, val_ds, eval_ds, train_eval_ds, class_names_list, num_classes


# =============================================================================
# PIPELINE tf.data
# =============================================================================
def preprocess_datasets(train_ds, val_ds, eval_ds, train_eval_ds):
    augmentation = build_augmentation()
    AUTOTUNE = tf.data.AUTOTUNE

    norm = lambda x, y: (x / 255.0, y)
    aug  = lambda x, y: (augmentation(x / 255.0, training=True), y)

    train_ds      = train_ds.map(aug,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds        = val_ds.map(norm,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    eval_ds       = eval_ds.map(norm,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    train_eval_ds = train_eval_ds.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, eval_ds, train_eval_ds


# =============================================================================
# CONSTRUCCIÓN DEL MODELO (con Top-3 accuracy)
# =============================================================================
def build_model(num_classes):
    print("[INFO] Construyendo modelo MobileNetV2 + cabezal multiclase...")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="AgroVisionIA_Hybrid")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ]
    )

    model.summary()
    return model, base_model


# =============================================================================
# CALLBACKS
# =============================================================================
def get_callbacks(model_output_path, monitor="val_accuracy"):
    return [
        EarlyStopping(monitor=monitor, patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_output_path, monitor=monitor,
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]


# =============================================================================
# FINE-TUNING
# =============================================================================
def fine_tune_model(model, base_model, train_ds, val_ds,
                    model_output_path, script_dir):
    print(f"\n{'='*62}")
    print("  Fase 2: Fine-tuning")
    print(f"{'='*62}")
    print(f"[INFO] Descongelando últimas {FINE_TUNE_LAYERS} capas del base model...")

    base_model.trainable = True

    # Congelar todas menos las últimas FINE_TUNE_LAYERS
    freeze_until = len(base_model.layers) - FINE_TUNE_LAYERS
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] Capas entrenables en base_model: {trainable_count}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ]
    )

    ft_callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_output_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    print(f"[INFO] Fine-tuning por {FINE_TUNE_EPOCHS} épocas (lr={FINE_TUNE_LR})...\n")
    ft_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=ft_callbacks,
        verbose=1
    )

    # Gráficas fine-tuning
    _save_single_plot(
        ft_history.history["accuracy"],
        ft_history.history["val_accuracy"],
        "Fine-Tuning — Accuracy",
        "Accuracy",
        os.path.join(script_dir, PLOT_FT_ACC)
    )
    _save_single_plot(
        ft_history.history["loss"],
        ft_history.history["val_loss"],
        "Fine-Tuning — Loss",
        "Loss",
        os.path.join(script_dir, PLOT_FT_LOSS)
    )

    return ft_history


# =============================================================================
# GUARDAR GRÁFICAS
# =============================================================================
def _save_single_plot(train_vals, val_vals, title, ylabel, path, label_prefix=("Train", "Val")):
    plt.figure(figsize=(10, 5))
    plt.plot(train_vals, label=f"{label_prefix[0]}", marker="o")
    plt.plot(val_vals,   label=f"{label_prefix[1]}", marker="s")
    plt.title(f"AgroVisionIA — {title}")
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Gráfica → {path}")


def save_plots(history, script_dir):
    h = history.history
    _save_single_plot(h["accuracy"],     h["val_accuracy"],
                      "Accuracy por Época",   "Accuracy",
                      os.path.join(script_dir, PLOT_ACC_PATH))

    _save_single_plot(h["loss"],         h["val_loss"],
                      "Loss por Época",       "Loss",
                      os.path.join(script_dir, PLOT_LOSS_PATH))

    if "top_3_accuracy" in h:
        _save_single_plot(h["top_3_accuracy"], h["val_top_3_accuracy"],
                          "Top-3 Accuracy por Época", "Top-3 Accuracy",
                          os.path.join(script_dir, PLOT_TOP3_PATH))


# =============================================================================
# EVALUACIÓN DIAGNÓSTICA: TRAIN SIN AUGMENTATION
# =============================================================================
def evaluate_train_no_aug(model, train_eval_ds):
    print(f"\n{'='*62}")
    print("  [TRAIN EVAL SIN AUGMENTATION]")
    print("  (accuracy real del modelo sobre datos de entrenamiento)")
    print(f"{'='*62}")
    results = model.evaluate(train_eval_ds, verbose=1)
    names   = model.metrics_names
    for name, val in zip(names, results):
        print(f"   {name:<20}: {val:.4f}")
    print()
    print("  NOTA: Si esta accuracy es alta (>80%) y la reportada")
    print("  durante entrenamiento fue ~12%, es efecto del augmentation")
    print("  y el dropout activos — NO indica problema en el modelo.\n")


# =============================================================================
# EVALUACIÓN FINAL
# =============================================================================
def evaluate_model(model, eval_ds, class_names_list, has_test, script_dir):
    source = "test" if has_test else "val"
    print(f"\n[INFO] Evaluación final en {source}/...")

    results = model.evaluate(eval_ds, verbose=1)
    names   = model.metrics_names
    print(f"\n[RESULTADO FINAL]")
    for name, val in zip(names, results):
        print(f"   {name:<20}: {val:.4f}")

    if not HAS_SKLEARN:
        return

    y_true, y_pred_probs = [], []
    for images, labels in eval_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    y_true       = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print(f"\n{'='*62}")
    print("  Classification Report")
    print(f"{'='*62}")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names_list,
        output_dict=True,
        zero_division=0
    )
    print(classification_report(
        y_true, y_pred,
        target_names=class_names_list,
        zero_division=0
    ))

    # Peores 10 clases por F1
    _print_worst_classes(report, class_names_list)

    # Matriz de confusión
    _save_confusion_matrix(y_true, y_pred, class_names_list, script_dir)

    return y_true, y_pred, report


# =============================================================================
# PEORES CLASES POR F1
# =============================================================================
def _print_worst_classes(report, class_names_list):
    f1_scores = [
        (name, report[name]["f1-score"])
        for name in class_names_list
        if name in report
    ]
    f1_scores.sort(key=lambda x: x[1])

    print(f"\n{'='*62}")
    print("  Clases con peor rendimiento (F1)")
    print(f"{'='*62}")
    for rank, (name, f1) in enumerate(f1_scores[:10], 1):
        bar = "█" * int(f1 * 20)
        print(f"  {rank:>2}. {name:<50} F1: {f1:.2f}  {bar}")
    print()


# =============================================================================
# MATRIZ DE CONFUSIÓN
# =============================================================================
def _save_confusion_matrix(y_true, y_pred, class_names_list, script_dir):
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred)
    n  = len(class_names_list)

    # ── PNG ───────────────────────────────────────────────────────────────────
    fig_size = max(14, n // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names_list, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names_list, fontsize=6)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión — AgroVisionIA")

    # Anotar celdas con valor (solo si n <= 20 para no saturar)
    if n <= 20:
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    png_path = os.path.join(script_dir, CONFUSION_PNG)
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[INFO] Matriz de confusión → {png_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(script_dir, CONFUSION_CSV)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names_list)
        for i, row in enumerate(cm):
            writer.writerow([class_names_list[i]] + list(row))
    print(f"[INFO] Matriz CSV          → {csv_path}")


# =============================================================================
# COMPATIBILIDAD CON CLASS_NAMES.PY (post-entrenamiento)
# =============================================================================
def check_class_compatibility(trained_classes):
    if not HAS_APP_CLASSES:
        return

    mismatches = [
        (i, t, a)
        for i, (t, a) in enumerate(zip(trained_classes, APP_CLASS_NAMES))
        if t != a
    ]

    print(f"\n{'='*62}")
    print("  Verificación post-entrenamiento (class_names.py)")
    print(f"{'='*62}")
    if not mismatches:
        print(f"[OK] {len(trained_classes)} clases coinciden con class_names.py.")
        print("     Modelo 100% compatible con la app.\n")
    else:
        print(f"[ADVERTENCIA] {len(mismatches)} diferencias de orden:")
        for idx, trained, app in mismatches:
            print(f"   [{idx:02d}] dataset='{trained}' | app='{app}'")
        print("\n[ACCIÓN] Actualiza frontend/data/class_names.py con el orden del dataset.\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Validar rutas
    train_path, val_path, eval_path, has_test = validate_paths()

    # 2. Validar clases ANTES de cargar TF
    validate_classes(train_path, val_path)

    # 3. Cargar datasets
    train_ds, val_ds, eval_ds, train_eval_ds, class_names_list, num_classes = (
        load_datasets(train_path, val_path, eval_path)
    )

    # 4. Preprocesar
    train_ds, val_ds, eval_ds, train_eval_ds = preprocess_datasets(
        train_ds, val_ds, eval_ds, train_eval_ds
    )

    # 5. Construir modelo
    model, base_model = build_model(num_classes)

    # 6. Callbacks
    model_output_path = os.path.normpath(os.path.join(script_dir, MODEL_OUTPUT))
    callbacks = get_callbacks(model_output_path)

    # 7. Entrenamiento fase 1 (base congelada)
    print(f"\n[INFO] Fase 1: Transfer learning ({EPOCHS} épocas máx.)...")
    print(f"[INFO] Modelo → {model_output_path}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Gráficas fase 1
    save_plots(history, script_dir)

    # 9. Diagnóstico: evaluar train SIN augmentation
    evaluate_train_no_aug(model, train_eval_ds)

    # 10. Fine-tuning (si está activado)
    if ENABLE_FINE_TUNING:
        fine_tune_model(model, base_model, train_ds, val_ds,
                        model_output_path, script_dir)

    # 11. Guardar class_indices.json
    indices_path  = os.path.normpath(os.path.join(script_dir, INDICES_OUTPUT))
    class_indices = {name: idx for idx, name in enumerate(class_names_list)}
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2, ensure_ascii=False)
    print(f"[INFO] class_indices.json → {indices_path}")

    # 12. Evaluación final + confusion matrix + peores clases
    evaluate_model(model, eval_ds, class_names_list, has_test, script_dir)

    # 13. Compatibilidad con la app
    check_class_compatibility(class_names_list)

    # 14. Resumen
    print(f"\n{'='*62}")
    print("  Entrenamiento completado")
    print(f"{'='*62}")
    print(f"  Modelo          : {model_output_path}")
    print(f"  Class indices   : {indices_path}")
    print(f"  Accuracy plot   : {os.path.join(script_dir, PLOT_ACC_PATH)}")
    print(f"  Loss plot       : {os.path.join(script_dir, PLOT_LOSS_PATH)}")
    print(f"  Top-3 plot      : {os.path.join(script_dir, PLOT_TOP3_PATH)}")
    print(f"  Confusion PNG   : {os.path.join(script_dir, CONFUSION_PNG)}")
    print(f"  Confusion CSV   : {os.path.join(script_dir, CONFUSION_CSV)}")
    if ENABLE_FINE_TUNING:
        print(f"  FT accuracy     : {os.path.join(script_dir, PLOT_FT_ACC)}")
        print(f"  FT loss         : {os.path.join(script_dir, PLOT_FT_LOSS)}")
    print(f"\n  Para correr la app: python main.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
