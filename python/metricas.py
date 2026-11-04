"""Accuracy, top-k, precisión, recall y F1 calculados solo con TensorFlow."""
import csv
from pathlib import Path
import tensorflow as tf


def calcular_metricas(reales, probabilidades, numero_clases):
    reales = tf.convert_to_tensor(reales, tf.int32)
    probabilidades = tf.convert_to_tensor(probabilidades, tf.float32)
    tf.debugging.assert_positive(tf.size(reales))
    tf.debugging.assert_equal(tf.shape(probabilidades), [tf.size(reales), numero_clases])
    tf.debugging.assert_all_finite(probabilidades, "Scores no finitos")
    tf.debugging.assert_greater_equal(probabilidades, 0.0)
    tf.debugging.assert_near(tf.reduce_sum(probabilidades, axis=1), tf.ones_like(tf.cast(reales, tf.float32)), atol=1e-4)
    tf.debugging.assert_greater_equal(reales, 0)
    tf.debugging.assert_less(reales, numero_clases)
    predichas = tf.argmax(probabilidades, axis=1, output_type=tf.int32)
    matriz = tf.math.confusion_matrix(reales, predichas, num_classes=numero_clases, dtype=tf.float32)
    verdaderos_positivos = tf.linalg.diag_part(matriz)
    soporte = tf.reduce_sum(matriz, axis=1)
    precision = tf.math.divide_no_nan(verdaderos_positivos, tf.reduce_sum(matriz, axis=0))
    recall = tf.math.divide_no_nan(verdaderos_positivos, soporte)
    f1 = tf.math.divide_no_nan(2*precision*recall, precision+recall)
    top = tf.math.top_k(probabilidades, k=min(3, numero_clases)).indices
    resumen = {
        "accuracy": float(tf.reduce_mean(tf.cast(reales == predichas, tf.float32))),
        "top3_accuracy": float(tf.reduce_mean(tf.cast(tf.reduce_any(top == reales[:, None], axis=1), tf.float32))),
        "precision_macro": float(tf.reduce_mean(precision)),
        "recall_macro": float(tf.reduce_mean(recall)),
        "f1_macro": float(tf.reduce_mean(f1)),
        "n": int(tf.size(reales)),
    }
    por_clase = tf.stack([precision, recall, f1, soporte], axis=1)
    return resumen, matriz, por_clase


def predecir_dataset(modelo, dataset):
    rutas, reales, probabilidades = [], [], []
    for x, y, archivos in dataset:
        rutas.extend(r.decode() for r in archivos.numpy().tolist())
        reales.append(tf.cast(y, tf.int32))
        probabilidades.append(modelo(x, training=False))
    if not reales:
        raise ValueError("Dataset vacío")
    return rutas, tf.concat(reales, axis=0), tf.concat(probabilidades, axis=0)


def guardar_metricas(destino, reales, probabilidades, clases, rutas):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    resumen, matriz, por_clase = calcular_metricas(reales, probabilidades, len(clases))
    with (destino / "metricas.csv").open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resumen))
        escritor.writeheader()
        escritor.writerow(resumen)
    with (destino / "por_clase.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["clase", "precision", "recall", "f1", "soporte"])
        for nombre, valores in zip(clases, por_clase.numpy().tolist()):
            escritor.writerow([nombre]+valores)
    with (destino / "matriz_confusion.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["referencia/prediccion"]+clases)
        for clase, fila in zip(clases, matriz.numpy().tolist()):
            escritor.writerow([clase]+fila)
    predichas = tf.argmax(probabilidades, axis=1).numpy().tolist()
    valores_reales = tf.convert_to_tensor(reales).numpy().tolist()
    with (destino / "predicciones.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["ruta", "clase_real", "clase_predicha"]+[f"score_{c}" for c in clases])
        for ruta, real, predicha, scores in zip(rutas, valores_reales, predichas, probabilidades.numpy().tolist()):
            escritor.writerow([ruta, clases[real], clases[predicha]]+scores)
    return resumen
