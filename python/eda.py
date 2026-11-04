"""EDA en TensorFlow. Matplotlib se utiliza únicamente para dibujar figuras."""
import csv
from pathlib import Path
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from procesamiento import CARPETA_DATOS, CARPETA_RESULTADOS, preparar_imagen

# PARÁMETROS MANUALES.
NUMERO_MUESTRAS = 12
SEMILLA = 42


def ejecutar_eda():
    if not (CARPETA_DATOS / "manifiesto.csv").exists():
        raise FileNotFoundError("Primero termina python python/etl.py: no existe datos/manifiesto.csv")
    with (CARPETA_DATOS / "manifiesto.csv").open(newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    clases = (CARPETA_DATOS / "clases.txt").read_text().splitlines()
    destino = CARPETA_RESULTADOS / "eda"
    destino.mkdir(parents=True, exist_ok=True)
    def guardar(figura, nombre):
        for extension in ("png", "pdf"):
            figura.savefig(destino / f"{nombre}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figura)
    # Estadísticas computadas con TensorFlow; no pandas ni NumPy.
    conteos = {}
    for split in ("train", "validation", "test"):
        etiquetas = tf.constant([int(f["etiqueta"]) for f in filas if f["split"] == split], tf.int32)
        conteos[split] = tf.math.bincount(etiquetas, minlength=len(clases), maxlength=len(clases))
    with (destino / "distribucion_clases.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["clase", "train", "validation", "test"])
        for i, clase in enumerate(clases):
            escritor.writerow([clase] + [int(conteos[s][i]) for s in conteos])
    figura, eje = plt.subplots(figsize=(10, max(4, len(clases)*0.15)))
    eje.barh(clases, conteos["train"].numpy())
    eje.set(xlabel="Imágenes", title="Distribución de entrenamiento")
    guardar(figura, "clases_train")
    entrenamiento = [f for f in filas if f["split"] == "train"]
    if not entrenamiento:
        raise ValueError("No hay imágenes de entrenamiento")
    anchos = tf.constant([int(f["ancho"]) for f in entrenamiento], tf.float32)
    altos = tf.constant([int(f["alto"]) for f in entrenamiento], tf.float32)
    variables = [("ancho", anchos), ("alto", altos), ("aspecto", anchos/altos)]
    with (destino / "dimensiones_train.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["variable", "minimo", "maximo", "media", "desviacion_poblacional"])
        for nombre, valores in variables:
            escritor.writerow([nombre, float(tf.reduce_min(valores)), float(tf.reduce_max(valores)),
                                float(tf.reduce_mean(valores)), float(tf.math.reduce_std(valores))])
    figura, ejes = plt.subplots(1, 3, figsize=(12, 3))
    for eje, (nombre, valores) in zip(ejes, variables):
        eje.hist(valores.numpy(), bins=min(25, len(entrenamiento)))
        eje.set_title(nombre)
    guardar(figura, "dimensiones_train")
    indices = tf.random.experimental.stateless_shuffle(tf.range(len(entrenamiento)), seed=[SEMILLA, 0])[:NUMERO_MUESTRAS]
    figura, ejes = plt.subplots(3, 4, figsize=(10, 8))
    for eje in ejes.flat:
        eje.axis("off")
    for eje, indice in zip(ejes.flat, indices.numpy().tolist()):
        fila = entrenamiento[indice]
        eje.imshow(tf.cast(tf.round(preparar_imagen(str(CARPETA_DATOS / fila["ruta"]))), tf.uint8).numpy())
        eje.set_title(fila["clase"], fontsize=9)
    guardar(figura, "muestras_train")
    print(f"EDA guardado en {destino}")


if __name__ == "__main__":
    ejecutar_eda()
