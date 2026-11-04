"""Funciones compartidas de entrenamiento, guardado y evidencia científica.
"""
import csv
import sys
import platform
import time
from pathlib import Path
import tensorflow as tf
from procesamiento import (CARPETA_DATOS, CARPETA_RESULTADOS, crear_dataset,
                          leer_clases, verificar_imagenes, hash_archivo)
from metricas import predecir_dataset, guardar_metricas


def iniciar_corrida(nombre, semilla, parametros):
    verificar_imagenes()
    tf.keras.utils.set_random_seed(semilla)
    tf.config.experimental.enable_op_determinism()
    destino = CARPETA_RESULTADOS / nombre / f"semilla_{semilla}"
    destino.mkdir(parents=True, exist_ok=False)
    texto = "\n".join(f"{clave}: {valor}" for clave, valor in parametros.items())
    texto += (f"\nPython: {sys.version}\nPlataforma: {platform.platform()}\n"
              f"TensorFlow: {tf.__version__}\nDispositivos: {tf.config.list_physical_devices()}\n"
              f"Semilla: {semilla}\nClases: {leer_clases()}\n")
    (destino / "parametros_y_entorno.txt").write_text(texto)
    (destino / "estado.txt").write_text("en_ejecucion\n")
    with (destino / "hashes_codigo.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["archivo", "sha256"])
        for ruta in sorted(Path(__file__).parent.glob("*.py")):
            escritor.writerow([ruta.name, hash_archivo(ruta)])
    return destino


def entrenar_etapa(modelo, destino, etapa, semilla, batch_size, epocas,
                    paciencia, learning_rate, reconstruccion=False):
    carpeta = destino / etapa
    carpeta.mkdir()
    clases = leer_clases()
    if reconstruccion:
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    else:
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, len(clases)), name="top3_accuracy")])
    train = crear_dataset("train", batch_size, entrenamiento=True, reconstruccion=reconstruccion, semilla=semilla)
    validacion = crear_dataset("validation", batch_size, reconstruccion=reconstruccion)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(carpeta / "mejor.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(str(carpeta / "ultimo.keras")),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(carpeta / "historial.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    inicio = time.perf_counter()
    historia = modelo.fit(train, validation_data=validacion, epochs=epocas, callbacks=callbacks, verbose=1)
    duracion = time.perf_counter()-inicio
    tf.debugging.assert_all_finite(tf.constant(historia.history["loss"]+historia.history["val_loss"]), "Pérdida no finita")
    mejor = tf.keras.models.load_model(carpeta / "mejor.keras", compile=False)
    modelo.set_weights(mejor.get_weights())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figura, eje = plt.subplots(figsize=(6, 4))
    eje.plot(historia.history["loss"], label="Train")
    eje.plot(historia.history["val_loss"], label="Validación")
    eje.set(xlabel="Época (índice desde 0)", ylabel="Pérdida", title=etapa)
    eje.legend()
    for extension in ("png", "pdf"):
        figura.savefig(carpeta / f"curva.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figura)
    with (carpeta / "resumen.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["segundos", "epocas", "mejor_val_loss"])
        escritor.writerow([duracion, len(historia.history["loss"]), min(historia.history["val_loss"])])
    return min(historia.history["val_loss"])


def terminar_corrida(modelo, destino, batch_size):
    clases = leer_clases()
    modelo.save(destino / "clasificador.keras")
    recargado = tf.keras.models.load_model(destino / "clasificador.keras", compile=False)
    x, _ = next(iter(crear_dataset("validation", batch_size)))
    tf.debugging.assert_near(modelo(x, training=False), recargado(x, training=False), atol=1e-5, rtol=1e-4)
    rutas, reales, scores = predecir_dataset(modelo, crear_dataset("validation", batch_size, incluir_rutas=True))
    resumen = guardar_metricas(destino / "validacion", reales, scores, clases, rutas)
    with (destino / "huellas.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["archivo", "sha256"])
        for ruta in [CARPETA_DATOS / "manifiesto.csv", CARPETA_DATOS / "clases.txt", destino / "clasificador.keras"]:
            escritor.writerow([ruta.name, hash_archivo(ruta)])
    with (destino / "tamano_modelo.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["parametros_totales", "parametros_entrenables", "bytes"])
        entrenables = sum(int(tf.size(variable)) for variable in modelo.trainable_weights)
        escritor.writerow([modelo.count_params(), entrenables, (destino / "clasificador.keras").stat().st_size])
    (destino / "estado.txt").write_text("completado\n")
    print("Corrida guardada:", destino)
    print("Validación:", resumen)


def registrar_fallo(destino, error):
    (destino / "estado.txt").write_text(f"fallido\n{type(error).__name__}: {error}\n")
