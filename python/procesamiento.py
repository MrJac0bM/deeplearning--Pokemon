from pathlib import Path
import csv
import hashlib
import os
import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# EDITAR AQUÍ si necesitas otras rutas o tamaño. Sin archivos de configuración.
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATOS = CARPETA_PROYECTO / "datos"
CARPETA_RESULTADOS = CARPETA_PROYECTO / os.environ.get("POKEMON_RESULTADOS", "resultados")
TAMANO_IMAGEN = 224


def leer_rgb(ruta):
    """JPEG o PNG a RGB uint8; PNG transparente se compone sobre blanco.

    El dataset original ya está autoorientado. No se corrige EXIF de fotos nuevas.
    """
    contenido = tf.io.read_file(ruta)
    def png():
        rgba = tf.io.decode_png(contenido, channels=4)
        rgb = tf.cast(rgba[..., :3], tf.float32)
        alfa = tf.cast(rgba[..., 3:], tf.float32) / 255.0
        return tf.cast(tf.round(rgb * alfa + 255.0 * (1.0 - alfa)), tf.uint8)
    imagen = tf.cond(tf.io.is_jpeg(contenido),
                     lambda: tf.io.decode_jpeg(contenido, channels=3), png)
    imagen.set_shape([None, None, 3])
    return imagen


def preparar_imagen(ruta):
    """224×224 float32 [0,255]; padding blanco sin deformar la imagen."""
    imagen = tf.cast(leer_rgb(ruta), tf.float32)
    return 255.0 - tf.image.resize_with_pad(255.0-imagen, TAMANO_IMAGEN, TAMANO_IMAGEN)


def leer_clases():
    return (CARPETA_DATOS / "clases.txt").read_text().splitlines()


def hash_archivo(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024*1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_manifiesto():
    ruta = CARPETA_DATOS / "manifiesto.csv"
    certificado = (CARPETA_DATOS / "auditoria_aprobada.txt").read_text().splitlines()
    if certificado != [hash_archivo(ruta), hash_archivo(CARPETA_DATOS / "clases.txt")]:
        raise ValueError("Manifiesto o clases cambiaron después de la auditoría")
    with ruta.open(newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise ValueError("Manifiesto vacío")
    return filas


def verificar_imagenes():
    """Llamar antes de entrenar para detectar cambios en los datos auditados."""
    for fila in leer_manifiesto():
        if hash_archivo(CARPETA_DATOS / fila["ruta"]) != fila["sha256_archivo"]:
            raise ValueError(f"Imagen modificada: {fila['ruta']}")


def crear_dataset(split, batch_size=32, entrenamiento=False, reconstruccion=False,
                  semilla=42, aumentar=True, incluir_rutas=False):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split desconocido")
    if entrenamiento and split != "train":
        raise ValueError("Solo train permite entrenamiento/aumentos")
    filas = [f for f in leer_manifiesto() if f["split"] == split]
    if not filas or batch_size < 1:
        raise ValueError("Split vacío o batch inválido")
    rutas = [str(CARPETA_DATOS / f["ruta"]) for f in filas]
    etiquetas = [int(f["etiqueta"]) for f in filas]
    dataset = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    if entrenamiento:
        dataset = dataset.shuffle(len(filas), seed=semilla, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda ruta, y: (preparar_imagen(ruta), y, ruta), num_parallel_calls=1)
    dataset = dataset.batch(batch_size)
    aumentos = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=semilla),
        tf.keras.layers.RandomRotation(0.04, fill_mode="constant", fill_value=255, seed=semilla+1),
    ]) if entrenamiento and aumentar else None
    def preparar_batch(x, y, rutas_batch):
        if aumentos is not None:
            x = tf.clip_by_value(aumentos(x, training=True), 0, 255)
        if incluir_rutas:
            return x, y, rutas_batch
        # Entrada y objetivo reconstruido se aumentan juntos, sin desalinearlos.
        return x, x/255.0 if reconstruccion else y
    dataset = dataset.map(preparar_batch, num_parallel_calls=1)
    opciones = tf.data.Options()
    opciones.experimental_deterministic = True
    opciones.threading.private_threadpool_size = 4
    return dataset.with_options(opciones).prefetch(2)
