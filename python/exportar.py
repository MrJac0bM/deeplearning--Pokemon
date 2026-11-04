"""Exporta el modelo, su arquitectura exacta y sus pesos."""
from pathlib import Path
import argparse
import json
import shutil
import tensorflow as tf

from procesamiento import CARPETA_PROYECTO, hash_archivo


def exportar(modelo_path, clases_path, destino):
    modelo_path = Path(modelo_path)
    clases_path = Path(clases_path)
    destino = Path(destino)

    clases = clases_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(set(clases)) != len(clases) or len(clases) < 2:
        raise ValueError("Lista de clases inválida")

    modelo = tf.keras.models.load_model(
        modelo_path, compile=False
    )

    if modelo.output_shape[-1] != len(clases):
        raise ValueError("Modelo y clases no coinciden")

    # Reconstruir exactamente la arquitectura guardada.
    arquitectura = modelo.to_json()
    reconstruido = tf.keras.models.model_from_json(
        arquitectura
    )

    destino.mkdir(parents=True, exist_ok=False)

    shutil.copy2(modelo_path, destino / "modelo.keras")
    shutil.copy2(clases_path, destino / "clases.txt")

    (destino / "arquitectura.json").write_text(
        arquitectura, encoding="utf-8"
    )

    modelo.save_weights(destino / "modelo.weights.h5")
    reconstruido.load_weights(
        destino / "modelo.weights.h5"
    )

    x = tf.ones((1, 224, 224, 3), tf.float32) * 127

    tf.debugging.assert_near(
        modelo(x, training=False),
        reconstruido(x, training=False),
        atol=1e-5,
        rtol=1e-4,
    )

    metadatos = {
        "arquitectura": "EfficientNetB0 con clasificador",
        "entrada": [224, 224, 3],
        "rango": [0, 255],
        "padding": "blanco",
        "tensorflow": tf.__version__,
        "clases": len(clases),
        "sha256": {
            archivo.name: hash_archivo(archivo)
            for archivo in destino.iterdir()
            if archivo.is_file()
        },
    }

    (destino / "metadatos.json").write_text(
        json.dumps(metadatos, indent=2),
        encoding="utf-8",
    )

    print("Modelo y pesos exportados y verificados:", destino)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelo", type=Path, required=True
    )
    parser.add_argument(
        "--clases", type=Path, required=True
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=CARPETA_PROYECTO / "modelo_final",
    )

    args = parser.parse_args()
    exportar(args.modelo, args.clases, args.salida)