"""Predice una imagen y guarda evidencia PNG, PDF, JSON y CSV."""
from pathlib import Path
import argparse
import csv
import json
from datetime import datetime
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from procesamiento import CARPETA_PROYECTO, preparar_imagen, hash_archivo
from efficientnet import construir_efficientnet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--imagen', type=Path, required=True)
    p.add_argument('--modelo-dir', type=Path, default=CARPETA_PROYECTO / 'modelo_final')
    p.add_argument('--solo-pesos', action='store_true')
    p.add_argument('--clase-real', help='Opcional: etiqueta conocida, nunca inferida')
    a = p.parse_args()
    if not a.imagen.is_file():
        raise FileNotFoundError(a.imagen)
    clases = (a.modelo_dir / 'clases.txt').read_text(encoding='utf-8').splitlines()
    if a.solo_pesos:
        arquitectura = (a.modelo_dir / "arquitectura.json").read_text(
        encoding="utf-8"
        )
        modelo = tf.keras.models.model_from_json(arquitectura)
        peso = a.modelo_dir / 'modelo.weights.h5'
        modelo.load_weights(peso)
    else:
        peso = a.modelo_dir / 'modelo.keras'
        modelo = tf.keras.models.load_model(peso, compile=False)
    if modelo.output_shape[-1] != len(clases):
        raise ValueError('Modelo y clases incompatibles')
    x = preparar_imagen(str(a.imagen.resolve()))
    scores = modelo(x[None], training=False).numpy()[0]
    indices = scores.argsort()[::-1][:min(5, len(clases))]
    top = [{'clase': clases[i], 'probabilidad_softmax': float(scores[i])} for i in indices]
    destino = CARPETA_PROYECTO / 'evidencias' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destino.mkdir(parents=True)
    registro = {'imagen': str(a.imagen.resolve()), 'sha256_imagen': hash_archivo(a.imagen),
        'modelo': str(peso.resolve()), 'sha256_modelo': hash_archivo(peso),
        'clase_real_declarada': a.clase_real, 'top5': top,
        'nota': 'Softmax no es una medida calibrada de certeza; no detecta clases desconocidas.'}
    (destino / 'prediccion.json').write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding='utf-8')
    with (destino / 'top5.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['clase', 'probabilidad_softmax'])
        w.writeheader(); w.writerows(top)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8).numpy())
    axes[0].axis('off')
    axes[0].set_title('Imagen de entrada procesada')
    axes[1].barh([r['clase'] for r in top][::-1], [r['probabilidad_softmax'] for r in top][::-1])
    axes[1].set(xlim=(0, 1), xlabel='Probabilidad softmax', title='Cinco primeras predicciones')
    fig.suptitle('Predicción: ' + top[0]['clase'] + (f' | Real: {a.clase_real}' if a.clase_real else ''))
    for ext in ('png', 'pdf'):
        fig.savefig(destino / ('evidencia.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(registro, indent=2, ensure_ascii=False))
    print('Evidencia:', destino)


if __name__ == '__main__':
    main()
