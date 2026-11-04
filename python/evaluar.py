"""Test del modelo exportado; protege resultados existentes."""
import argparse
from pathlib import Path
import tensorflow as tf
from procesamiento import CARPETA_PROYECTO, CARPETA_RESULTADOS, crear_dataset, leer_clases, verificar_imagenes
from metricas import predecir_dataset, guardar_metricas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelo-dir', type=Path, default=CARPETA_PROYECTO / 'modelo_final')
    a = p.parse_args()
    verificar_imagenes()
    clases = leer_clases()
    if (a.modelo_dir / 'clases.txt').read_text(encoding='utf-8').splitlines() != clases:
        raise ValueError('Orden de clases distinto del dataset')
    salida = CARPETA_RESULTADOS / 'evaluacion_final' / 'efficientnet_semilla_42'
    salida.mkdir(parents=True, exist_ok=False)
    modelo = tf.keras.models.load_model(a.modelo_dir / 'modelo.keras', compile=False)
    rutas, reales, scores = predecir_dataset(modelo, crear_dataset('test', 8, incluir_rutas=True))
    print(guardar_metricas(salida, reales, scores, clases, rutas))
    (salida / 'estado.txt').write_text('completado\n')


if __name__ == '__main__':
    main()
