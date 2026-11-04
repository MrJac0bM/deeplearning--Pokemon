"""Genera figuras adicionales desde CSV, sin TensorFlow ni reentrenamiento."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def guardar(fig, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(destino.with_suffix('.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)


def generar(raiz):
    salida = raiz / 'graficas_adicionales'
    for archivo in raiz.rglob('historial.csv'):
        df = pd.read_csv(archivo)
        etiqueta = '__'.join(archivo.parent.relative_to(raiz).parts)
        for metrica in ('loss', 'accuracy', 'top3_accuracy'):
            if metrica not in df:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            epocas = df['epoch'].to_numpy() + 1
            ax.plot(epocas, df[metrica], label='Entrenamiento')
            if 'val_' + metrica in df:
                ax.plot(epocas, df['val_' + metrica], label='Validación')
            if 'val_loss' in df:
                mejor = int(np.nanargmin(df['val_loss']))
                ax.axvline(epocas[mejor], color='gray', linestyle='--', label='Mejor val_loss')
            ax.set(xlabel='Época', ylabel=metrica, title=etiqueta)
            ax.legend()
            guardar(fig, salida / (etiqueta + '__' + metrica))

    filas = []
    for archivo in raiz.rglob('metricas.csv'):
        df = pd.read_csv(archivo)
        if df.empty or 'accuracy' not in df:
            continue
        etiqueta = '__'.join(archivo.parent.relative_to(raiz).parts)
        registro = df.iloc[0].to_dict()
        registro['evaluacion'] = etiqueta
        filas.append(registro)
        por_clase = archivo.parent / 'por_clase.csv'
        if por_clase.exists():
            tabla = pd.read_csv(por_clase)
            if {'clase', 'f1', 'soporte'} <= set(tabla):
                tabla = tabla[tabla.soporte > 0].sort_values('f1')
                fig, ax = plt.subplots(figsize=(8, 6))
                tabla.head(15).plot.barh(x='clase', y='f1', ax=ax, legend=False)
                ax.set(title=etiqueta + ': 15 menores F1', xlim=(0, 1))
                guardar(fig, salida / (etiqueta + '__clases'))
        matriz = archivo.parent / 'matriz_confusion.csv'
        if matriz.exists():
            tabla = pd.read_csv(matriz, index_col=0)
            valores = tabla.to_numpy(dtype=float)
            normalizada = np.divide(valores, valores.sum(1, keepdims=True),
                out=np.zeros_like(valores), where=valores.sum(1, keepdims=True) != 0)
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(normalizada, vmin=0, vmax=1, cmap='Blues')
            ax.set(title=etiqueta, xlabel='Clase predicha (orden clases.txt)', ylabel='Clase real')
            fig.colorbar(im, ax=ax, label='Proporción por clase real')
            guardar(fig, salida / (etiqueta + '__confusion_normalizada'))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(np.arange(len(tabla)), valores.sum(0))
            ax.set(title=etiqueta, xlabel='Clase predicha', ylabel='Número de predicciones')
            guardar(fig, salida / (etiqueta + '__predicciones'))

    if filas:
        tabla = pd.DataFrame(filas).set_index('evaluacion')
        salida.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(salida / 'metricas_reunidas.csv')
        # Validación y test se comparan en figuras distintas.
        for nombre, mascara in (
            ('validacion', tabla.index.str.contains('validacion')),
            ('test', tabla.index.str.startswith('evaluacion_final__')),
        ):
            seleccion = tabla.loc[mascara, ['accuracy', 'top3_accuracy', 'f1_macro']]
            if seleccion.empty:
                continue
            fig, ax = plt.subplots(figsize=(11, 5))
            seleccion.plot.bar(ax=ax)
            ax.set(title=nombre, ylim=(0, 1), ylabel='Valor', xlabel='')
            guardar(fig, salida / ('comparacion_' + nombre))
    print('Gráficas guardadas:', salida)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultados', type=Path, default=Path(__file__).resolve().parent.parent / 'resultados')
    generar(parser.parse_args().resultados)
