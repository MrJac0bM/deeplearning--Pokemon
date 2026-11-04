"""EDA -> EfficientNet -> exportación -> test -> gráficas."""
from pathlib import Path
import subprocess
import sys
from procesamiento import CARPETA_PROYECTO, CARPETA_RESULTADOS, CARPETA_DATOS, verificar_imagenes


def run(nombre, *args):
    subprocess.run([sys.executable, '-u', str(Path(__file__).parent / nombre), *map(str, args)], cwd=CARPETA_PROYECTO, check=True)


def main():
    run('verificar_gpu.py')
    verificar_imagenes()
    run('eda.py')
    corrida = CARPETA_RESULTADOS / 'efficientnet/semilla_42'
    if corrida.exists():
        estado = corrida / 'estado.txt'
        if not estado.exists() or estado.read_text().strip() != 'completado':
            raise RuntimeError('Corrida parcial: preserva checkpoints; recuperación manual necesaria.')
        print('Se reutiliza corrida completa.')
    else:
        run('efficientnet.py')
    final = CARPETA_PROYECTO / 'modelo_final'
    if not final.exists():
        run('exportar.py', '--modelo', corrida / 'clasificador.keras', '--clases', CARPETA_DATOS / 'clases.txt')
    else:
        import json
        from procesamiento import hash_archivo
        m = json.loads((final / 'metadatos.json').read_text())
        for nombre, digest in m['sha256'].items():
            if hash_archivo(final / nombre) != digest:
                raise ValueError('Exportación modificada: ' + nombre)
        if hash_archivo(corrida / 'clasificador.keras') != hash_archivo(final / 'modelo.keras'):
            raise ValueError('Exportación no corresponde a la corrida actual')
    test = CARPETA_RESULTADOS / 'evaluacion_final/efficientnet_semilla_42'
    if not test.exists():
        run('evaluar.py')
    elif not (test / 'estado.txt').exists() or (test / 'estado.txt').read_text().strip() != 'completado':
        raise RuntimeError('Evaluación parcial: no se sobrescribe.')
    run('graficas.py')
    print('Terminado. Usa python/predecir.py para generar tu evidencia.')


if __name__ == '__main__':
    main()
