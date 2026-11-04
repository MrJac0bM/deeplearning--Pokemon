"""ETL de Pokémon: inventario, revisión y reparto por grupos."""

from pathlib import Path, PurePosixPath
from collections import Counter
from datetime import datetime
import csv
import hashlib
import random
import shutil
import zipfile

import tensorflow as tf

from procesamiento import (
    CARPETA_DATOS,
    CARPETA_RESULTADOS,
    leer_rgb,
    hash_archivo,
)


# ============================================================
# PARÁMETROS MANUALES
# ============================================================

SEMILLA_REPARTO = 42
DISTANCIA_HASH = 2



REPARTIR_POR_GRUPOS = True

PARES_MISMA_FUENTE = {
    (
        "originales/train/Beedrill/6e3a5971cebf1ae01a24260f039cdc3f_jpg.rf.1d7e8a8c5dc143d3ae0eb42a425673ce.jpg",
        "originales/train/Beedrill/c57b272d60a84cba986162be7ebf24e1_jpg.rf.ce4ea157072f8351a7c9c58075866dee.jpg",
    ),
    (
        "originales/train/Doduo/c57a5edd73cc4fcb984b22eab97dff8e_jpg.rf.c271be679581b15f2d084dcc5d4c00cd.jpg",
        "originales/train/Doduo/d09d8ebf36dd4b76b5f076ad339bd659_jpg.rf.b52e447682e761e4c1d3b6a2e4598e7d.jpg",
    ),
    (
        "originales/train/Dratini/203afd4980494b8b9638e3e0916a9fd9_jpg.rf.4051d898e75ba5d2c34419952d1eaef1.jpg",
        "originales/train/Dratini/203afd4980494b8b9638e3e0916a9fd9_jpg.rf.d4235203c06d37059209e0d0ccccb5ae.jpg",
    ),
    (
        "originales/train/Dratini/8febde5ce6184b85b0469367c03f4493_jpg.rf.107acb23c09e81a3500c79f999bf7908.jpg",
        "originales/train/Dratini/8febde5ce6184b85b0469367c03f4493_jpg.rf.5158a368e063d02820cf219769ca0463.jpg",
    ),
    (
        "originales/train/Dratini/a2c3515f47464c27b181f64f8af08cd1_jpg.rf.93826d73f2accfad1d360beab9cf0334.jpg",
        "originales/train/Dratini/a2c3515f47464c27b181f64f8af08cd1_jpg.rf.a374ac8f584f028561c6c087445613cf.jpg",
    ),
    (
        "originales/train/Dratini/b64858aee72d4e32b7133a5dd78e133d_jpg.rf.3bd430e6540dd3687f259b6fe827d014.jpg",
        "originales/train/Dratini/b64858aee72d4e32b7133a5dd78e133d_jpg.rf.874f4483b9b9ad98b33f7c1bdfc955b8.jpg",
    ),
    (
        "originales/train/Nidorina/dcd5fccb6e0a4385b38e6e98aec7f1a5_jpeg.rf.0e9837835295c63715c9ae9951295532.jpg",
        "originales/train/Nidorina/dcd5fccb6e0a4385b38e6e98aec7f1a5_jpg.rf.c2bf92fdc34fe46f888107fbd2227f14.jpg",
    ),
    (
        "originales/train/Pikachu/00000001_png.rf.1a8979d63fcb95233767920f838e5d32.jpg",
        "originales/train/Pikachu/39a31e1c0aea475aa37ce3c60d1495f5_jpg.rf.6a9c0472c284db99f4822139eb850be4.jpg",
    ),
    (
        "originales/train/Poliwhirl/235e136ad0564826a1fd12410ff87d75_jpeg.rf.3d16051acd7758e469ce40844d4d4232.jpg",
        "originales/train/Poliwhirl/235e136ad0564826a1fd12410ff87d75_jpg.rf.57d3eecb703799a6838e77886902866d.jpg",
    ),
    (
        "originales/train/Alakazam/64527ef991ff42c6883623f263f3c2f4_jpg.rf.451e51efab802819d98ca130742e797c.jpg",
        "originales/validation/Alakazam/64527ef991ff42c6883623f263f3c2f4_jpg.rf.993a72b9ff17505a03cee4b4bdbb4cce.jpg",
    ),
    (
        "originales/validation/Raichu/469ad0e3e1994a109ff1b2c968dd535c_jpg.rf.0ca07dc332e508fd8cf739bb4970fc84.jpg",
        "originales/validation/Raichu/d646480481424765b5c01d3e0294d815_jpg.rf.4e1e1b1be245d6f90859af7e49f4826e.jpg",
    ),
    (
        "originales/validation/Vulpix/c7a27dd6c23c4e069a8d3fa6d5449456_jpeg.rf.ed0294e2d95941fe136994824ca58d27.jpg",
        "originales/validation/Vulpix/c7a27dd6c23c4e069a8d3fa6d5449456_jpg.rf.39182d57c738dddf5bf926b86a930ca3.jpg",
    ),
    (
        "originales/test/Scyther/29a4a5a77fc142148145fdc7ecd78976_jpg.rf.875f672ad04ed3e88cd122fa30ad467b.jpg",
        "originales/test/Scyther/e8ff2e5ae9c24ef39d0bb3e8055e1982_jpg.rf.ca2f48715d697103b55e7e9180fd773a.jpg",
    ),
}

PARES_DISTINTOS_VERIFICADOS = set()

EXCLUIR_IMAGENES_REVISION = {
    "originales/train/Poliwrath/7ecc511d19ea1ba7d94944991d0ee76a_jpg.rf.67b47155eec2af0f9ca0971e59c7fab4.jpg",
    "originales/validation/Poliwhirl/468156a4e3b8454daa9118c7a7dd7ec7_jpg.rf.df35c7e11bedeb8106593235405850bb.jpg",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def guardar_csv(ruta, filas, columnas):
    with Path(ruta).open(
        "w", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=columnas,
        )
        escritor.writeheader()
        escritor.writerows(filas)


def respaldar_auditoria_anterior():
    """Respaldar salidas anteriores sin tocar las imágenes."""
    nombres = (
        "manifiesto.csv",
        "clases.txt",
        "auditoria.txt",
        "auditoria_aprobada.txt",
        "duplicados_exactos.csv",
        "similares.csv",
        "rechazadas.csv",
        "excluidas_revision.csv",
    )

    existentes = [
        CARPETA_DATOS / nombre
        for nombre in nombres
        if (CARPETA_DATOS / nombre).exists()
    ]

    if not existentes:
        return

    # No reconstruir particiones después de entrenar.
    if Path(CARPETA_RESULTADOS).exists():
        if any(Path(CARPETA_RESULTADOS).rglob("*.keras")):
            raise RuntimeError(
                "Ya existen modelos guardados. "
                "No reconstruyas las particiones de ese experimento. "
                "Utiliza otra copia del proyecto para un reparto nuevo."
            )

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    respaldo = CARPETA_DATOS / f"respaldo_auditoria_{fecha}"
    respaldo.mkdir()

    for archivo in existentes:
        shutil.move(
            str(archivo),
            str(respaldo / archivo.name),
        )

    print(f"Auditoría anterior respaldada en: {respaldo}")


def extraer_zip_seguro(archivo, destino):
    """Conservada para compatibilidad con las pruebas del proyecto."""
    destino = Path(destino).resolve()

    if destino.exists():
        raise FileExistsError(f"Ya existe: {destino}")

    with zipfile.ZipFile(archivo) as contenido:
        for entrada in contenido.infolist():
            nombre = PurePosixPath(entrada.filename)

            if (
                nombre.is_absolute()
                or ".." in nombre.parts
                or "\\" in entrada.filename
            ):
                raise ValueError("Ruta insegura dentro del ZIP.")

            tipo = (entrada.external_attr >> 16) & 0o170000

            if tipo == 0o120000:
                raise ValueError("Enlace simbólico dentro del ZIP.")

        destino.mkdir(parents=True)
        contenido.extractall(destino)


# ============================================================
# ETL
# ============================================================

def ejecutar_etl():
    origen = CARPETA_DATOS / "originales"
    splits = ("train", "validation", "test")

    clases_por_split = {}

    for split in splits:
        carpeta = origen / split

        if not carpeta.is_dir():
            raise FileNotFoundError(
                f"Falta la carpeta de imágenes: {carpeta}"
            )

        clases_por_split[split] = {
            ruta.name
            for ruta in carpeta.iterdir()
            if ruta.is_dir()
        }

    clases = sorted(
        set().union(*clases_por_split.values())
    )

    if len(clases) < 2:
        raise ValueError("Se requieren al menos dos clases.")

    faltantes_train = (
        set(clases) - clases_por_split["train"]
    )

    if faltantes_train and not REPARTIR_POR_GRUPOS:
        raise ValueError(
            "Hay clases ausentes de train. "
            "Debes reconstruir el reparto antes de entrenar."
        )

    respaldar_auditoria_anterior()

    etiquetas = {
        clase: indice
        for indice, clase in enumerate(clases)
    }

    print(f"Clases encontradas: {len(clases)}")
    print("Leyendo imágenes...")

    filas = []
    rechazadas = []
    excluidas = []

    for split in splits:
        for carpeta in sorted((origen / split).iterdir()):
            if not carpeta.is_dir():
                continue

            for ruta in sorted(carpeta.rglob("*")):
                if (
                    not ruta.is_file()
                    or ruta.suffix.lower()
                    not in {".jpg", ".jpeg", ".png"}
                ):
                    continue

                relativa = ruta.relative_to(
                    CARPETA_DATOS
                ).as_posix()

                if relativa in EXCLUIR_IMAGENES_REVISION:
                    excluidas.append({
                        "ruta": relativa,
                        "motivo": (
                            "Misma imagen con etiquetas "
                            "Poliwhirl/Poliwrath contradictorias"
                        ),
                        "sha256_archivo": hash_archivo(ruta),
                    })
                    continue

                try:
                    imagen = leer_rgb(str(ruta))

                    alto, ancho = (
                        tf.shape(imagen)[:2]
                        .numpy()
                        .tolist()
                    )

                    contenido = tf.io.encode_png(imagen).numpy()

                    hash_pixeles = hashlib.sha256(
                        contenido
                    ).hexdigest()

                    gris = tf.image.rgb_to_grayscale(
                        tf.cast(imagen, tf.float32)
                    )

                    miniatura = tf.image.resize(
                        gris, (8, 9)
                    )

                    bits = tf.reshape(
                        miniatura[:, 1:] > miniatura[:, :-1],
                        [-1],
                    ).numpy().tolist()

                    hash_perceptual = sum(
                        int(bit) << posicion
                        for posicion, bit in enumerate(bits)
                    )

                    filas.append({
                        "ruta": relativa,
                        "split": split,
                        "split_original": split,
                        "clase": carpeta.name,
                        "etiqueta": etiquetas[carpeta.name],
                        "ancho": ancho,
                        "alto": alto,
                        "hash_pixeles": hash_pixeles,
                        "sha256_archivo": hash_archivo(ruta),
                        "hash_perceptual": (
                            f"{hash_perceptual:016x}"
                        ),
                    })

                except Exception as error:
                    rechazadas.append({
                        "ruta": relativa,
                        "motivo": str(error),
                    })

    if not filas:
        raise ValueError("No se encontraron imágenes válidas.")

    padres = list(range(len(filas)))

    def buscar(indice):
        while padres[indice] != indice:
            padres[indice] = padres[padres[indice]]
            indice = padres[indice]
        return indice

    def unir(a, b):
        padres[buscar(a)] = buscar(b)

    revisados_iguales = {
        tuple(sorted(par))
        for par in PARES_MISMA_FUENTE
    }

    revisados_distintos = {
        tuple(sorted(par))
        for par in PARES_DISTINTOS_VERIFICADOS
    }

    if revisados_iguales & revisados_distintos:
        raise ValueError(
            "Un par está marcado como igual y distinto."
        )

    indices_por_ruta = {
        fila["ruta"]: indice
        for indice, fila in enumerate(filas)
    }

    for ruta_a, ruta_b in revisados_iguales:
        if (
            ruta_a in indices_por_ruta
            and ruta_b in indices_por_ruta
        ):
            unir(
                indices_por_ruta[ruta_a],
                indices_por_ruta[ruta_b],
            )

    exactos = {}

    for indice, fila in enumerate(filas):
        clave = fila["hash_pixeles"]

        if clave in exactos:
            unir(indice, exactos[clave])

        exactos[clave] = indice

    print("Revisando similitud entre imágenes...")

    hashes = [
        int(fila["hash_perceptual"], 16)
        for fila in filas
    ]

    candidatos = []

    for indice, fila in enumerate(filas):
        for anterior in range(indice):
            otra = filas[anterior]

            if fila["hash_pixeles"] == otra["hash_pixeles"]:
                continue

            distancia = (
                hashes[indice] ^ hashes[anterior]
            ).bit_count()

            if distancia > DISTANCIA_HASH:
                continue

            par = tuple(sorted((
                fila["ruta"],
                otra["ruta"],
            )))

            if par in revisados_iguales:
                decision = "misma_fuente"
            elif par in revisados_distintos:
                decision = "distintas"
            else:
                decision = "pendiente"

            candidatos.append({
                "ruta_a": par[0],
                "ruta_b": par[1],
                "distancia": distancia,
                "decision": decision,
                "indice_a": indice,
                "indice_b": anterior,
            })

    grupos = {}

    for indice, fila in enumerate(filas):
        grupo = filas[buscar(indice)]["ruta"]
        fila["grupo"] = grupo
        grupos.setdefault(grupo, []).append(fila)

    conflictos_etiqueta = [
        miembros
        for miembros in grupos.values()
        if len({
            fila["clase"]
            for fila in miembros
        }) > 1
    ]

    if REPARTIR_POR_GRUPOS and not conflictos_etiqueta:
        for etiqueta in range(len(clases)):
            identificadores = sorted(
                grupo
                for grupo, miembros in grupos.items()
                if miembros[0]["etiqueta"] == etiqueta
            )

            random.Random(
                SEMILLA_REPARTO + etiqueta
            ).shuffle(identificadores)

            cantidad = len(identificadores)

            n_validacion = (
                max(1, round(cantidad * 0.15))
                if cantidad >= 3 else 0
            )

            n_test = (
                max(1, round(cantidad * 0.15))
                if cantidad >= 3 else 0
            )

            n_validacion = min(
                n_validacion,
                max(0, cantidad - 2),
            )

            n_test = min(
                n_test,
                cantidad - n_validacion - 1,
            )

            for posicion, grupo in enumerate(
                identificadores
            ):
                if posicion < n_validacion:
                    split = "validation"
                elif posicion < n_validacion + n_test:
                    split = "test"
                else:
                    split = "train"

                for fila in grupos[grupo]:
                    fila["split"] = split

    conflictos_split = [
        miembros
        for miembros in grupos.values()
        if len({
            fila["split"]
            for fila in miembros
        }) > 1
    ]

    pendientes = [
        candidato
        for candidato in candidatos
        if (
            candidato["decision"] == "pendiente"
            and filas[candidato["indice_a"]]["split"]
            != filas[candidato["indice_b"]]["split"]
        )
    ]

    conservar = []
    duplicados = []
    vistos = set()

    for fila in filas:
        clave = (
            fila["hash_pixeles"],
            fila["etiqueta"],
        )

        if clave in vistos:
            duplicados.append(fila)
        else:
            conservar.append(fila)
            vistos.add(clave)

    columnas = list(filas[0])

    guardar_csv(
        CARPETA_DATOS / "manifiesto.csv",
        conservar,
        columnas,
    )

    guardar_csv(
        CARPETA_DATOS / "duplicados_exactos.csv",
        duplicados,
        columnas,
    )

    guardar_csv(
        CARPETA_DATOS / "rechazadas.csv",
        rechazadas,
        ["ruta", "motivo"],
    )

    guardar_csv(
        CARPETA_DATOS / "excluidas_revision.csv",
        excluidas,
        ["ruta", "motivo", "sha256_archivo"],
    )

    guardar_csv(
        CARPETA_DATOS / "similares.csv",
        [
            {
                clave: valor
                for clave, valor in candidato.items()
                if not clave.startswith("indice_")
            }
            for candidato in candidatos
        ],
        ["ruta_a", "ruta_b", "distancia", "decision"],
    )

    (CARPETA_DATOS / "clases.txt").write_text(
        "\n".join(clases) + "\n",
        encoding="utf-8",
    )

    conteos = Counter(
        fila["split"] for fila in conservar
    )

    etiquetas_train = {
        fila["etiqueta"]
        for fila in conservar
        if fila["split"] == "train"
    }

    resumen = (
        f"Clases: {len(clases)}\n"
        f"Conteos finales: {dict(conteos)}\n"
        f"Rechazadas: {len(rechazadas)}\n"
        f"Excluidas por etiquetas contradictorias: "
        f"{len(excluidas)}\n"
        f"Grupos con conflicto de etiqueta: "
        f"{len(conflictos_etiqueta)}\n"
        f"Grupos cruzados: {len(conflictos_split)}\n"
        f"Pares cruzados sin revisar: {len(pendientes)}\n"
        f"Reparto nuevo por grupos: "
        f"{REPARTIR_POR_GRUPOS}\n"
        f"Semilla reparto: {SEMILLA_REPARTO}\n"
    )

    (CARPETA_DATOS / "auditoria.txt").write_text(
        resumen,
        encoding="utf-8",
    )

    print("\n" + resumen)

    hay_problemas = (
        bool(conflictos_etiqueta)
        or bool(conflictos_split)
        or bool(pendientes)
        or len(etiquetas_train) != len(clases)
        or len(conteos) != 3
    )

    if hay_problemas:
        raise ValueError(
            "Auditoría pendiente. Revisa auditoria.txt "
            "y similares.csv antes de entrenar."
        )

    certificado = (
        hash_archivo(CARPETA_DATOS / "manifiesto.csv")
        + "\n"
        + hash_archivo(CARPETA_DATOS / "clases.txt")
        + "\n"
    )

    (CARPETA_DATOS / "auditoria_aprobada.txt").write_text(
        certificado,
        encoding="utf-8",
    )

    print("ETL terminado. Auditoría aprobada.")
    print("Ahora puedes ejecutar: python python/cnn.py")

    return conservar


if __name__ == "__main__":
    ejecutar_etl()
