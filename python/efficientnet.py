"""EfficientNetB0: transferencia con ImageNet y fine-tuning opcional."""
import hashlib
import tensorflow as tf
from procesamiento import TAMANO_IMAGEN, leer_clases
from entrenamiento import iniciar_corrida, entrenar_etapa, terminar_corrida, registrar_fallo

# PARÁMETROS MANUALES.
SEMILLA = 42
BATCH_SIZE = 8
EPOCAS_CABEZA = 40
PACIENCIA = 6
LEARNING_RATE_CABEZA = 0.001
PESOS = "imagenet"  
HACER_FINE_TUNING = True
EPOCAS_FINE_TUNING = 20
LEARNING_RATE_FINE_TUNING = 0.00001


def construir_efficientnet(numero_clases, pesos=PESOS):
    capas = tf.keras.layers
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=pesos, input_shape=(TAMANO_IMAGEN, TAMANO_IMAGEN, 3), pooling="avg")
    base.trainable = False
    entrada = capas.Input((TAMANO_IMAGEN, TAMANO_IMAGEN, 3), name="rgb_0_255")
    x = base(entrada, training=False)
    x = capas.Dense(128, activation="relu")(x)
    x = capas.Dropout(0.3)(x)
    salida = capas.Dense(numero_clases, activation="softmax")(x)
    return tf.keras.Model(entrada, salida, name="efficientnet_clasificador"), base


def descongelar_ultimo_bloque(base):
    base.trainable = True
    nombres = []
    for capa in base.layers:
        capa.trainable = (capa.name.startswith("block7") or capa.name.startswith("top_")) and not isinstance(capa, tf.keras.layers.BatchNormalization)
        if capa.trainable and capa.weights:
            nombres.append(capa.name)
    if not nombres:
        raise ValueError("No se encontró el bloque esperado; revisar versión de Keras")
    return nombres


def ejecutar_efficientnet():
    destino = iniciar_corrida("efficientnet", SEMILLA, {
        "batch_size": BATCH_SIZE, "epocas_cabeza": EPOCAS_CABEZA, "paciencia": PACIENCIA,
        "learning_rate_cabeza": LEARNING_RATE_CABEZA, "pesos": PESOS,
        "fine_tuning": HACER_FINE_TUNING, "epocas_fine_tuning": EPOCAS_FINE_TUNING,
        "learning_rate_fine_tuning": LEARNING_RATE_FINE_TUNING})
    try:
        modelo, base = construir_efficientnet(len(leer_clases()), PESOS)
        digest = hashlib.sha256()
        for variable in base.weights:
            digest.update(variable.numpy().tobytes())
        (destino / "hash_pesos_iniciales.txt").write_text(digest.hexdigest()+"\n")
        modelo.summary()
        perdida_cabeza = entrenar_etapa(modelo, destino, "cabeza", SEMILLA, BATCH_SIZE,
                                        EPOCAS_CABEZA, PACIENCIA, LEARNING_RATE_CABEZA)
        modelo.save(destino / "solo_cabeza.keras")
        elegida = "cabeza"
        if HACER_FINE_TUNING:
            nombres = descongelar_ultimo_bloque(base)
            (destino / "capas_fine_tuning.txt").write_text("\n".join(nombres)+"\n")
            perdida_fine = entrenar_etapa(modelo, destino, "fine_tuning", SEMILLA, BATCH_SIZE,
                                          EPOCAS_FINE_TUNING, PACIENCIA, LEARNING_RATE_FINE_TUNING)
            if perdida_fine < perdida_cabeza:
                elegida = "fine_tuning"
            else:
                modelo = tf.keras.models.load_model(destino / "solo_cabeza.keras", compile=False)
        (destino / "etapa_elegida_validacion.txt").write_text(elegida+"\n")
        terminar_corrida(modelo, destino, BATCH_SIZE)
    except Exception as error:
        registrar_fallo(destino, error)
        raise


if __name__ == "__main__":
    ejecutar_efficientnet()
