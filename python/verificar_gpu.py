import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPU:", gpus)
if not gpus:
    raise SystemExit("No se detecta GPU. Revisa driver NVIDIA y WSL2 antes de entrenar.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device('/GPU:0'):
    x = tf.random.normal((512, 512))
    y = tf.matmul(x, x)
print("Operación realizada en:", y.device)
print("Resultado finito:", bool(tf.reduce_all(tf.math.is_finite(y))))
