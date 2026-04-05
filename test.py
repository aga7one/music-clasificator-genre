import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # memoria dinámica para no reservar toda la VRAM de golpe
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    print(f"GPU detectada: {gpus[0].name}")
else:
    print("No se detectó GPU, usando CPU")

