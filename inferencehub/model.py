import tensorflow as tf


def get_model(weights_path) -> tf.keras.Model:
    return tf.keras.models.load_model(weights_path)
