import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# DATA
INPUT_SHAPE = (25, 9, 9, 5)
NUM_CLASSES = 3

# TUBELET EMBEDDING
PATCH_SIZE = (3, 3, 1)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0])**2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 8


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim=9, patch_size=2, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim=9, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim=self.embed_dim)
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # inputs = keras.layers.GaussianNoise(0.1)(inputs)
    # inputs = keras.layers.BatchNormalization()(inputs)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        x1 = layers.Dropout(0.2)(x1)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.2,
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential([
            layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu, kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(units=embed_dim, activation=tf.nn.gelu, kernel_regularizer=keras.regularizers.l2(0.01)),
        ])(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    representation = layers.Dropout(0.5)(representation)
    representation = keras.layers.BatchNormalization()(representation)
    outputs = layers.Dense(units=num_classes, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01))(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
