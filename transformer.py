from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import MultiHeadAttention, Dropout, LayerNormalization, Conv1D, TimeDistributed, GlobalAveragePooling1D, Dense

n_classes = 3


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(LayerNormalization(epsilon=1e-6))(x)
    res = x + inputs

    # Feed Forward Part
    x = TimeDistributed(Conv1D(filters=ff_dim, kernel_size=1, activation="relu"))(res)
    x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(Conv1D(filters=inputs.shape[-1], kernel_size=1))(x)
    x = TimeDistributed(LayerNormalization(epsilon=1e-6))(x)
    return x + res


def build_transformer(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = TimeDistributed(GlobalAveragePooling1D(data_format="channels_first"))(x)
    for dim in mlp_units:
        x = TimeDistributed(Dense(dim, activation="relu"))(x)
        x = TimeDistributed(Dropout(mlp_dropout))(x)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    outputs = Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)