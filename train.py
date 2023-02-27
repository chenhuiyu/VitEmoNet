import os
import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
from transformer import build_transformer
from vivit import *
from logger import get_logger
from callbacks import create_callbacks
# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# keras.utils.set_random_seed(SEED)

# DATA
DATASET_NAME = "SEED"
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE

# OPTIMIZER
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 300


@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[..., tf.newaxis],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return dataloader


def prepare_data(data_type='1d'):
    if data_type == "1d":
        data_path = './input_data_1d/'
        label_path = './input_data_1d/'
    else:
        data_path = './input_data/'
        label_path = './input_data/'

    def shuffle_dataset(data, label):
        shuffle_index = list(range(len(data)))
        np.random.shuffle(shuffle_index)
        X = data[shuffle_index]
        Y = label[shuffle_index]

        return X, Y

    train_data = np.load(os.path.join(data_path, "train_data.npy"))
    train_label = np.load(os.path.join(label_path, "train_label.npy"))
    train_data, train_label = shuffle_dataset(train_data, train_label)
    # 计算类别权重
    my_class_weight = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label).tolist()
    # 需要转成字典
    class_weight_dict = dict(zip([x for x in np.unique(train_label)], my_class_weight))
    val_data = np.load(os.path.join(data_path, "val_data.npy"))
    val_label = np.load(os.path.join(data_path, "val_label.npy"))
    val_data, val_label = shuffle_dataset(val_data, val_label)

    test_data = np.load(os.path.join(data_path, "test_data.npy"))
    test_label = np.load(os.path.join(data_path, "test_label.npy"))
    test_data, test_label = shuffle_dataset(test_data, test_label)

    trainloader = prepare_dataloader(train_data, train_label, "train")
    validloader = prepare_dataloader(val_data, val_label, "valid")
    testloader = prepare_dataloader(test_data, test_label, "test")

    return trainloader, validloader, testloader, class_weight_dict


def run_experiment(output_path, logger, trainloader, validloader, testloader):
    # Initialize model
    # model = create_vivit_classifier(
    #     tubelet_embedder=TubeletEmbedding(embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE),
    #     positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    # )

    input_shape = (5, 25, 62)

    model = build_transformer(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    with open(os.path.join(output_path, 'model_summary.log'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    tf.keras.utils.plot_model(
        model=model,
        to_file=os.path.join(output_path, 'model.png'),
        show_shapes=True,
        show_layer_names=False,
        dpi=300,
    )

    NUM_SAMPLES_TEST = 128
    testsamples, labels = next(iter(testloader))
    testsamples, labels = testsamples[:NUM_SAMPLES_TEST], labels[:NUM_SAMPLES_TEST]
    # Create callbacks
    callbacks_list = create_callbacks(
        output_dir=output_path,
        logger=logger,
        model=model,
        testsamples=testsamples,
        labels=labels,
    )

    # # Compile the model with the optimizer, loss function
    # # and the metrics.
    # optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=10000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Train the model.
    history = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader, callbacks=callbacks_list, class_weight=class_weight_dict)

    loss, accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    logger.info('test loss:{}'.format(loss))
    logger.info('accuracy:{}'.format(accuracy))
    logger.info(history.history)
    return model


if __name__ == '__main__':
    # Create logger
    tf.get_logger().setLevel(logging.ERROR)
    log_file = os.path.join("save", 'train.log')
    logger = get_logger(log_file, level="info")
    logger.info("-----------------------log created-------------------------")
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("log file in :{}".format(log_file))

    output_path_root = "save"
    localtime = time.strftime("%m%d%H%M", time.localtime())
    output_path = os.path.join(output_path_root, localtime)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trainloader, validloader, testloader, class_weight_dict = prepare_data()

    run_experiment(output_path, logger, trainloader, validloader, testloader)