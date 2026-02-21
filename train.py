import os
import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from transformer import build_transformer
from logger import get_logger
from callbacks import create_callbacks

# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# DATA
BATCH_SIZE = 64

# OPTIMIZER
LEARNING_RATE = 1e-3

# TRAINING
EPOCHS = 300


@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess tensors and labels for Transformer input."""
    # Keep shape as (bands, time, channels) for 1D DE features.
    frames = tf.cast(frames, tf.float32)
    label = tf.cast(label, tf.int32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))
    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2, seed=SEED, reshuffle_each_iteration=True)

    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def normalize_labels(train_label, val_label, test_label):
    all_labels = np.concatenate([train_label, val_label, test_label])
    min_label = int(all_labels.min())
    max_label = int(all_labels.max())

    # Common SEED setups: labels in {1,2,3} or {0,1,2}
    if min_label == 1 and max_label == 3:
        return train_label - 1, val_label - 1, test_label - 1
    if min_label == 0 and max_label == 2:
        return train_label, val_label, test_label

    raise ValueError(f"Unexpected label range: [{min_label}, {max_label}]. Expected [1,3] or [0,2].")


def prepare_data(data_type="1d"):
    if data_type == "1d":
        data_path = "./input_data_1d/"
    else:
        data_path = "./input_data/"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}. Please generate npy files first.")

    def shuffle_dataset(data, label):
        shuffle_index = np.arange(len(data))
        np.random.shuffle(shuffle_index)
        return data[shuffle_index], label[shuffle_index]

    train_data = np.load(os.path.join(data_path, "train_data.npy"))
    train_label = np.load(os.path.join(data_path, "train_label.npy")).astype(np.int32)
    val_data = np.load(os.path.join(data_path, "val_data.npy"))
    val_label = np.load(os.path.join(data_path, "val_label.npy")).astype(np.int32)
    test_data = np.load(os.path.join(data_path, "test_data.npy"))
    test_label = np.load(os.path.join(data_path, "test_label.npy")).astype(np.int32)

    train_label, val_label, test_label = normalize_labels(train_label, val_label, test_label)

    train_data, train_label = shuffle_dataset(train_data, train_label)
    val_data, val_label = shuffle_dataset(val_data, val_label)
    test_data, test_label = shuffle_dataset(test_data, test_label)

    # class_weight keys must match normalized class ids [0,1,2]
    classes = np.unique(train_label)
    weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=train_label)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    trainloader = prepare_dataloader(train_data, train_label, "train")
    validloader = prepare_dataloader(val_data, val_label, "valid")
    testloader = prepare_dataloader(test_data, test_label, "test")

    return trainloader, validloader, testloader, class_weight_dict, test_label


def random_baseline_accuracy(test_labels: np.ndarray, n_classes: int = 3):
    preds = np.random.RandomState(SEED).randint(0, n_classes, size=len(test_labels))
    return float((preds == test_labels).mean())


def run_experiment(output_path, logger, trainloader, validloader, testloader, class_weight_dict):
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

    with open(os.path.join(output_path, "model_summary.log"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    NUM_SAMPLES_TEST = 128
    testsamples, labels = next(iter(testloader))
    testsamples, labels = testsamples[:NUM_SAMPLES_TEST], labels[:NUM_SAMPLES_TEST]

    callbacks_list = create_callbacks(
        output_dir=output_path,
        logger=logger,
        model=model,
        testsamples=testsamples,
        labels=labels,
    )

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    history = model.fit(
        trainloader,
        epochs=EPOCHS,
        validation_data=validloader,
        callbacks=callbacks_list,
        class_weight=class_weight_dict,
    )

    loss, accuracy = model.evaluate(testloader)
    logger.info(f"test loss: {loss}")
    logger.info(f"test accuracy: {accuracy}")
    logger.info(history.history)
    return model, float(accuracy)


if __name__ == "__main__":
    tf.get_logger().setLevel(logging.ERROR)

    output_path_root = "save"
    localtime = time.strftime("%m%d%H%M", time.localtime())
    output_path = os.path.join(output_path_root, localtime)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, "train.log")
    logger = get_logger(log_file, level="info")
    logger.info("-----------------------log created-------------------------")
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    trainloader, validloader, testloader, class_weight_dict, test_label = prepare_data(data_type="1d")
    baseline_acc = random_baseline_accuracy(test_label, n_classes=3)
    logger.info(f"random baseline accuracy: {baseline_acc:.4f}")

    _, test_acc = run_experiment(output_path, logger, trainloader, validloader, testloader, class_weight_dict)
    logger.info(f"final test accuracy: {test_acc:.4f}")
