'''
Date: 2021-03-22 15:21:27
LastEditors: Chenhuiyu
LastEditTime: 2021-09-01 11:27:07
FilePath: \\2021-07-AttenEmotionNet\\callbacks.py
'''

import os

import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

from metrics import get_confusion_matrix, get_epoch_metrics, print_metrics


def create_callbacks(output_dir, logger, model, testsamples, labels):
    # def create_callbacks(output_dir):
    # 创建检查点的路径和检查点管理器（manager）。这将用于在每 n个周期（epochs）保存检查点。
    # 保存checkpoints
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'checkpoint', 'weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
    )

    # save the confuse matrix
    confuse_matrix_dir = os.path.join(output_dir, 'confuse_matrix')
    if not os.path.exists(confuse_matrix_dir):
        os.mkdir(confuse_matrix_dir)

    class metrics_Callback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            save_path = os.path.join(confuse_matrix_dir, 'val_{:2d}-{:.2f}.png'.format(epoch, logs['val_accuracy']))
            predictions = np.argmax(model.predict(testsamples, batch_size=16), axis=-1)
            assert len(predictions) == len(labels)
            y_ture = labels
            y_pred = predictions
            metrics_list = get_epoch_metrics(y_true=y_ture, predictions=y_pred)
            print_metrics(logger, epoch, logs, metrics_list)
            get_confusion_matrix(y_true=y_ture, y_pred=y_pred, save_path=save_path, is_test=False)

    tensorboard = TensorBoard(log_dir=output_dir)

    callbacks_list = [checkpointer, tensorboard, metrics_Callback()]

    return callbacks_list
