import tensorflow as tf
from tensorpack import *
from tensorpack.utils import logger
from qenet import qenet
from dataflow import ModelNetDataFlow


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, 64, 9, 3], dtype=tf.float32, name='input_points'),
                tf.TensorSpec([None, 64, 9, 4], dtype=tf.float32, name='input_quat')]

    def build_graph(self, x, q):
        capsule = q[:, :, :, None]
        activation = tf.ones(tf.shape(q)[:-1])

        capsule, activation = qenet(x, capsule, activation, [64, 64], 64)
        return tf.reduce_sum(capsule)

    def get_optimizer(self):
        return tf.train.AdamOptimizer(0)


if __name__ == '__main__':
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset = BatchData(MultiProcessPrefetchData(ModelNetDataFlow(), 6, 6), 2)

    # How many iterations you want in each epoch.
    # This len(data) is the default value.
    steps_per_epoch = len(dataset)

    # get the config which contains everything necessary in a training
    config = TrainConfig(
        model=Model(),
        dataflow=dataset,
        callbacks=[
            ModelSaver(),  # save the model after every epoch
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )
    launch_train_with_config(config, SimpleTrainer())
