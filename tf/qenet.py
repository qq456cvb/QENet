import tensorflow as tf
from tensorpack import *


def quat_dist(a, b):
    '''

    :param a:
    :param b:
    :return:
    '''
    return 2 * tf.acos(tf.reduce_sum(a * b, -1))  # there should be no abs


def quat_prod(a, b):
    '''

    :param a: ... x 4
    :param b: ... x 4
    :return:
    '''
    return tf.concat([a[..., :1] * b[..., :1] - tf.reduce_sum(a[..., 1:] * b[..., 1:], -1, keepdims=True),
                      a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + tf.linalg.cross(a[..., 1:], b[..., 1:])], -1)


def quat_avg(quats, weights=None):
    '''
    average
    :param quats: ... x K x 4
    :param dim:
    :param weights:  x K
    :return:
    '''
    if weights is not None:
        cov = tf.matmul(tf.matmul(quats, tf.linalg.diag(weights), transpose_a=True), quats)
    else:
        cov = tf.matmul(quats, quats, transpose_a=True)

    _, v = tf.self_adjoint_eig(cov)
    return v[..., -1]


def qedr(input_t, input_capsule, input_a, k=3):
    '''

    :param input_t: B x N x K x Nc x M x 4
    :param input_capsule: B x N x K x Nc x 4
    :param input_a: B x N x K x Nc
    :param k: iterations
    :return:
    '''
    shape = input_t.get_shape().as_list()
    votes = quat_prod(tf.tile(input_capsule[..., None, :], [1, 1, 1, 1, shape[-2], 1]), input_t)  # B x N x K x Nc x M x 4
    votes = tf.transpose(votes, perm=[0, 1, 4, 2, 3, 5])
    votes = tf.reshape(votes, [-1, shape[1], shape[4], shape[2] * shape[3], 4])  # B x N x M x KNc x 4
    activation = tf.reshape(input_a, [-1, shape[1], 1, shape[2] * shape[3]])  # B x N x 1 x KNc

    # with tf.control_dependencies([tf.print(tf.shape(tf.broadcast_to(activation, tf.shape(votes)[:-1])), tf.shape(votes))]):
    output_capsule = quat_avg(votes, tf.broadcast_to(activation, tf.shape(votes)[:-1]))  # B x N x M x 4
    for _ in range(1):
        assignment = activation * tf.sigmoid(-quat_dist(tf.stop_gradient(output_capsule[..., None, :]), votes))  # B x N x M x KNc
        output_capsule = quat_avg(tf.reshape(votes, (-1, 9, 4)), tf.reshape(assignment, (-1, 9)))  # B x N x M x 4
        with tf.control_dependencies([tf.print(tf.shape(assignment), tf.shape(output_capsule))]):
            output_capsule = tf.identity(output_capsule)
    output_activation = tf.sigmoid(-tf.reduce_mean(quat_dist(output_capsule[..., None, :], votes), -1))
    return output_capsule, output_activation


def qenet(input_x, input_capsule, input_a, transform_units, output_channels):
    '''
    Quaternion Equivariant Network
    :param input_x: B x N x K x 3
    :param input_capsule: B x N x K x Nc x 4
    :param input_a: B x N x K x Nc
    :param transform_units:
    :param output_channels:
    :return:
    '''
    shape = input_capsule.get_shape().as_list()
    miu = quat_avg(tf.transpose(input_capsule, perm=[0, 1, 3, 2, 4]))  # B x N x Nc x 4
    x = tf.concat([tf.zeros([tf.shape(input_x)[0], shape[1], shape[2], 1]), input_x], -1)  # B x N x K x 4

    miu_broad = tf.broadcast_to(tf.concat([miu[..., :1], -miu[..., 1:]], -1)[..., None, :, :], tf.shape(input_capsule))
    x_broad = tf.broadcast_to(x[..., None, :], tf.shape(input_capsule))
    x = quat_prod(quat_prod(miu_broad, x_broad), tf.broadcast_to(miu[..., None, :, :], tf.shape(input_capsule)))  # we need conjugate for rotation

    # x should have 0 on real axis
    x = x[..., 1:]  # B x N x K x Nc x 3
    x = tf.reshape(x, [-1, shape[1], shape[2], shape[3] * 3])
    for i, unit in enumerate(transform_units):
        x = tf.layers.dense(x, unit, activation=tf.nn.relu, name='FC%d' % i)

    t = tf.layers.dense(x, shape[3] * output_channels * 4, name='FC_out')
    t = tf.reshape(t, [-1, shape[1], shape[2], shape[3], output_channels, 4])
    output_capsule, output_activation = qedr(t, input_capsule, input_a)
    return output_capsule, output_activation



