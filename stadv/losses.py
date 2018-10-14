import tensorflow as tf


def flow_loss(flows, padding_mode='SYMMETRIC', epsilon=1e-8):
    """Computes the flow loss designed to "enforce the locally smooth
    spatial transformation perturbation". See Eq. (4) in Xiao et al.
    (arXiv:1801.02612).
    
    Args:
        flows (tf.Tensor): flows of shape `(B, 2, H, W)`, where the second
                           dimension indicates the dimension on which the pixel
                           shift is applied.
        padding_mode (str): how to perform padding of the boundaries of the
                            images. The value should be compatible with the
                            `mode` argument of ``tf.pad``. Expected values are:

                            * ``'SYMMETRIC'``: symmetric padding so as to not
                              penalize a significant flow at the boundary of
                              the images;
                            * ``'CONSTANT'``: 0-padding of the boundaries so as
                              to enforce a small flow at the boundary of the
                              images.
        epsilon (float): small value added to the argument of ``tf.sqrt``
                         to prevent NaN gradients when the argument is zero.

    Returns:
         1-D `tf.Tensor` of length `B` of the same type as `flows`.
    """
    with tf.variable_scope('flow_loss'):
        # following the notation from Eq. (4):
        # \Delta u^{(p)} is flows[:, 1],
        # \Delta v^{(p)} is flows[:, 0], and
        # \Delta u^{(q)} is flows[:, 1] shifted by
        # (+1, +1), (+1, -1), (-1, +1), or (-1, -1) pixels
        # and \Delta v^{(q)} is the same but for shifted flows[:, 0]

        paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
        padded_flows = tf.pad(
            flows, paddings, padding_mode, constant_values=0,
            name='padded_flows'
        )

        shifted_flows = [
            padded_flows[:, :, 2:, 2:],  # bottom right
            padded_flows[:, :, 2:, :-2],  # bottom left
            padded_flows[:, :, :-2, 2:],  # top right
            padded_flows[:, :, :-2, :-2]  # top left
        ]

        return tf.reduce_sum(
            tf.add_n(
                [
                    tf.sqrt(
                        # ||\Delta u^{(p)} - \Delta u^{(q)}||_2^2
                        (flows[:, 1] - shifted_flow[:, 1]) ** 2 +
                        # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2
                        (flows[:, 0] - shifted_flow[:, 0]) ** 2 +
                        epsilon  # for numerical stability
                    )
                    for shifted_flow in shifted_flows
                ]
            ), axis=[1, 2], name='L_flow'
        )

def adv_loss(unscaled_logits, targets, kappa=None):
    """Computes the adversarial loss.
    It was first suggested by Carlini and Wagner (arXiv:1608.04644).
    See also Eq. (3) in Xiao et al. (arXiv:1801.02612).

    Args:
        unscaled_logits (tf.Tensor): logits of shape `(B, K)`, where `K` is the
                                     number of input classes.
        targets (tf.Tensor): `1-D` integer-encoded targets of length `B` with
                             value corresponding to the class ID.
        kappa (tf.Tensor): confidence parameter, see Carlini and Wagner
                           (arXiv:1608.04644). Defaults to 0.

    Returns:
        1-D `tf.Tensor` of length `B` of the same type as `unscaled_logits`.
    """
    if kappa is None:
        kappa = tf.constant(0., dtype=unscaled_logits.dtype, name='kappa')

    with tf.variable_scope('adv_loss'):
        unscaled_logits_shape = tf.shape(unscaled_logits)
        B = unscaled_logits_shape[0]
        K = unscaled_logits_shape[1]

        # first term in L_adv: maximum of the (unscaled) logits except target
        mask = tf.one_hot(
            targets,
            depth=K,
            on_value=False,
            off_value=True,
            dtype='bool'
        )
        logit_wout_target = tf.reshape(
            tf.boolean_mask(unscaled_logits, mask),
            (B, K - 1),
            name='logit_wout_target'
        )
        L_adv_1 = tf.reduce_max(logit_wout_target, axis=1, name='L_adv_1')

        # second term in L_adv: value of the unscaled logit corresponding to the
        # target
        L_adv_2 = tf.diag_part(
            tf.gather(unscaled_logits, targets, axis=1), name='L_adv_2'
        )

        return tf.maximum(L_adv_1 - L_adv_2, - kappa, name='L_adv')
