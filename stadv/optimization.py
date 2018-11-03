import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def lbfgs(
    loss, flows, flows_x0, feed_dict=None, grad_op=None,
    fmin_l_bfgs_b_extra_kwargs=None, sess=None
):
    """Optimize a given loss with (SciPy's external) L-BFGS-B optimizer.
    It can be used to solve the optimization problem of Eq. (2) in Xiao et al.
    (arXiv:1801.02612).
    See `the documentation on scipy.optimize.fmin_l_bfgs_b
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_
    for reference on the optimizer.

    
    Args:
        loss (tf.Tensor): loss (can be of any shape).
        flows (tf.Tensor): flows of shape `(B, 2, H, W)`, where the second
                           dimension indicates the dimension on which the pixel
                           shift is applied.
        flows_x0 (np.ndarray): Initial guess for the flows. If the input is not
                               of type `np.ndarray`, it will be converted as
                               such if possible.
        feed_dict (dict): feed dictionary to the ``tf.run`` operation (for
                          everything  which might be needed to execute the graph
                          beyond the input flows).
        grad_op (tf.Tensor): gradient of the loss with respect to the flows.
                             If not provided it will be computed from the input
                             and added to the graph.
        fmin_l_bfgs_b_extra_kwargs (dict): extra arguments to
                                           ``scipy.optimize.fmin_l_bfgs_b``
                                           (e.g. for modifying the stopping
                                           condition).
        sess (tf.Session): session within which the graph should be executed.
                           If not provided a new session will be started.

    Returns:
        `Dictionary` with keys ``'flows'`` (`np.ndarray`, estimated flows of the
        minimum), ``'loss'`` (`float`, value of loss at the minimum), and
        ``'info'`` (`dict`, information summary as returned by
        ``scipy.optimize.fmin_l_bfgs_b``).
    """
    def tf_run(x):
        """Function to minimize as provided to ``scipy.optimize.fmin_l_bfgs_b``.

        Args:
            x (np.ndarray): current flows proposal at a given stage of the
                            optimization (flattened `np.ndarray` of type
                            `np.float64` as required by the backend FORTRAN
                            implementation of L-BFGS-B).

        Returns:
            `Tuple` `(loss, loss_gradient)` of type `np.float64` as required
            by the backend FORTRAN implementation of L-BFGS-B.
        """
        flows_val = np.reshape(x, flows_shape)

        feed_dict.update({flows: flows_val})
        loss_val, gradient_val = sess_.run(
            [loss, loss_gradient], feed_dict=feed_dict
        )
        loss_val = np.sum(loss_val).astype(np.float64)
        gradient_val = gradient_val.flatten().astype(np.float64)

        return loss_val, gradient_val

    flows_x0 = np.asarray(flows_x0, dtype=np.float64)
    flows_shape = flows_x0.shape

    if feed_dict is None:
        feed_dict = {}
    if fmin_l_bfgs_b_extra_kwargs is None:
        fmin_l_bfgs_b_extra_kwargs = {}

    fmin_l_bfgs_b_kwargs = {
        'func': tf_run,
        'approx_grad': False,  # we want to use the gradients from TensorFlow
        'fprime': None,
        'args': ()
    }

    for key in fmin_l_bfgs_b_extra_kwargs.keys():
        if key in fmin_l_bfgs_b_kwargs:
            raise ValueError(
                "The argument " + str(key) + " should not be overwritten by "
                "fmin_l_bfgs_b_extra_kwargs"
            )

    # define the default extra arguments to fmin_l_bfgs_b
    default_extra_kwargs = {
        'x0': flows_x0.flatten(),
        'factr': 10.0,
        'm': 20,
        'iprint': -1
    }

    fmin_l_bfgs_b_kwargs.update(default_extra_kwargs)
    fmin_l_bfgs_b_kwargs.update(fmin_l_bfgs_b_extra_kwargs)

    if grad_op is not None:
        loss_gradient = grad_op
    else:
        loss_gradient = tf.gradients(loss, flows, name='loss_gradient')[0]
        if loss_gradient is None:
            raise ValueError(
                "Cannot compute the gradient d(loss)/d(flows). Is the graph "
                "really differentiable?"
            )

    sess_ = tf.Session() if sess is None else sess
    raw_results = fmin_l_bfgs_b(**fmin_l_bfgs_b_kwargs)
    if sess is None:
        sess_.close()

    return {
        'flows': np.reshape(raw_results[0], flows_shape),
        'loss': raw_results[1],
        'info': raw_results[2]
    }
