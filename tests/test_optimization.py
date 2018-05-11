from .context import stadv, call_assert

import tensorflow as tf
import numpy as np


class LBFGSCase(tf.test.TestCase):
    """Test the lbfgs optimization function.
    Note: we are NOT testing the LBFGS implementation from SciPy, instead we
    test our wrapping and its interplay with TensorFlow."""

    def setUp(self):
        self.example_flow = np.array([[0.5, 0.4], [-0.2, 0.7]])
        self.flows = tf.Variable(self.example_flow, name='flows')
        self.loss_l2 = tf.reduce_sum(tf.square(self.flows), name='loss_l2')
        self.loss_dummy = tf.Variable(1.4, name='loss_dummy')

        tf.global_variables_initializer()

    def test_l2_norm_loss(self):
        """Check that simple L2 loss leads to 0 loss and gradient in the end."""
        results = stadv.optimization.lbfgs(
            self.loss_l2, self.flows, flows_x0=self.example_flow
        )
        call_assert(
            self.assertEqual,
            results['flows'].shape, self.example_flow.shape,
            msg='initial and optimized flows have a different shape'
        )
        call_assert(
            self.assertAllClose,
            results['flows'], np.zeros(results['flows'].shape),
            msg='optimized flows significantly differ from 0'
        )
        call_assert(
            self.assertAllClose,
            results['loss'], np.zeros(results['loss'].shape),
            msg='final gradients significantly differ from 0'
        )

    def test_dummy_loss(self):
        """Make sure a dummy loss (no computable gradient) gives an error."""
        with self.assertRaises(ValueError):
            stadv.optimization.lbfgs(
                self.loss_dummy, self.flows, flows_x0=self.example_flow
            )

    def test_overwriting_optimized_function(self):
        """Make sure we cannot overwrite argument defining the function to
        optimize."""
        with self.assertRaises(ValueError):
            stadv.optimization.lbfgs(
                self.loss_dummy, self.flows, flows_x0=self.example_flow,
                fmin_l_bfgs_b_extra_kwargs={'func': np.sqrt}
            )

if __name__ == '__main__':
    tf.test.main()
