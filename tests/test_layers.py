from .context import stadv, call_assert

import tensorflow as tf
import numpy as np
import skimage.transform


class FlowStCase(tf.test.TestCase):
    """Test the flow_st layer."""

    def setUp(self):
        np.random.seed(0)  # predictable random numbers

        # dimensions of the input images -- "random" shape for generic cases
        self.N, self.H, self.W, self.C = 2, 7, 6, 2
        self.data_formats = {
            'NHWC': (self.N, self.H, self.W, self.C),
            'NCHW': (self.N, self.C, self.H, self.W)
        }

        # generate test images and flows
        self.test_images = {}
        self.test_images_float = {}
        for data_format, dim_tuple in self.data_formats.items():
            self.test_images[data_format] = np.random.randint(0, 256, dim_tuple)
            self.test_images_float[data_format] = (
                self.test_images[data_format] / 255.
            )
        flow_shape = (self.N, 2, self.H, self.W)
        self.flow_zero = np.zeros(flow_shape, dtype=int)
        self.flow_random = np.random.random_sample(flow_shape) - 0.5

        # building a minimal graph
        # setting None shape in both placeholders because we want to test
        # a number of possible shape issues
        self.images = tf.placeholder(tf.float32, shape=None, name='images')
        self.flows = tf.placeholder(tf.float32, shape=None, name='flows')
        self.outputs = {}
        for data_format in self.data_formats.keys():
            self.outputs[data_format] = stadv.layers.flow_st(
                self.images, self.flows, data_format
            )

    def test_output_shape_consistency(self):
        """Check that the input images and output images shapes are the same."""
        with self.test_session():
            for data_format, output in self.outputs.items():
                call_assert(
                    self.assertEqual,
                    output.eval(feed_dict={
                        self.images: self.test_images[data_format],
                        self.flows: self.flow_random
                    }).shape,
                    self.test_images[data_format].shape,
                    msg='the output shape differs from the input one for shape '
                        + data_format
                )

    def test_mismatch_flow_shape_image_shape(self):
        """Make sure that input flows with a wrong shape raise the expected
        Exception."""
        flow_zeros_wrongdim1 = np.zeros(
            (self.N, 2, self.H + 1, self.W), dtype=int
        )
        flow_zeros_wrongdim2 = np.zeros(
            (self.N, 2, self.H, self.W - 1), dtype=int
        )

        with self.test_session():
            for data_format, output in self.outputs.items():
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    output.eval(feed_dict={
                        self.images: self.test_images[data_format],
                        self.flows: flow_zeros_wrongdim1
                    })
                    output.eval(feed_dict={
                        self.images: self.test_images[data_format],
                        self.flows: flow_zeros_wrongdim2
                    })

    def test_noflow_consistency(self):
        """Check that no flow displacement gives the input image."""
        with self.test_session():
            for data_format, output in self.outputs.items():
                call_assert(
                    self.assertAllEqual,
                    output.eval(feed_dict={
                        self.images: self.test_images[data_format],
                        self.flows: self.flow_zero
                    }),
                    self.test_images[data_format],
                    msg='output differs from input in spite of 0 displacement '
                        'flow for shape ' + data_format + ' and int input'
                )

    def test_noflow_consistency_float_image(self):
        """Check that no flow displacement gives the float input image."""
        with self.test_session():
            for data_format, output in self.outputs.items():
                call_assert(
                    self.assertAllClose,
                    output.eval(feed_dict={
                        self.images: self.test_images_float[data_format],
                        self.flows: self.flow_zero
                    }),
                    self.test_images_float[data_format],
                    msg='output differs from input in spite of 0 displacement '
                        'flow for shape ' + data_format + ' and float input'
                )

    def test_min_max_output_consistency(self):
        """Test that the min and max values in the output do not exceed the
        min and max values in the input images for a random flow."""
        with self.test_session():
            # test both data formats
            for data_format, output in self.outputs.items():
                in_images = self.test_images[data_format]
                out_images = output.eval(
                    feed_dict={
                        self.images: in_images,
                        self.flows: self.flow_random
                    }
                )

                # derive min and max values for every image in the batch
                # for the input and output
                taxis = (1,2,3)
                minval_in = np.amin(in_images, axis=taxis)
                maxval_in = np.amax(in_images, axis=taxis)
                minval_out = np.amin(out_images, axis=taxis)
                maxval_out = np.amax(out_images, axis=taxis)

                call_assert(
                    self.assertTrue,
                    np.all(np.less_equal(minval_in, minval_out)),
                    msg='min value in output image less than min value in input'
                )
                call_assert(
                    self.assertTrue,
                    np.all(np.greater_equal(maxval_in, maxval_out)),
                    msg='max value in output image exceeds max value in input'
                )

    def test_bilinear_interpolation(self):
        """Test that the home-made bilinear interpolation matches the one from
        scikit-image."""
        data_format = 'NHWC'
        in_image = self.test_images[data_format][0]

        translation_x = 1.5
        translation_y = 0.8

        # define the transformation with scikit-image
        tform = skimage.transform.EuclideanTransform(
            translation=(translation_x, translation_y)
        )

        skimage_out = skimage.transform.warp(
            in_image / 255., tform, order=1
        ) * 255.

        # do the same with our tool
        constant_flow_1 = np.zeros((self.H, self.W)) + translation_y
        constant_flow_2 = np.zeros((self.H, self.W)) + translation_x
        stacked_flow = np.stack([constant_flow_1, constant_flow_2], axis=0)
        final_flow = np.expand_dims(stacked_flow, axis=0)
        with self.test_session():
            tf_out = self.outputs[data_format].eval(
                feed_dict={
                    self.images: np.expand_dims(in_image, axis=0),
                    self.flows: final_flow
                }
            )[0]

        # we only want to check equality up to boundary effects
        cut_x = - np.ceil(translation_x).astype(int)
        cut_y = - np.ceil(translation_y).astype(int)
        skimage_out_crop = skimage_out[:cut_y, :cut_x]
        tf_out_crop = tf_out[:cut_y, :cut_x]

        call_assert(
            self.assertAllClose,
            tf_out_crop, skimage_out_crop,
            msg='bilinear interpolation differs from scikit-image'
        )

if __name__ == '__main__':
    tf.test.main()
