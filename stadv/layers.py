import tensorflow as tf


def flow_st(images, flows, data_format='NHWC'):
    """Flow-based spatial transformation of images.
    See Eq. (1) in Xiao et al. (arXiv:1801.02612).
    
    Args:
        images (tf.Tensor): images of shape `(B, H, W, C)` or `(B, C, H, W)`
                            depending on `data_format`.
        flows (tf.Tensor): flows of shape `(B, 2, H, W)`, where the second
                           dimension indicates the dimension on which the pixel
                           shift is applied.
        data_format (str): ``'NHWC'`` or ``'NCHW'`` depending on the format of
                           the input images and the desired output.

    Returns:
         `tf.Tensor` of the same shape and type as `images`.
    """
    if data_format == 'NHWC':
        i_H = 1
    elif data_format == 'NCHW':
        i_H = 2
    else:
        raise ValueError("Provided data_format is not valid.")

    with tf.variable_scope('flow_st'):
        images_shape = tf.shape(images)
        flows_shape = tf.shape(flows)

        batch_size = images_shape[0]
        H = images_shape[i_H]
        W = images_shape[i_H + 1]

        # make sure that the input images and flows have consistent shape
        with tf.control_dependencies(
            [tf.assert_equal(
                tf.identity(images_shape[i_H:i_H + 2], name='images_shape_HW'),
                tf.identity(flows_shape[2:], name='flows_shape_HW')
            )]
        ):
            # cast the input to float32 for consistency with the rest
            images = tf.cast(images, 'float32', name='images_float32')
            flows = tf.cast(flows, 'float32', name='flows_float32')

            if data_format == 'NCHW':
                images = tf.transpose(images, [0, 2, 3, 1])

            # basic grid: tensor with shape (2, H, W) with value indicating the
            # pixel shift in the x-axis or y-axis dimension with respect to the
            # original images for the pixel (2, H, W) in the output images,
            # before applying the flow transforms
            basegrid = tf.stack(
                tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
            )

            # go from (2, H, W) tensors to (B, 2, H, W) tensors with simple copy
            # across batch dimension
            batched_basegrid = tf.tile([basegrid], [batch_size, 1, 1, 1])

            # sampling grid is base grid + input flows
            sampling_grid = tf.cast(batched_basegrid, 'float32') + flows

            # separate shifts in x and y is easier--also we clip to the
            # boundaries of the image
            sampling_grid_x = tf.clip_by_value(
                sampling_grid[:, 1], 0., tf.cast(W - 1, 'float32')
            )
            sampling_grid_y = tf.clip_by_value(
                sampling_grid[:, 0], 0., tf.cast(H - 1, 'float32')
            )

            # now we need to interpolate

            # grab 4 nearest corner points for each (x_i, y_i)
            # i.e. we need a square around the point of interest
            x0 = tf.cast(tf.floor(sampling_grid_x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(sampling_grid_y), 'int32')
            y1 = y0 + 1

            # clip to range [0, H/W] to not violate image boundaries
            # - 2 for x0 and y0 helps avoiding black borders
            # (forces to interpolate between different points)
            x0 = tf.clip_by_value(x0, 0, W - 2, name='x0')
            x1 = tf.clip_by_value(x1, 0, W - 1, name='x1')
            y0 = tf.clip_by_value(y0, 0, H - 2, name='y0')
            y1 = tf.clip_by_value(y1, 0, H - 1, name='y1')

            # b is a (B, H, W) tensor with (B, H, W) = B for all (H, W)
            b = tf.tile(
                tf.reshape(
                    tf.range(0, batch_size), (batch_size, 1, 1)
                ),
                (1, H, W)
            )

            # get pixel value at corner coordinates
            # we stay indices along the last dimension and gather slices
            # given indices
            # the output is of shape (B, H, W, C)
            Ia = tf.gather_nd(images, tf.stack([b, y0, x0], 3), name='Ia')
            Ib = tf.gather_nd(images, tf.stack([b, y1, x0], 3), name='Ib')
            Ic = tf.gather_nd(images, tf.stack([b, y0, x1], 3), name='Ic')
            Id = tf.gather_nd(images, tf.stack([b, y1, x1], 3), name='Id')

            # recast as float for delta calculation
            x0 = tf.cast(x0, 'float32')
            x1 = tf.cast(x1, 'float32')
            y0 = tf.cast(y0, 'float32')
            y1 = tf.cast(y1, 'float32')

            # calculate deltas
            wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
            wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
            wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
            wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

            # add dimension for addition
            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)

            # compute output
            perturbed_image = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

            if data_format == 'NCHW':
                # convert back to NCHW to have consistency with the input
                perturbed_image = tf.transpose(perturbed_image, [0, 3, 1, 2])

            return perturbed_image
