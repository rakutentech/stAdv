"""Load the stadv package (try to do so explicitly) to be agnostic of the
installation status of the package."""

import sys
import os
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import stadv

# next is a trick to have unit tests run for TensorFlow < 1.7, when the msg
# argument was not always present in the assert methods of tf.test.TestCase
import inspect

def call_assert(f, *args, **kwargs):
    if 'msg' not in inspect.getargspec(f) and 'msg' in kwargs.keys():
        kwargs.pop('msg')
    return f(*args, **kwargs)
