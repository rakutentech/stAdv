#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import setup


# heavily borrows from Kenneth Reitz's A Human's Ultimate Guide to setup.py
# see https://github.com/kennethreitz/setup.py

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

package_name = 'stadv'
# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, package_name, '__version__.py')) as f:
    exec(f.read(), about)


setup(
    name=package_name,
    version=about['__version__'],
    description='Spatially Transformed Adversarial Examples with TensorFlow',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Beranger Dumont',
    author_email='beranger.dumont@rakuten.com',
    url='https://github.com/rakutentech/stAdv',
    license='MIT',
    packages=[package_name],
    python_requires='>=2.7',
    keywords='tensorflow adversarial examples CNN deep learning',
    # install_requires without tensorflow because of CPU vs. GPU install issues
    install_requires=['numpy', 'scipy'],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
    ],
    project_urls={
        'Documentation': 'http://stadv.readthedocs.io/en/latest/stadv.html',
        'Source': 'https://github.com/rakutentech/stAdv',
        'Tracker': 'https://github.com/rakutentech/stAdv/issues'
    }
)
