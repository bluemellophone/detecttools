#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup


if __name__ == '__main__':
    setuptools_setup(
        setup_fpath=__file__,
        name='detecttools',
        version='1.0.0.dev1',
        description=('Utilities for writing detectors (like pyrf)'),
        url='https://github.com/bluemellophone/pyrf',
        author='Jason Parham',
        author_email='bluemellophone@gmail.com',
        packages=['detecttools'],
        py_modules=['detecttools'],
    )
