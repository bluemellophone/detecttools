#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from setuptools import setup


if __name__ == '__main__':
    kwargs = setuptools_setup(
        setup_fpath=__file__,
        name='detecttools',
        description=('Utilities for writing detectors (like pyrf)'),
        url='https://github.com/bluemellophone/pyrf',
        author='Jason Parham',
        author_email='bluemellophone@gmail.com',
        packages=['detecttools',
                  'detecttools.ctypes_interface',
                  'detecttools.directory',
                  'detecttools.ibeisdata',
                  'detecttools.pypascalxml',
                  ],
        py_modules=['detecttools'],
    )
    setup(**kwargs)
