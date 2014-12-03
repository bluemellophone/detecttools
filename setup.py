#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from setuptools import setup
from utool import util_setup

setupkw = dict(
    setup_fpath=__file__,
    name='detecttools',
    version=util_setup.parse_package_for_version('detecttools'),
    licence=util_setup.read_license('LICENSE'),
    long_description=util_setup.parse_readme('README.md'),
    description=('Utilities for writing detectors (like pyrf)'),
    url='https://github.com/bluemellophone/pyrf',
    author='Jason Parham',
    author_email='bluemellophone@gmail.com',
    #packages=util_setup.find_packages(),
    packages=['detecttools',
              'detecttools.ctypes_interface',
              'detecttools.directory',
              'detecttools.ibeisdata',
              'detecttools.pypascalmarkup',
              ],
    py_modules=['detecttools'],
)

if __name__ == '__main__':
    # Preprocess and setup kwargs for real setup
    kwargs = util_setup.setuptools_setup(**setupkw)
    setup(**kwargs)
