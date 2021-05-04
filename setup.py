#!/usr/bin/env python

from setuptools import setup
import sys

# assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
#     "Safety Starter Agents is designed to work with Python 3.6 and greater. " \
#     + "Please install it before proceeding."

setup(
    name='rl_library',
    packages=['algorithms', 'runners', 'utils'],
    # install_requires=[
    #     'matplotlib==3.1.3',
    #     'numpy~=1.18.1',
    #     'seaborn==0.10.0',
    # ],
)