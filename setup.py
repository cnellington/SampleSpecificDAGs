from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='notmad',
      py_modules=['notmad'],
      install_requires=[
          'tensorflow>=2.4.0',
          'tensorflow-addons',
          'numpy>=1.19.2',
          'tqdm',
          'scikit-learn',
          'python-igraph',
      ],
)
