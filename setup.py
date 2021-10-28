import setuptools

setuptools.setup(name='notmad',
      packages=['notmad'],
      version='0.0.0',
      install_requires=[
          'tensorflow>=2.4.0',
          'tensorflow-addons',
          'numpy>=1.19.2',
          'tqdm',
          'scikit-learn',
          'python-igraph',
      ],
)
