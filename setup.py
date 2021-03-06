from setuptools import setup

setup(name='pzblend',
      version='0.1',
      description='Impact of blending on photo-zs',
      long_description='Python module for studying the impact of galaxy blending on photo-zs using LSST DESC DC2 truth catalogs and image catalogs.',
      url='https://github.com/LSSTDESC/pz_blend',
      author='LSST DESC Photoz Blending Group',
      author_email='lsstdesc.org',
      license='MIT',
      packages=['pzblend'],
      install_requires=[
          "numpy",
          "scipy",
          "dask", 
          "dask-ml", 
          "pandas>=0.25.3", 
          "seaborn", 
          "FoFCatalogMatching @ git+https://github.com/enourbakhsh/FoFCatalogMatching", 
          "fast3tree @ git+https://github.com/enourbakhsh/fast3tree", 
          "joblib", 
          "scikit-learn>=0.22.1", 
          "tqdm",
      ],
      zip_safe=False)
