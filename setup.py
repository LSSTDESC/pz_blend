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
      requires=[
          "numpy",
          "scipy",
          "dask", 
          "dask_ml", 
          "pandas", 
          "seaborn", 
          # "logging", 
          "FoFCatalogMatching", 
          "fast3tree", 
          "joblib", 
          "sklearn", 
          "tqdm",
          "skgof",
      ],
      install_requires=[
          "numpy",
          "scipy",
          "dask", 
          "dask_ml", 
          "pandas", 
          "seaborn", 
          # "logging", 
          "FoFCatalogMatching", 
          "fast3tree", 
          "joblib", 
          "sklearn", 
          "tqdm",
          "skgof",
      ],
      dependency_links=['git+https://github.com/cosmicshear/fast3tree',
                        'git+https://github.com/cosmicshear/FoFCatalogMatching'],
      zip_safe=False)

# the following does not work anymore :/ do it manually
# E.N. (cosmicshear) : use --process-dependency-link with pip to force my customizad versions of fast3tree and FoFCatalogMatching with MPI and progressbar
