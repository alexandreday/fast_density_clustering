""" from distutils.core import setup
from Cython.Build import cythonize

setup(name='fdc',
      version='0.1',
      description='Clustering in low dimensional space',
      url='https://github.com/alexandreday/fast_density_clustering',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['fdc'],
      zip_safe=False,
      )

"""
from setuptools import setup
  
with open("READMEpypi.md", "r") as fh:
    long_description = fh.read()

setup(
      name='fdc',
      version='0.99',
      description='Fast Densitiy Clustering in low dimensional space',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['fdc'],
      install_requires =['scikit-learn'],
      zip_safe=False,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://alexandreday.github.io/",
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
