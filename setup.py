from setuptools import setup
  
with open("READMEpypi.md", "r") as fh:
    long_description = fh.read()

setup(
      name='fdc',
      version='1.01',
      description='Fast Densitiy Clustering in low-dimension',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['fdc'],
      install_requires =['scikit-learn>=0.19'],
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
