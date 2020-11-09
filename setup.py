from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='axelerate',
      version="0.6.0",
      description='Keras-based framework for AI on the Edge',
      install_requires=requirements,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Dmitry Maslov',
      author_email='dmitrywat@gmail.com',
      url='https://github.com/AIWintermuteAI',
      packages=find_packages(),
     )
