from setuptools import find_packages, setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rfcounterfactuals',
    packages=find_packages(include=['rfcounterfactuals']),
    version='0.1.0',
    description='Random Forest counterfactual explanation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Maciej Leszczyk',
    author_email="maciej.leszczyk98@email.com",
    license='MIT',
    install_requires=['numpy>=1.22.4', 'pandas>=1.4.2'],
    setup_requires=['pytest-runner==6.0.0'],
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)