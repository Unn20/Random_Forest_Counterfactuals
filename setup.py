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
    name='rf_counterfactuals',
    packages=find_packages(),
    version='0.1.3',
    description='Random Forest counterfactual explanation',
    keywords='counterfactual, explainable, xai, random forest',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Maciej Leszczyk',
    author_email="maciej.leszczyk98@email.com",
    url='https://github.com/Unn20/Random_Forest_Counterfactuals',
    license='CC 4.0',
    python_requires='>=3.8',
    install_requires=['numpy>=1.23.1', 'pandas>=1.4.3', 'scikit-learn>=1.1.2', 'scipy>=1.9.0', 'joblib>=1.1.0'],
    setup_requires=['pytest-runner==6.0.0'],
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)
