from setuptools import setup, find_packages

setup(
    name="real_estate_ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'scikit-learn',
        'pandas',
        'joblib'
    ],
)