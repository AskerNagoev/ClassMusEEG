from setuptools import setup, find_packages

setup(
    name='classmuseeg',
    version='0.1.0',
    description='EEG classification and feature extraction tools',
    author='Asker Nagoev',
    packages=find_packages(),
    py_modules=['data_preprocessing', 'feature_extraction', 'model_operations'],
    install_requires=[
        'numpy>=1.24.4',
        'pandas>=1.5.3',
        'scipy>=1.10.1',
        'scikit-learn>=1.2.2',
        'seaborn>=0.12.2',
        'matplotlib>=3.7.1',
        'optuna>=3.1.1',
        'tensorflow>=2.12.0',
        'pywavelets>=1.4.1',
    ],
    python_requires='>=3.8',
) 