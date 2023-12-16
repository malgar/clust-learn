from setuptools import setup, find_packages
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md')) as f:
    README = f.read()


setup(
    name='clust-learn',
    version="0.2.5",
    description="A Python package for explainable cluster analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Miguel Alvarez-Garcia, Raquel Ibar-Alonso, Mar Arenas-Parra",
    author_email="@gmail.com",
    url="https://github.com/malgar/clust-learn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    license='GPLv3',
    install_requires=[
	    "imbalanced-learn>=0.10.0",
        "kneed>=0.7.0",
        "matplotlib>=3.4.3",
        "networkx>=2.6.3",
        "numpy>=1.20.3",
        "pandas>=1.3.4,<2.0",
        "pingouin>=0.5.3",
        "prince==0.10.4",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.1",
        "seaborn>=0.11.2",
        "shap>=0.40.0",
        "statsmodels>=0.13.2",
        "xgboost>=1.5.2"
    ],
    python_requires='>=3.9'
)
