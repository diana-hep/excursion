#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import sys

this_directory = path.abspath(path.dirname(__file__))
if sys.version_info.major < 3:
    from io import open
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as readme_md:
    long_description = readme_md.read()

extras_require = {
    'develop': [
        'jupyter',
        'matplotlib',
        'pyflakes',
        'pytest>=3.5.1',
        'pytest-cov>=2.5.1',
        'papermill'
    ],
    'plots': [
      'matplotlib',
      'scikit-image'
    ],
    'parallel': [
      'joblib'
    ],
    'baselines': [
      'pyDOE'
    ]
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='excursion',
    version='0.0.1',
    description='excursion set estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/diana-hep/excursion',
    author='Lukas Heinrich',
    author_email='lukas.heinrich@cern.ch',
    license='Apache',
    keywords='bayesian optimization excursion set estimation',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    install_requires=[
        'sklearn',
        'scipy',
        'pyDOE',
        'click>=6.0',
        'sklearn',
        'pyyaml'
    ],
    extras_require=extras_require,
    entry_points={'console_scripts': ['excursion=excursion.commandline:main']},
    dependency_links=[],
)
