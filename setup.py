#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'pandas>=1.2.4',
    'loguru>=0.7.0',
    'omegaconf>=2.3.0'
]

test_requirements = [ ]

setup(
    author="Mikhail Repin",
    author_email='repin.mihail.09@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A package for controlled experiments.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='TestLab',
    name='TestLab',
    packages=find_packages(include=['TestLab', 'TestLab.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/repinmihail/TestLab.git',
    version='1.0.0',
    zip_safe=False,
)
