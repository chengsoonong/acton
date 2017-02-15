#!/usr/bin/env python3

from setuptools import setup

from acton import __version__

with open('README.rst') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = [
]

test_requirements = [
    'flake8==3.2.0',
    'pyflakes>=1.3.0',
    'nose==1.3.7',
]

setup(
    name='acton',
    version=__version__,
    description="A scientific research assistant",
    long_description=readme,  # + '\n\n' + history,
    url='https://github.com/chengsoonong/acton',
    # Setup scripts don't support multiple authors, so this should be the main
    # author or the author that should be contacted regarding the module.
    author='Cheng Soon Ong',
    author_email='chengsoon.ong@anu.edu.au',
    packages=[
        'acton',
    ],
    package_dir={'acton':
                 'acton'},
    entry_points={
        'console_scripts': [
            'acton=acton.cli:main',
            'acton-predict=acton.cli:predict',
            'acton-recommend=acton.cli:recommend',
            'acton-label=acton.cli:label',
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='machine-learning active-learning classification regression',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
