from setuptools import setup

setup(
    name='gsc',
    version='0.1.0',
    py_modules=['gsc'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'gsc = main:cli',
        ],
    },
)