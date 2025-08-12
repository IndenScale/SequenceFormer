from setuptools import setup, find_packages

setup(
    name='textformer',
    version='0.1.0',
        packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'textformer = textformer.main:main',
        ],
    },
)
