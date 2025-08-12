from setuptools import setup, find_packages

setup(
    name='sequence-former',
    version='0.1.0',
        packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'sequence-former = sequence_former.main:main',
        ],
    },
)
