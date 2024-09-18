from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='toytrack',
    version='0.1.14',
    url='https://github.com/murnanedaniel/ToyTrack',
    author='Daniel Murnane',
    author_email='daniel.thomas.murnane@cern.ch',
    description='A package for generating toy tracking data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['numpy >= 2.0.0', 'pandas >= 2.0.0', 'matplotlib >= 3.8.0'],
    extras_require={
        'torch': ['torch', 'pyyaml'],
    },
)