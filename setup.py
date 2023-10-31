from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='toytrack',
    version='0.1.5',
    url='https://github.com/murnanedaniel/ToyTrack',
    author='Author Name',
    author_email='dtmurnane@lbl.gov',
    description='A package for generating toy tracking data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'pandas >= 0.18.1'],
)