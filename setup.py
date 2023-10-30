from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='toytrack',
    version='0.1.3',
    url='https://github.com/yourusername/toytrack',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'pandas >= 0.18.1'],
)