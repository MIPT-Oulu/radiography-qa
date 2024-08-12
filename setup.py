# Allows installation as a package, using command 'pip install .'

from setuptools import setup, find_packages

setup(
    name='normi13_qa',
    version='0.1',
    description='Quality assurance of computed radiography devices.',
    author='Santeri Rytky',
    author_email='santeri.rytky@pohde.fi',
    url='',
    packages=find_packages(),
    include_package_data=False,
    install_requires=open('requirements.txt').read(),
    license='LICENSE',
    long_description=open('README.md').read(),
)
