from os import path
from codecs import open
from setuptools import setup, find_packages
from rosey_keras import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')


install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]


setup(
    name='rosey-keras',
    version=__version__,
    url='https://github.com/arose13/rosey-keras',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='',
    packages=find_packages(),
    author='Stephen Rose',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='me@stephenro.se',
    zip_safe=False
)