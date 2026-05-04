from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='pyreite',
      version=get_version("pyreite/__init__.py"),
      description='Pythonic, Yet Robust, EIT expert',
      long_description=read("README.md"),
      long_description_content_type='text/markdown',
      url='https://github.com/harmening/pyreite',
      author='Nils Harmening',
      author_email='nils.harmening@tu-berlin.de',
      license='GNU General Public License v3.0',
      #packages=['pyreite'],
      packages=find_packages(include=['pyreite', 'pyreite.*']),
      python_requires='>=3.7',
      install_requires=[
          'numpy>=1.21.6',
          'h5py>=3.8.0',
          'scipy>=1.7.3',
      ],
      zip_safe=False)
