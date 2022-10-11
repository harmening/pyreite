from setuptools import setup, find_packages

setup(name='pyreite',
      version='0.1',
      description='Pythonic, Yet Rudimentary, EIT expert',
      url='https://github.com/harmening/pyreite',
      author='Nils Harmening',
      author_email='nils.harmening@tu-berlin.de',
      license='GNU General Public License v3.0',
      #packages=['pyreite'],
      packages=find_packages(include=['pyreite', 'pyreite.*']),
      zip_safe=False)
