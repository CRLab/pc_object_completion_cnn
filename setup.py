## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup



d = generate_distutils_setup()

d['name'] = "shape_completion_server"
d['description'] = "shape completion server"
d['packages'] = ['shape_completion_server']
d['package_dir'] = {'': 'scripts'}



setup(**d)

