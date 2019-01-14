# -*- coding: utf-8 -*-

from setuptools import setup
from pyblocksim.release import __version__

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

setup(
    name='pyblocksim',
    version=__version__,
    author='Carsten Knoll',
    author_email='Carsten.Knoll@tu-dresden.de',
    packages=['pyblocksim'],
    url='https://github.com/TUD-RST/pycartan',
    license='GPL3',
    description='Python library for block-oriented modeling and simlation in control theory',
    long_description="""Pyblocksim aims to mitigate the lack of a tool like Simulink (or scicos)
in the world of python based scientific computing.
It aims to enable you to quickly implement a model of a dynamic system
which is suited to be modeled by the feedback-free
directed interconnection of blocks such as transfer functions.
    """,
    keywords='control theory, simulation, modelling, feedback',
    install_requires=requirements,
)
