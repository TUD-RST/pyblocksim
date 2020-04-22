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
    description='Python library for block-oriented modeling and simulation in control theory',
    long_description="""Pyblocksim aims to mitigate the lack of a tool like Simulink (or scicos)
in the world of python based scientific computing. It aims to enable an low-effort implementation
of a model of a dynamical system which is composed of interconnected blocks such as rational
transfer functions and static nonlinearities. Delay blocks are supported via ring-buffers
and discrete states can also be emulated (e.g. for hysteresis).
    """,
    keywords='control theory, simulation, modelling, feedback',
    install_requires=requirements,
)
