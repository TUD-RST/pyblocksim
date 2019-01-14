## GENERAL INFORMATION

Pyblocksim aims to mitigate the lack of a tool like Simulink (or scicos)
in the world of python based scientific computing.
It aims to enable you to quickly implement a model of a dynamic system
which is suited to be modeled by the feedback-free
directed interconnection of blocks such as transfer functions.

pyblocksim provides a framework for:

- easily describing blocks and their interconnections (including explicit feedback)
- converting the whole model into a state space form
- passing it to the numerical integrator of scipy
- easily accessing the simulation results

Pyblocksim does *not* provide a graphical user interface.
Model description has to be done textually.


Currently, only little development work has been invested.
What already works is the following:

- specifying a model consisting of linear proper transfer functions
  (i.e. degree(num)<= degree(denominator)) and (possibly nonlinear)
  interconnections of such
-  describing system inputs as python functions
- simulate (i.e. numerically integrate) the models with the given inputs
- collecting the simulation results and make them available for e.g.
  visualization.


The following features should be possible to be implemented

-  representation of systems with dead time
-  representation of (linear and nonlinear) systems in state space
-  graphical visualization of the blocks and their interconnections
  (for visual model checking)
-  things like nyquist and bode plots
- a module for controller/observer design

HINT:
    Some of these features (and many others) are available in python-control,
    see: http://python-control.sourceforge.net/


## Warning

There is absolutely NO GUARANTEE that the produced results are correct.
There was only a quick plausibility check for the examples by the author.

## Installation

The command
```
pip install pyblocksim
```

should install the package including all examples.
If something goes wrong, feel free to file an issue or contact the author(s).

## Getting started
The recommended entry point ist the `examples`-folder.
*Note*: the examples are not installed with pip and have to be downloaded
 separately.

## Dependencies

Since 2016-11-09 this package uses python3 syntax.
The python2.7.x code is still available in the branch `leagacy-py2`.

Besides the Python standard library pyblocksim depends on

- scipy, numpy
- sympy
- matplotlib (for visualization only)

When installing with `pip` the dependencies should be automatically
installed. See also `requirements.txt`.


## Documentation

Currently no handwritten documentation exists.
The examples should be self-explainatory and
cover the whole functionality. Tanking a glance at the source code should
reveal the internals (at least if the meaning of the terminology can be
guessed). In the case of questions asking for support is encouraged.


## License

Pyblocksim is released under the terms of the GNU GPLv3.
See the `LICENSE`-file for details.


## Contact and Feedback

Any feedback (suggestions, bug reports, etc) is highly welcome.

https://github.com/cknoll/pyblocksim/wiki/Contact
