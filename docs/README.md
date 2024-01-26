This directory contains the documentation.


Run `pip install -r requirements.txt` in this directory to install the dependencies to build the documentation.

Run `make html` in this directory to build the docs on your system. See `build/html` for the result.


Automatically generated documentation is available at: <https://pyblocksim.readthedocs.io>. (Be sure to select the branch of interest from the menu in the lower right).


### Debug Readthedocs Build

In `/docs/source` run `python -m sphinx -T -E -W --keep-going -b html -d _build/doctrees -D language=en . build/html`.
