RET
===

`ret` is a package that extends the functionality of the `mesa` package to simplify and enhance the use for analysts and combat modellers.
ret stands for Rapid Exploratory modelling Toolset.


Requirements
------------------
The use of this package requires Python. This distribution supports python versions >= 3.9.

This framework should be capable of being run on any platform that can support a python environment and has sufficient computing power.
The only recommended and/or supported platform is Windows 10 at a level consistent with modern business laptops.

This package has a key dependency of mesa_local, which must be installed prior to the installation of RET. It is a local fork of the mesa package with a number of changes to allow the functionality desired of RET.

Installation instructions
-------------------------

Install `mesa_local`. Note that the RET development team are making minor modifications to MESA alongside changes to RET, and these may not be included in the public version of MESA, but may be fundamental for RET functionality. Therefore, as part of the RET install, the local version of MESA should also be installed. This is achieved by running the following command from your local checkout of the `ret` repository:

```bash
pip install ./mesa_local
```

Install `ret` in editable mode by running the following command from your local checkout of the `ret` repository:

```bash
pip install -e .
```
Development extras are available with specifier [dev]

Create desktop shortcuts by running the following commands:

```bash
create_retplay_shortcut
```
Running tests
-------------
To run the tests with coverage reports, simply run:
```bash
pytest --cov --cov-report=html
```

To run the tests without coverage reports, simply run:
```bash
pytest
```

Getting started/Usage instructions
---------------
Once installed, ret provides the building blocks to produce an agent-based model for operational scenarios and vignettes.

Example models are provided in the examples folder.

ret provides basic objects and methods for creating operational models, however it is expected that bespoke components may need to be created to provide specific functionality required by a model. Several such items exists in the examples and codebase which extend and adapt existing components to provide new functionality, it is recommended that these examples are used to provide a design pattern for any new bespoke components created.

Generating Documentation
------------------------

Documentation is automatically produced from Python doc-strings using Sphinx. To create documentation, run the following commands from the `ret` folder:

```bash
sphinx-apidoc -o sphinx ret/ret
```

This will create `.rst` files for each module in ret. If the sphinx folder already contains `.rst` files for the project and you wish to override them, use the `-f` flag.

```bash
sphinx-build -b html sphinx sphinx-out
```

This will create an HTML version of the documentation in the `sphinx-out` folder.

Caveats
-------

This framework is there to make model development easier, as such it is down to the modeller to validate whatever they model.

Release notes
-------------
Version 2.0.0
+ Tracked Projectile and Guided Projectile weapons added
+ Platform Locating Sensor added
+ Position and culture of agents now logged
+ Results now can now be saved periodically during batch run
+ All data logs now include model step
+ All data logs are now comma delimited
+ Model seed is now settable and logged
+ The weapon that fired a shot is now logged
+ Simulation time and force composition now displayed in RetPlay
+ Current task within compound task now highlighted in RetPlay
+ Ammunition capacity and consumption now modelled and logged
+ Progress bar enabled for multiprocess batch run
+ Usability improvements
+ Batchrunner saves to timestamped directory

Version history
---------------

### 15/03/2024 Release Version 2.0.0

+ This is the second release version.
+ This repo works with the amended mesa_local repo, version 1.0.0
+ This is intended for use by experienced analysts with software and simulation development experience.
+ This version was written by Frazer-Nash Consultancy (https://www.fnc.co.uk/)

### 30/03/2022 Release Version 1.0.0

+ This is the initial release version.
+ This repo works with the amended mesa_local repo, version 1.0.0
+ This is intended for use by experienced analysts with software and simulation development experience.
+ This version was written by Frazer-Nash Consultancy (https://www.fnc.co.uk/)
