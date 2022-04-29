RET
===

``mesa_ret`` is a package that extends the functionality of the ``mesa`` package to simplify and enhance the use for analysts and combat modellers.
ret stands for Rapid Exploratory modelling Toolset.

Release Notes
-------------

Version
-------

This version is 1.0.0

30/03/2022 Release Version 1.0.0
---------------------------------

| This is the first release version.
  This repo works with the amended mesa repo, version 1.0.0
| This is intended for use by experienced analysts with software and simulation development experience.
| This version was written by Frazer-Nash Consultancy.
.. image:: fnc.jpg
   :width: 200px
   :height: 100px
   :scale: 100 %
   :alt: Frazer-Nash Consultancy Logo
   :target: https://www.fnc.co.uk/
| https://www.fnc.co.uk/


Using mesa_ret
--------------

Install ``mesa``. Note that the RET development team are making minor modifications to MESA alongside changes to RET, and these may not be included in the public version of MESA, but may be fundamental for RET functionality. Therefore, as part of the RET install, the local version of MESA should also be installed. This is achieved by running the following command from your local checkout of the `ret` repository:

.. code:: bash

    pip install ./mesa

Install ``mesa_ret`` in editable mode by running the following command from your local checkout of the ``mesa_ret`` repository:

.. code:: bash

    pip install -e ./mesa_ret[dev]

Running tests
-------------

There are 3 modules to be tested. They each have their own code coverage but (some) shared tests.
To run the tests, run the following command from your checkout of the ``ret`` repository, assuming an environment that has ``mesa_ret`` and all dependencies installed:

.. code:: bash

    ./run_tests_with_coverage_report.sh
    
This will generate coverage reports in the /htmlcov/ directory.

To run the test without coverage reports, simply run:

.. code:: bash

    pytest


Code coverage without the integration tests may be obtained by adding the following flag to any of the above test commands:

.. code:: bash

    --ignore=mesa_ret/testsret/integration_tests/

Generating Documentation
------------------------

Documentation is automatically produced from Python doc-strings using Sphinx. To create documentation, run the following commands from the ``mesa_ret`` folder:

.. code:: bash

    sphinx-apidoc -o sphinx mesa_ret/mesa_ret

This will create ``.rst`` files for each module in mesa_ret. If the sphinx folder already contains ``.rst`` files for the project and you wish to override them, use the ``-f`` flag.

.. code:: bash

    sphinx-build -b html sphinx sphinx-out

This will create an HTML version of the documentation in the ``sphinx-out`` folder.

Caveats
-------

This framework is there to make model development easier, as such it is down to the modeller to validate whatever they model.

Operating platform
------------------

This framework should be capable of being run on any platform that can support a python environment and has sufficient computing power.
The only recommended and/or supported platform is Windows 10 at a level consistent with modern business laptops.
