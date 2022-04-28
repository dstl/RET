System overview
===============

The RET ecosystem extends tooling developed under the MESA package, and introduces new graphical tools for preparing and reviewing agent-based models. The illustration below shows the core components of RET, where items in red are sourced from the MESA package, and items in blue represent future extensions that are not currently included.

.. image:: system-overview.jpg

Each of the components is described in detail below.

User interfaces
---------------

**RET Gen** is the RET Generation user interface. This Graphical User Interface (GUI) allows users to create, view and update RET model definition files, and has three main configuration elements:

* The scenario-independent data, which will typically be features such as the definition of agents and sensors.
* The scenario-dependent data, including aspects such as terrain and the agents selected from the scenario-independent data.
* Study-specific parameters, which is the set of parameters that the user wishes to vary within their analysis (i.e., those parameters that are provided to the sampler), including the distributions of the parameter.

**RET Play** is a GUI for visualising, exploring and analysing the results of a RET model run. RET Play can visualise any RET run which was configured to produce a playback file, including individual runs from a batch-analysis.

The RET Play GUI is based on Dash. While it provides the capability to play, pause and rewind a model, this is simply changing the position in the playback file, rather than re-running RET. Therefore, the information displayed in the RET Play GUI is purely deterministic, and is fully contained in the RET Play file, should analysts wish to interrogate the file directly.

**MESA GUI** is a rudimentary JavaScript-based GUI, that provides a live visualisation of the MESA analysis. The MESA GUI is very flexible, but has a rudimentary appearance and support for user interaction.

For most purposes, the MESA GUI is superseded by the RET Play GUI. However, RET Play is based on the Playback file, rather than a live instrumentation of a MESA model, and therefore there may be some cases where it is still advantageous to use the MESA GUI to interrogate the behaviour of a model in single-run mode.

Command-line tools
---------------------

**RET** is the agent-based modelling analysis engine, which loads a Model definition file, and runs either a single analysis or a batch analysis, depending on the contents of the model definition file. RET is an extension of MESA, which provides the core agent-based modelling capability.

Data files
-------------

**Scenario independent data** is a controlled set of model data, which is automatically loaded into the RET Gen graphical user interface.

**Scenario-dependent data** is provided by users through RET Gen. Some aspects of the scenario independent data are provided by the user through interaction with on-screen widgets (e.g., numbers and types of agents), whereas other aspects are loaded from file (such as background terrain files).

**Study controls** allow the user to configure how RET will run the model, such as the number of runs, what type of sampler to user, and which parameters to vary.

**Model definition file** is the core output from RET Gen, and contains sufficient information to launch a RET instance. Therefore, the model definition file includes data drawn from scenario independent data, scenario independent data, and study specific parameters. In some instances, such as for the provision of background terrain files, the model definition file will include path references to these data, rather than the content of the files.

As RET is developed, it is expected that the model definition file format will change. Therefore, all model definition files contain a file format version number. Both RET and RET Gen can load old versions of a Model definition file and convert them to the latest format.

**Playback file** is a complete summary of a single RET run, which can be generated either from running RET in single-run mode, or instrumenting individual runs within a batch run. The RET playback file contains all the information needed for RET Play to visualise the run it represents.

**Logging output** provides a secondary source of information to analysts wishing to interrogate the results of a RET run, and is particularly useful where statistical analysis of multiple runs is required. Logging output is provided as a series of CSV files.
