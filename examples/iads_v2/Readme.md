## Release Notes

### Version

This version is 20220330.

### 30/03/2022 Release Version 20220330

This is an example for demonstration purposes. The source code should not be used as the basis for any decisions.
It is solely intended as an initial showcase of some of the features.
As it was initially implemented with an older version of the code, many newer features are not present.
This repo works with the amended mesa repo, bundled with this example, version 1.0.0
This repo works with the ret repo, version 1.0.0
This is intended for use by experienced analysts.
This is not a formal contractual deliverable (at this stage), but was developed under contract PO410000130089.
Unit icons created using: <https://spatialillusions.com/>

# IADS

A representation of an Integrated Air Defence System (IADS).
It consists of a strike package taking off from Coningsby, flying to Amsterdam where there is an IADS.

This model demonstrates several features of the RET package.
It is hoped that this may be used in future as an example from which to make other scenarios for investigation.

## Installing Mesa and Ret

1. Create a new python environment
2. Activate that environment
3. Install the mesa package, inside your checkout of the mesa package run the following command:

   ```bash
   pip install .
   ```

4. Install the RET package, inside your checkout of the RET package run the following command:

   ```bash
   pip install .
   ```

5. Install the IADs scenario model. From inside this folder, run:

```bash
pip install .
```

## Running the Model

To launch the model run the following command:

```bash
python run.py
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) (if not done for you) and press Start.
The speed of the simulation may be adjusted with the slider.

### Running the Batch Runner

To run the batch-runner, run the following command:

```bash
python run_batch.py
```

### Running the AI Gym

To run the model using AI Gym, run the following command:

```bash
python run_gym.py
```

## Key Files

- [IADS/model.py](IADS/model.py): Core model file; contains the IADS class.
- [IADS/server.py](IADS/server.py): Sets up the visualization.
- [IADS/portrayal.py](IADS/portrayal.py): Exposes the agent parameters for display by the visualization.
- [images/Base_map.png](images/Background.png): The background image file.
- [run.py](run.py): Launches the visualization.
