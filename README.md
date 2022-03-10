## A variation of the OpenAI Frozen Lake environment.

# Setup
You can create a conda environment using the provided yml file as follow:

`conda env create -f environment.yml`

# Usage
Please make sure to activate the conda environment first.

`conda activate frozenlake`


Once the conda environment is ready, you can run the model as follow:

`python main.py`

To retrain, please change line 98

from `LOAD_MODEL_FROM_FILE = True` to `LOAD_MODEL_FROM_FILE = False`.

Here is an example output

![map_13.png](map_13.png?raw=true "Example Output")