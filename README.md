# COREGS

## Compile GRAPS

To compile GRAPS, you will need the intel oneAPI HPC toolkit, which is dependent on the oneAPI base toolkit. 
You can find information on installing these toolkits at the [Intel oneAPI webpage](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.pvef6v). 
You will also need the [GNU Make Utility](https://www.gnu.org/software/make/).
This will allow you to easily compile GRAPS on a Linux machine. If you are running Mac or Windows, it will be easiest to use the docker image to run this code [ADD DOCKER IMAGE].

After you have installed the toolkits and sourced the setup script (e.g., `source /opt/intel/oneapi/setvars.sh` if you installed the toolkits under the `/opt` directory), you can compile GRAPS by changing to the `graps` directory and running `make` from the command line. 
This will compile a shared library and store it at `graps/lib/graps.so` and an executable and store it at `graps/bin/graps`. 
COREGS relies on the shared library and does not use the executable but it is generated for your convenience.

## Setup python environment

It is highly recommended to use virtual environments for your python projects.
Two common approaches to this are using `venv` with `pip` or using `conda`.

If you are using `venv` and `pip`, ensure you are using Python 3.7 to create the environment.

### `venv` environment setup

To create the virtual environment for COREGS, execute the following command in
- `python3.7 -m venv coregs-env`
- `source coregs-env/bin/activate`
- `pip install -r requirements.txt`

### `conda` environment setup

If you are using the `conda` package manager, you can create the python environment for COREGS by executing `conda env create -f environment.yml` in the root directory of this project. 
To activate this environment so COREGS has access to the necessary packages, run `conda activate coregs`.

Regardless of which method you choose to setup the virtual environment, it must be activated before running the `coregs.py` file.

## Data

### Data Retrieval

The original data associated with this model can be found on Zenodo **MAKE THIS A LINK**.
Before running COREGS, this data should be downloaded and all files should be placed under the `data` directory in the project root.

### GRAPS

GRAPS will populate scenarios based on files in the `graps_input/default` directory and will modify their data using files in the `data` directory. 
The default files were created by the [GRAPS interface](https://github.com/lcford2/graps_gui).

### Temoa

Temoa relies on the SQLite database at `data/tva_temoa.sqlite`. 
The `.sql` file that created that database is also included at `data/tva_temoa.sql`.
This file can be used to create a database for your own system.
For more information regarding the Temoa database structure, refer to the [Temoa Project Documentation](https://temoacloud.com/temoaproject/Documentation.html#database-construction).

## Running COREGS

COREGS is ran with the `coregs.py` file.
Running `python coregs.py --help` will display the usage and command line argument descriptions for COREGS.
To run COREGS with no other options, execute `python coregs.py <start_year>-<start_month> <nusers> <method>`.
`<start_year>` should be the four digit year that your scenario begins.
`<start_month>` should be the two digit (zero padded) month your scenario begins.
`<nusers>` should be the number of users GRAPS expects to see.
`<method>` should be the optimization method you want to use. Accepted options are `icorps`, `mhb`, `mhp`, or `single`. 
For example, to run the Winter 2004 scenario from the original analysis with ICORPS, the command would be `python coregs.py 2004-12 29 icorps`.

## Model Output

Model scenarios are tagged with unique identifiers depending on the start date and method used.
The first letters of the three months that make up the seasonal scenario are the first part of the scenario ID (e.g., `DJF` for December, January, February), followed by the four digit year (e.g., 2004). 
The last part of the ID is the optimization method used (e.g., `icorps`) and these are all combined using underscores (e.g., `DJF_2004_icorps`).
This ID will be used to identify all output from GRAPS and Temoa.

There are three main locations model output will be stored: reservoir operation data from GRAPS can be found under `graps_output/<scenario_id>`, generation data from Temoa can be found under `generation_output/<scenario_id>.csv`, and all electricity system information can be found by querying the `data/tva_temoa.sqlite` database.
Objective information is stored under `objective_output/<scenario_id>.csv` where the first column is the iteration number and the second column is the cost to meet power demand for that scenario.
