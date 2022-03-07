# COREGS

[![DOI](https://zenodo.org/badge/455716754.svg)](https://zenodo.org/badge/latestdoi/455716754)

## Overview

This version of COREGS is setup for use within a Docker environment. 
A few key differences between this version and the main version are:
  - Temoa is installed in the current directory instead of a directory higher
  - The default solver is GLPK
    - Should a CPLEX installation file be placed in the current directory when the container is built, the default solver will then become CPLEX.
      - This is the advised strategy as the problems COREGS is suited for can grow quite large and GLPK may not provide a solution in a reasonable amount of time.
    - Due to the version of Pyomo required by Temoa, *CPLEX version 12.8 is the highest that can be supported by COREGS.*
    - The divergence from Gurobi here is due to the inability to use Gurobi academic licenses within containers.
  - Several new files have been added to assist with running COREGS within the container
    - `Dockerfile`: this file is used to build the container
    - `install_cplex.sh`: this file is executed when building the container if a CPLEX installer file is found in the build directory 
    - `docker_run_coregs.sh`: this file is executed within the container to run COREGS
    - `docker_coregs.py`: this file should be executed from the host machine to run COREGS in the container. The proper data directories are mounted within the container to facilitate input and output from COREGS. You should pass arguments to this script the same way you would to COREGS and you can run this with the `--help` flag to find the usage. 
    - `cplex.properties`: this file provides CPLEX installers with the information needed to install CPLEX properly
  - Additionally, the conda environment has been made more specific for the target container
  - GRAPS is compiled with `gfortran` rather than `ifort` (Intel's Fortran compiler). This is done to reduce the size of the container.

To setup and use COREGS, first download and install GRAPS, Temoa, and the data for TVA using `python coregs_init_setup.py` then follow the instructions in either [Docker with CPLEX](#docker-with-cplex) or [Docker without CPLEX](#docker-without-cplex). 
The Docker setup instructions assume you have Docker installed properly on your machine.
If you do not, you can follow the instruction at [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/) to get it. 
After setting up the docker container, you can run `python docker_coregs.py --help` to print out the usage COREGS expects. 
The `docker_coregs.py` script accepts arguments in the same manner as the `coregs.py` script and will simply pass them along.
You can find basic usage information in the [Running COREGS](#running-coregs) section.

## Docker with CPLEX

To use the COREGS container with CPLEX you must:
  1. Register for a CPLEX license (IBM offers free academic licenses for students and educators)
  1. Download a CPLEX studio version less than 12.9 for x86-64 Linux architecture
    - Regardless of the host machine, the CPLEX version should be for x86-64 Linux as this is the container it will run in
  1. Move the `cplex_studioVVV.linux-x86-64.bin` file to the root directory of this project
  1. Build the container by executing `docker build -t coregs:1.0.0 .` from the root directory of this project

## Docker without CPLEX

If you do not plan to use CPLEX, you can still build the container by executing the command in [Step 4](#docker-with-cplex).
Alternatively, you can pull the prebuilt image from Docker Hub by executing `docker pull lcford/coregs:1.0.0`.
This image is built without CPLEX and thus will use GLPK to solve Temoa.

## Running COREGS

To run COREGS in the Docker container, use the `docker_coregs.py` script.
Running `python docker_coregs.py --help` will display the usage and command line argument descriptions for COREGS.
To run COREGS with no other options, execute `python docker_coregs.py <start_year>-<start_month> <nusers> <method>`.
  - `<start_year>` should be the four digit year that your scenario begins.
  - `<start_month>` should be the two digit (zero padded) month your scenario begins.
  - `<nusers>` should be the number of users GRAPS expects to see.
  - `<method>` should be the optimization method you want to use. Accepted options are `icorps`, `mhb`, `mhp`, or `single`. 
For example, to run the Winter 2004 scenario from the original analysis with ICORPS, the command would be `python docker_coregs.py 2004-12 29 icorps`.

## Model Output

Model scenarios are tagged with unique identifiers depending on the start date and method used.
The first letters of the three months that make up the seasonal scenario are the first part of the scenario ID (e.g., `DJF` for December, January, February), followed by the four digit year (e.g., 2004). 
The last part of the ID is the optimization method used (e.g., `icorps`) and these are all combined using underscores (e.g., `DJF_2004_icorps`).
This ID will be used to identify all output from GRAPS and Temoa.

There are three main locations model output will be stored: reservoir operation data from GRAPS can be found under `graps_output/<scenario_id>`, generation data from Temoa can be found under `generation_output/<scenario_id>.csv`, and all electricity system information can be found by querying the `data/tva_temoa.sqlite` database.
Objective information is stored under `objective_output/<scenario_id>.csv` where the first column is the iteration number and the second column is the cost to meet power demand for that scenario.
