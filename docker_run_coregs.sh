#!/bin/bash

source /usr/local/bin/activate coregs-env

# check args
if [ $# -eq 0 ]; then
    # if nothing is provided, provide help
    args="-h"
    check_cplex=false
else
    args=$@
    check_cplex=true
fi

if [ $check_cplex = true ]; then
    # if cplex is in it, it was explicitly specified
    # if it is not in it and a solver is not specified, 
    # check the default solver in coregs
    if grep -q "cplex" <<< $* || \
        (! grep -q "solver" <<< $* && grep -q "cplex" coregs.py); then
        if ! [ -d "/opt/cplex" ]; then
            printf "Cannot find cplex installation.\n"
            printf "Ensure that a cplex studio installer file of version <=12.8\n"
            printf "is present when building docker container.\n"
            printf "e.g. cplex_studio128.linux-x86-64.bin\n"
            exit 1
        fi
    fi
fi

export CPLEX_HOME="/opt/cplex/cplex"
export PATH="${PATH}:${CPLEX_HOME}/bin/x86-64_linux"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CPLEX_HOME}/lib/x86-64_linux"

python coregs.py $args
