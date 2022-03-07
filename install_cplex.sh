#!/bin/bash

find_cplex_file () {
    CPLEX_FILES="$(ls cplex_studio*.linux-x86-64.bin)"

    max_ver=0
    for file in ${CPLEX_FILES[@]};
    do
        version="$(echo $file | grep -E -oh '[0-9]{3}')"
        if [ "$version" -lt 129 ] && [ "$version" -gt "$max_ver" ]; then
            max_ver=$version
        fi
    done
    best_cplex="cplex_studio${max_ver}.linux-x86-64.bin"
}

# this script is only executed after it is confirmed
# that a cplex_studio file exists in this directory
# so after calling find_cplex_file, best_cplex and max_ver
# are guaranteed to exist. 
find_cplex_file

if [ -x "$best_cplex" ]; then
    sed -i "s|USER_INSTALL_DIR=/opt/cplex.*|USER_INSTALL_DIR=/opt/cplex${max_ver}|" cplex.properties
    # install cplex and setup coregs to find it
    # if install fails, do not make a link and do not replace glpk
    ./$best_cplex -f cplex.properties && \
        ln -s /opt/cplex${VER} /opt/cplex && \
        sed -i 's/glpk/cplex/g' coregs.py
else
    echo "Cannot install cplex because cplex_studio${VER}.linux-x86-64.bin is not executable"
   exit 1
fi

echo "export CPLEX_HOME=/opt/cplex${VER}/cplex" >> ~/.bashrc
echo 'export PATH="${PATH}:${CPLEX_HOME}/bin/x86-64_linux"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CPLEX_HOME}/lib/x86-64_linux"' >> ~/.bashrc
