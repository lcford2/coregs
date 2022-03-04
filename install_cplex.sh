#!/bin/bash

VER=128

if [ -x "./cplex_studio${VER}.linux-x86-64.bin" ]; then
    sed -i "s|USER_INSTALL_DIR=/opt/cplex.*|USER_INSTALL_DIR=/opt/cplex${VER}|" cplex.properties
    ./cplex_studio${VER}.linux-x86-64.bin -f cplex.properties
    ln -s /opt/cplex${VER} /opt/cplex
else
    echo "Cannot install cplex because cplex_studio${VER}.linux-x86-64.bin is not executable"
   exit 1
fi

echo "export CPLEX_HOME=/opt/cplex${VER}/cplex" >> ~/.bashrc
echo 'export PATH="${PATH}:${CPLEX_HOME}/bin/x86-64_linux"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CPLEX_HOME}/lib/x86-64_linux"' >> ~/.bashrc
