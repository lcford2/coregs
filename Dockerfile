FROM  conda/miniconda3-centos7:latest

# update and install dependencies
RUN yum update -y && \
    yum install -y gcc-gfortran make java-1.8.0-openjdk-devel

# need to mount the directory with our code
COPY . /code/coregs/

# setup cplex if it is available
WORKDIR /code/coregs
RUN if ls ./cplex*.linux-x86-64.bin 1> /dev/null 2>&1; then ./install_cplex.sh; fi

# build graps
WORKDIR /code/coregs/graps
RUN make

# create conda environment
WORKDIR /code/coregs
RUN conda env create -f environment.yml

# default run command
CMD ["./docker_run_coregs.sh"]

