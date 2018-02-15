#!/bin/bash
set -e

# Modified from bootstrap-conda.sh script, see:
# https://bitbucket.org/bombora-datascience/bootstrap-conda

# Some variables to cut down on space. Modify these if you want.
URL_PREFIX=https://repo.continuum.io/archive/
ANACONDA_VARIANT=3
ANACONDA_VERSION=5.0.1
OS_ARCH=Linux-x86_64
ANACONDA_FULL_NAME="Anaconda${ANACONDA_VARIANT}-${ANACONDA_VERSION}-${OS_ARCH}.sh"
CONDA_INSTALL_PATH="/opt/conda"
PROJ_DIR=${PWD}
LOCAL_CONDA_PATH=${PROJ_DIR}/anaconda
ANACONDA_SCRIPT_PATH="${PROJ_DIR}/${ANACONDA_FULL_NAME}"


## Install Anaconda
if [[ -f "/etc/profile.d/conda.sh" ]]; then
    echo "file /etc/profile.d/conda.sh exists! Dataproc has installed conda previously. Skipping install!"
    command -v conda >/dev/null && echo "conda command detected in $PATH"
else
    # Download Anaconda installer
    echo "Defined Anaconda script path: ${ANACONDA_SCRIPT_PATH}"
    if [[ -f "${ANACONDA_SCRIPT_PATH}" ]]; then
      echo "Found existing Anaconda script at: ${ANACONDA_SCRIPT_PATH}"
    else
      echo "Downloading Anaconda script to: ${ANACONDA_SCRIPT_PATH} ..."
      wget ${URL_PREFIX}${ANACONDA_FULL_NAME} -P "${PROJ_DIR}"
      echo "Downloaded ${ANACONDA_FULL_NAME}!"
      ls -al ${ANACONDA_SCRIPT_PATH}
      chmod 755 ${ANACONDA_SCRIPT_PATH}
    fi

    # Install conda
    if [[ ! -d ${LOCAL_CONDA_PATH} ]]; then
        echo "Installing ${ANACONDA_FULL_NAME} to ${CONDA_INSTALL_PATH}..."
        rm -rf "${LOCAL_CONDA_PATH}"
        bash ${ANACONDA_SCRIPT_PATH} -b -p ${CONDA_INSTALL_PATH} -f
        chmod 755 ${CONDA_INSTALL_PATH}
        ln -sf ${CONDA_INSTALL_PATH} "${LOCAL_CONDA_PATH}"
        chmod 755 "${LOCAL_CONDA_PATH}"
    else
        echo "Existing directory at path: ${LOCAL_CONDA_PATH}, skipping install!"
    fi
fi


## Update PATH and configure conda
echo "Setting environment variables..."
CONDA_BIN_PATH="${CONDA_INSTALL_PATH}/bin"
export PATH="${CONDA_BIN_PATH}:${PATH}"
echo "Updated PATH: ${PATH}"
echo "And also HOME: ${HOME}"
hash -r
which conda
conda config --set always_yes true --set changeps1 false
conda info -a # Useful printout for debugging any issues with conda


## Update global profiles to add Anaconda to the PATH
# based on: http://stackoverflow.com/questions/14637979/how-to-permanently-set-path-on-linux
# and also: http://askubuntu.com/questions/391515/changing-etc-environment-did-not-affect-my-environemtn-variables
# and this: http://askubuntu.com/questions/128413/setting-the-path-so-it-applies-to-all-users-including-root-sudo
echo "Updating global profiles to export anaconda bin location to PATH..."
if grep -ir "CONDA_BIN_PATH=${CONDA_BIN_PATH}" /etc/profile.d/conda.sh
    then
    echo "CONDA_BIN_PATH found in /etc/profile.d/conda.sh , skipping..."
else
    echo "Adding path definition to profiles..."
    echo "export CONDA_BIN_PATH=$CONDA_BIN_PATH" | tee -a /etc/profile.d/conda.sh #/etc/*bashrc /etc/profile
    echo 'export PATH=$CONDA_BIN_PATH:$PATH' | tee -a /etc/profile.d/conda.sh  #/etc/*bashrc /etc/profile

fi


## Update global profiles to add the anaconda location to PATH
echo "Updating global profiles to export anaconda bin location to PATH and set PYTHONHASHSEED ..."
if grep -ir "export PYTHONHASHSEED=0" /etc/profile.d/conda.sh
    then
    echo "export PYTHONHASHSEED=0 detected in /etc/profile.d/conda.sh , skipping..."
else
    # Fix issue with Python3 hash seed.
    # Issue here: https://issues.apache.org/jira/browse/SPARK-13330 (fixed in Spark 2.2.0 release)
    # Fix here: http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed/
    echo "Adding PYTHONHASHSEED=0 to profiles and spark-defaults.conf..."
    echo "export PYTHONHASHSEED=0" | tee -a  /etc/profile.d/conda.sh  #/etc/*bashrc  /usr/lib/spark/conf/spark-env.sh
    echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf
fi


## Ensure that Anaconda Python and PySpark play nice
# http://blog.cloudera.com/blog/2015/09/how-to-prepare-your-apache-hadoop-cluster-for-pyspark-jobs/
echo "Ensure that Anaconda Python and PySpark play nice by all pointing to same Python distro..."
if grep -ir "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python" /etc/profile.d/conda.sh
    then
    echo "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python detected in /etc/profile.d/conda.sh , skipping..."
else
    echo "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python" | tee -a  /etc/profile.d/conda.sh /etc/environment /usr/lib/spark/conf/spark-env.sh
fi


## DONE installing
echo "Finished bootstrapping via Anaconda, sourcing /etc/profile ..."
source /etc/profile


## Install and setup Python packages.
# Add commands to install and configure packages here.
conda update --all
pip install bigdl
