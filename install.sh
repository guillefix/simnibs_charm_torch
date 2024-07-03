#!/bin/bash

# FIRST INSTALL https://github.com/Nudge-github/nudge_simulation_system
# Check if the 'nse' package is installed

source ~/.bashrc

sudo apt install -y software-properties-common

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10
sudo apt install -y python3.10-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.10 get-pip.py

sudo apt-get install -y freeglut3 freeglut3-dev libglew-dev
sudo apt-get install -y mesa-utils

sudo apt install -y libtbb2 libtbb-dev libwebp7 libwebp-dev


PIP="python3 -m pip install"
$PIP install PyQt5==5.9.2
$PIP install mkl

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

# Update package lists and install system dependencies
sudo apt-get clean
sudo apt-get update
sudo apt-get install -y -f
sudo dpkg --configure -a
sudo apt-get install -y git build-essential wget vim tmux
sudo apt-get install -y libglib2.0-0
#sudo apt-get install -y libglib2.0-0t64
#sudo apt-get install -y libboost-all-dev
sudo apt-get remove --purge -y libboost-all-dev
sudo apt-get autoremove -y
sudo apt install -y libboost-all-dev=1.74.0.3ubuntu7
sudo apt install -y libboost1.74-dev
sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dev
sudo apt-get install -y libgmp-dev

# Install requirements
$PIP install --user -r requirements.txt

# Manually add the repository and key
#echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list
echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe" | sudo tee -a /etc/apt/sources.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 871920D1991BC93C

# Update package list
sudo apt update

# Install gcc-7 and g++-7
sudo apt install -y gcc-7 g++-7

# Set GCC 7 as the default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70

sudo apt-get remove --purge -y libboost-all-dev
sudo apt install -y libboost-all-dev=1.74.0.3ubuntu7
sudo apt install -y libboost1.74-dev
source ~/.bashrc

# Set environment variables for GCC
export CC=gcc-7
export CXX=g++-7

# Clone simnibs and install it
python3 setup.py clean
rm -r build/
python3 setup.py install --user

# Install simnibs in editable mode
#$PIP install --editable .
#$CONDA init

#export PATH=${CONDA_DIR}"/bin:$PATH"
export LD_LIBRARY_PATH=$(pwd)/simnibs/external/lib/linux/:$LD_LIBRARY_PATH
#echo 'export PATH=${CONDA_DIR}"/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH='$(pwd)'/simnibs/external/lib/linux/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Link external programs for simnibs
python3 simnibs/cli/link_external_progs.py
