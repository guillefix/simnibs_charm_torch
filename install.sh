#!/bin/bash

# FIRST INSTALL https://github.com/Nudge-github/nudge_simulation_system
# Check if the 'nse' package is installed

source ~/.bashrc

'''
# can we get rid of pip maybe?
# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found, installing Miniconda..."
    # Install Miniconda
    CONDA_DIR=~/miniconda3
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
    rm ~/miniconda.sh
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc
    echo "conda activate base" >> ~/.bashrc

    # Initialize Conda and activate the base environment
    $CONDA_DIR/bin/conda init
    echo "conda activate base" >> ~/.bashrc
    source ~/.bashrc

    # Update the base Conda environment to Python 3.10 and install pip
    $CONDA_DIR/bin/conda install -y python=3.10 pip
else
    echo "Conda is already installed."
    CONDA_DIR=$(dirname $(dirname $(which conda)))
    source ~/.bashrc
fi


'''

sudo apt install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
sudo apt install python3.10-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.10 get-pip.py

sudo apt-get install -y freeglut3 freeglut3-dev libglew-dev
sudo apt-get install -y mesa-utils

sudo apt install -y libtbb2 libtbb-dev libwebp7 libwebp-dev



PIP="python3 -m pip install"
$PIP install PyQt5==5.9.2
$PIP install mkl

source ~/.bashrc


package=$($PIP list | grep -w nse)

# If the package is not installed, run your commands
if [ -z "$package" ]; then
	cd ..
	git clone git@github.com:Nudge-github/nudge_simulation_system.git
	cd nudge_simulation_system
	./install.sh
	#$PIP install -e .
	cd ..
	cd nudge_studies_toolkit
else
	echo "The package 'nse' is already installed."
fi

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

#sudo rm -rf /usr/local/cuda*

#$CONDA_DIR/bin/conda install -y nvidia/label/cuda-12.2.0::cuda
#$CONDA_DIR/bin/conda install -c nvidia/label/cuda-12.2.0 -y cuda-toolkit

# Install some dependencies for simnibs
#$CONDA_DIR/bin/conda install -c conda-forge -y freeglut=3.0.0 tbb=2021.5.0 libwebp=1.2.2 pyqt=5.9.2 mkl=2022.0.1 gmp=6.3.0

# Set the path to the Conda-installed pip
#CONDA=$CONDA_DIR/bin/conda

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
#sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7
#sudo update-alternatives --set gcc /usr/bin/gcc-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70

#sudo update-alternatives --config gcc
#sudo update-alternatives --config g++


sudo apt-get remove --purge -y libboost-all-dev

#sudo apt-get install -y libcgal-dev
sudo apt install -y libboost-all-dev=1.74.0.3ubuntu7
sudo apt install -y libboost1.74-dev

source ~/.bashrc

# Set environment variables for GCC
export CC=gcc-7
export CXX=g++-7

# Clone simnibs and install it
cd ..
git clone git@github.com:guillefix/simnibs_charm_torch.git simnibs
cd simnibs
python3 setup.py clean
rm -r build/
python3 setup.py install --user

# Install simnibs in editable mode
#$PIP install --editable .
#$CONDA init

#export PATH=${CONDA_DIR}"/bin:$PATH"
export LD_LIBRARY_PATH=$(pwd)/simnibs/external/lib/linux/:$LD_LIBRARY_PATH
#echo 'export PATH=${CONDA_DIR}"/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$(pwd)/simnibs/external/lib/linux/:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/home/guillefix/lambda/simnibs/simnibs/external/lib/linux/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc



# Link external programs for simnibs
python3 simnibs/cli/link_external_progs.py
