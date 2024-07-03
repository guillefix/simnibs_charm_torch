#!/bin/bash
#
# from https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
#

#echo "deb http://deb.debian.org/debian sid main" | sudo tee /etc/apt/sources.list.d/sid.list

sudo apt update
sudo apt-get upgrade -y
sudo apt-get install -y linux-headers-generic

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-12 g++-12

sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

sudo apt-get install -y make

# Stop the Google Cloud Ops Agent
sudo systemctl stop google-cloud-ops-agent

# Move up one directory
cd ..

# Notify the user that the system will reboot after installation of NVIDIA drivers
#echo "NOTE: System will reboot after installation of NVIDIA drivers"

# Pause for 3 seconds
#sleep 3


#Installing the driver version 550
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

#Installing the toolkit version 12.2
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit
sudo sh cuda_12.4.0_550.54.14_linux.run --silent --driver

# Download the CUDA installer
#curl -L https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.0.0/cuda_installer.pyz --output cuda_installer.pyz

# Run the installer
#sudo python3 cuda_installer.pyz install_cuda

#sudo rm /etc/apt/sources.list
#sudo touch /etc/apt/sources.list

echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
sudo ln -sf /usr/bin/gcc-12 /usr/bin/cc
source ~/.bashrc



