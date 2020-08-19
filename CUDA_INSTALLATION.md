# CUDA & cuDNN installation

Original link: https://medium.com/@sh.tsang/tutorial-cuda-v10-2-cudnn-v7-6-5-installation-ubuntu-18-04-3d24c157473f

## Install CUDA 10.2

```bash
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

sudo sh cuda_10.2.89_440.33.01_linux.run

sudo nano ~/.bashrc

# Add the following to the end of file:
export PATH=/usr/local/cuda-10.2/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

source ~/.bashrc

# check installation:
nvcc -V
```

## Install cuDNN v7.6.5

1. Go to https://developer.nvidia.com/rdp/cudnn-archive and download
   * cuDNN Runtime Library for Ubuntu18.04 (Deb)
   * cuDNN Developer Library for Ubuntu18.04 (Deb)
2. Upload files to server OR get download url via Firefox Browser, then shorten it if needed, then download on server via 'wget' 
3. Execute the following:
```bash
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
```
