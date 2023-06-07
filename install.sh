# conda python 3.8.1
# conda create -n instant python=3.8.1
pip install aitviewer==1.9.0
pip install torch==1.13.1+cu116 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install fvcore iopath
pip install git+https://github.com/NVlabs/tiny-cuda-nn/@v1.6#subdirectory=bindings/torch
pip install pytorch-lightning==1.5.7
pip install opencv-python # reboot?
pip install imageio
pip install smplx==0.1.28
pip install hydra-core==1.1.2
pip install h5py ninja chumpy numpy==1.23.1
