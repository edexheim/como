# usage: source install.sh

conda create -n como python=3.10.13 -y
conda activate como

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install open3d pip install opencv-python-headless==4.8.1.78 pytorch-lightning==1.8.2 pyrealsense2
pip install PyOpenGL glfw PyGLM

git clone --recursive https://github.com/edexheim/lietorch.git
cd lietorch
python setup.py install
cd ..

pip install -e . 