pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openmim
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv-full==1.6.2
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmdet==2.28.2
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git -b 1.0
cd mmdetection3d
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
cd ..
pip uninstall -y opencv-python 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==4.5.1.48
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "opencv-python-headless<4.3"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt 
python setup.py develop
pip install numba==0.53.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
