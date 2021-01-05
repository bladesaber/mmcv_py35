# mmcv for python 3.5

since the Cambricon Company can only provide the docker image with pytorch 1.3.0 and python 3.5. I can not  
use mmcv in a normal way, so I had to rewrite the format of mmcv. God help, it seems that it can work.  

The original repository is in: https://github.com/open-mmlab/mmcv   

### Install
cd mmcv_py35  
MMCV_WITH_OPS=1 pip install -e .
