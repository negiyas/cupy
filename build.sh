
export NCCL_ROOT="$HOME/usr/local"
export CUDNN_ROOT="$HOME/use/cudnn/cuda/targets/ppc64le-linux"
export CPATH="$NCCL_ROOT/include:$CUDNN_ROOT/include:$CPATH"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$CUDNN_ROOT/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$NCCL_ROOT/lib:$CUDNN_ROOT/lib:$LIBRARY_PATH"


python setup.py install 2>&1 | tee build.log
pip install -e . 2>&1 | tee -a build.log
