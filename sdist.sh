
export NCCL_ROOT="$HOME/usr/local"
export CUDNN_ROOT="$HOME/use/cudnn/cuda/targets/ppc64le-linux"
export CPATH="$NCCL_ROOT/include:$CUDNN_ROOT/include:$CPATH"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$CUDNN_ROOT/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$NCCL_ROOT/lib:$CUDNN_ROOT/lib:$LIBRARY_PATH"


/bin/rm -f dist/*.tar.gz
python setup.py sdist 2>&1 | tee sdist.log
ls -l dist
