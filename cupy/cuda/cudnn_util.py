from collections import defaultdict
import cupy
import cupy.cuda.cudnn as libcudnn
import sys
from cupy.cuda import Device

memory_pool = cupy.get_default_memory_pool()
_required_workspace_sizes = defaultdict(lambda : [cupy.cudnn.get_max_workspace_size(), 0])
_log_enabled = False

def log(msg):
    if _log_enabled:
        print(msg, file=sys.stderr, flush=True)

def require_workspace_size(device, size):
    log('require_workspace_size')
    log(' Device {}, size {} bytes'.format(device, size))
    #print('require_workspace_size Device {}, size {} bytes'.format(device, size))
    sizes = _required_workspace_sizes[device]
    if size not in sizes:
        sizes.append(size)
        _required_workspace_sizes[device] = sorted(sizes, reverse=True)
    log('require_workspace_size end')

def allocate_workspace(devices, ratio=0.9):
    log("allocate_workspace")
    #print("allocate_workspace")
    for device in devices:
        with Device(device):
            workspace = get_workspace(device)
            free, _ = cupy.cuda.runtime.memGetInfo()
            free = free + (0 if workspace is None else workspace.size)
            free = int(ratio * free)
            for req_size in _required_workspace_sizes[device]:
                log(('  Device {}, free {} B, ratio {}, req_size {} B, '
                     'workspace {} B').format(
                         device, free, ratio, req_size,
                         'None' if workspace is None else workspace.size))
                #print(('  Device {}, free {} B, ratio {}, req_size {} B, '
                #       'workspace {} B').format(
                #         device, free, ratio, req_size,
                #         'None' if workspace is None else workspace.size))
                if workspace is not None and workspace.size >= req_size:
                    break
                if req_size < free:
                    pool = memory_pool.pool(device)
                    if workspace is not None:
                        pool.free_workspace()
                    pool.malloc_workspace(req_size)
                    break
    log('allocated_workspace end')

def get_workspace(device):
    log('get_workspace')
    workspace = memory_pool.pool(device).workspace()
    log('  Device {}, workspace {} bytes'.format(
        device, 'None' if workspace is None else workspace.size))
    log('get_workspace end')
    return workspace

def free_workspace(device):
    log('free_workspace at Device {}'.format(device))
    memory_pool.pool(device).free_workspace()
    _required_workspace_sizes[device] = [cupy.cudnn.get_max_workspace_size(), 0]
    log('free_workspace end')

_cudnn_version = libcudnn.getVersion()

# The number of algorithms is that in version 5.0
_count_fwd_algo = 8
_count_bwd_filter_algo = 5
_count_bwd_data_algo = 6

# Estimate workspace size

def _get_max_workspace_size(f, count_algo):
    s = 0
    for algo in range(count_algo):
        try:
            s = max(f(algo), s)
        except cupy.cuda.cudnn.CuDNNError:
            pass
    return s

def _fwd_max_workspace_size(handle, x_desc, filter_desc,
                            conv_desc, y_desc):
    log('getConvolutionForwardWorkspaceSize')
    #print('getConvolutionForwardWorkspaceSize')
    return _get_max_workspace_size(
        lambda algo: libcudnn.getConvolutionForwardWorkspaceSize(
            handle, x_desc, filter_desc,
            conv_desc, y_desc, algo),
        _count_fwd_algo)

def _bwd_filter_max_workspace_size(handle, x_desc, gy_desc,
                                   conv_desc, filter_desc):
    log('getConvolutionBackwardFilterWorkspaceSize')
    #print('getConvolutionBackwardFilterWorkspaceSize')
    get_size = libcudnn.getConvolutionBackwardFilterWorkspaceSize
    return _get_max_workspace_size(
        lambda algo: get_size(
            handle, x_desc, gy_desc,
            conv_desc, filter_desc, algo),
        _count_bwd_filter_algo)

def _bwd_data_max_workspace_size(handle, filter_desc, gy_desc,
                                 conv_desc, x_desc):
    log('getConvolutionBackwardDataWorkspaceSize')
    #print('getConvolutionBackwardDataWorkspaceSize')
    get_size = libcudnn.getConvolutionBackwardDataWorkspaceSize
    return _get_max_workspace_size(
        lambda algo: get_size(
            handle, filter_desc, gy_desc,
            conv_desc, x_desc, algo),
        _count_bwd_filter_algo)

# Get algorithms
_fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT

def _get_fwd_algo(handle, x_desc, x, filter_desc, W, conv_desc,
                  y_desc, y, workspace):
    log('getConvolutionForwardAlgorithm')
    return libcudnn.getConvolutionForwardAlgorithm(
        handle, x_desc, filter_desc, conv_desc, y_desc, _fwd_pref,
        workspace.size)

if _cudnn_version >= 4000:
    _bwd_filter_pref = \
      libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT

    def _get_bwd_filter_algo(handle, x_desc, gy_desc, conv_desc,
                             filter_desc, workspace):
        log('getConvolutionBackwardFilterAlgorithm')
        return libcudnn.getConvolutionBackwardFilterAlgorithm(
            handle, x_desc, gy_desc,
            conv_desc, filter_desc,
            _bwd_filter_pref, workspace.size)

    _bwd_data_pref = \
      libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT

    def _get_bwd_data_algo(handle, filter_desc, y_desc,
                           conv_desc, x_desc, x, workspace):
        log('getConvolutionBackwardDataAlgorithm')
        return libcudnn.getConvolutionBackwardDataAlgorithm(
            handle, filter_desc, y_desc,
            conv_desc, x_desc, _bwd_data_pref,
            workspace.size)

else:

    def _get_bwd_filter_algo(handle, x_desc, gy_desc, conv_desc,
                             filter_desc, workspace):
        log('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')
        return libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1

    def _get_bwd_data_algo(handle, filter_desc, y_desc,
                           conv_desc, x_desc, x, workspace):
        log('CUDNN_CONVOLUTION_BWD_DATA_ALGO_1')
        return libcudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1


# Find algorithms
if _cudnn_version >= 5000:
    def _find_fwd_algo(handle, x_desc, x, filter_desc, W, conv_desc,
                       y_desc, y, workspace):
        log('findConvolutionForwardAlgorithmEx')
        #cuda.Stream.null.synchronize()
        perfs = libcudnn.findConvolutionForwardAlgorithmEx(
            handle, x_desc, x.data.ptr, filter_desc, W.data.ptr, conv_desc,
            y_desc, y.data.ptr, 1, workspace.ptr, workspace.size)
        #print('findConvolutionForwardAlgorithmEx: algo={}'.format(perfs[0]['algo']))
        return perfs[0]['algo']

    def _find_bwd_filter_algo(handle, x_desc, x, gy_desc, gy, conv_desc,
                              filter_desc, gW, workspace):
        log('findConvolutionBackwardFilterAlgorithmEx')
        #cuda.Stream.null.synchronize()
        perfs = libcudnn.findConvolutionBackwardFilterAlgorithmEx(
            handle, x_desc, x.data.ptr,
            gy_desc, gy.data.ptr,
            conv_desc,
            filter_desc, gW.data.ptr,
            1, workspace.ptr, workspace.size)
        return perfs[0]['algo']

    def _find_bwd_data_algo(handle, filter_desc, y_desc,
                            conv_desc, x_desc, x, workspace):
        log('findConvolutionBackwardDataAlgorithmEx')
        #cuda.Stream.null.synchronize()
        perfs = libcudnn.findConvolutionBackwardDataAlgorithmEx(
            handle, filter_desc, gW.data.ptr,
            y_desc, gy.data.ptr,
            conv_desc,
            x_desc, x.data.ptr,
            1, workspace.ptr, workspace.size)
        return perfs[0]['algo']
else:
    def _find_fwd_algo(handle, x_desc, x, filter_desc, W, conv_desc,
                       y_desc, y, workspace):
        return _get_fwd_algo(handle, x_desc, filter_desc, conv_desc,
                             y_desc, workspace)

    def _find_bwd_filter_algo(handle, x_desc, x, gy_desc, gy, conv_desc,
                              filter_desc, gW, workspace):
        return _get_bwd_filter_algo(handle, x_desc, gy_desc, conv_desc,
                                    filter_desc, workspace)

    def _find_bwd_data_algo(handle, filter_desc, y_desc,
                            conv_desc, x_desc, x, workspace):
        return _get_bwd_data_algo(handle, filter_desc, y_desc,
                                  conv_desc, x_desc, x, workspace)

class ConvAlgo:
    def __init__(self):
        self.algo = None
        self.ws_size = -1
        self.max_ws_size = None

class Algo(object):
    def __init__(self):
        self.device = cupy.cuda.device.get_device_id()

    def workspace(self):
        workspace = get_workspace(self.device)
        if workspace is None:
            workspace_size = cupy.cudnn.get_max_workspace_size()
            workspace = cupy.empty((workspace_size,), dtype='b')
            workspace = cupy.cuda.memory.Workspace(workspace.data.mem)
        return workspace

class FindAlgo(Algo):
    def __init__(self):
        super(FindAlgo, self).__init__()
        self._fwd = defaultdict(ConvAlgo)
        self._bwd_filter = defaultdict(ConvAlgo)
        self._bwd_data = defaultdict(ConvAlgo)

    def forward(self, x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, workspace):
        fwd = self._fwd[x.shape]
        #print('Forward algo: called Forward fwd fwd.algo={}, fwd.ws_size={}, workspace.size={}'.format(fwd.algo, fwd.ws_size, workspace.size))
        if fwd.algo is None or workspace.size != fwd.ws_size:
            fwd.ws_size = workspace.size
            if fwd.max_ws_size is None:
                fwd.max_ws_size = _fwd_max_workspace_size(
                    handle, x_desc, filter_desc, conv_desc, y_desc)
                require_workspace_size(self.device, fwd.max_ws_size)
            algo, workspace_size = cupy.cudnn._get_algorithm_fwd(
                x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, workspace.size, workspace.ptr)
            fwd.algo = algo
            log('Forward algo: {}'.format(algo))
        #print('Forward algo: return Forward fwd.algo={}, workspace.size={}'.format(fwd.algo, workspace.size))
        return fwd.algo

    def backward_filter(self, x, gy, gW, conv_param, handle, x_desc, gy_desc,
                        conv_desc, filter_desc, workspace):
        bwd = self._bwd_filter[x.shape]
        if bwd.algo is None or workspace.size != bwd.ws_size:
            bwd.ws_size = workspace.size
            if bwd.max_ws_size is None:
                bwd.max_ws_size = _bwd_filter_max_workspace_size(
                    handle, x_desc, gy_desc, conv_desc, filter_desc)
                require_workspace_size(self.device, bwd.max_ws_size)
            algo, size = cupy.cudnn._get_algorithm_bwd_filter(
                x, gy, gW, conv_param, handle, x_desc, gy_desc, conv_desc,
                filter_desc, workspace.size, workspace.ptr)
            bwd.algo = algo
            log('Backward filter algo: {}'.format(algo))
        return bwd.algo

    def backward_data(self, W, x, y, conv_param, handle, filter_desc, x_desc,
                      conv_desc, y_desc, workspace):
        bwd = self._bwd_data[x.shape]
        if bwd.algo is None or workspace.size != bwd.ws_size:
            bwd.ws_size = workspace.size
            if bwd.max_ws_size is None:
                bwd.max_ws_size = _bwd_data_max_workspace_size(
                    handle, filter_desc, y_desc, conv_desc, x_desc)
                require_workspace_size(self.device,
                                            bwd.max_ws_size)
            algo, size = cupy.cudnn._get_algorithm_bwd_data(
                W, x, y, conv_param, handle, filter_desc, x_desc,
                conv_desc, y_desc, workspace.size, workspace.ptr)
            bwd.algo = algo
            log('Backward data algo: {}'.format(algo))
        return bwd.algo


class GetAlgo(Algo):
    def __init__(self):
        super(GetAlgo, self).__init__()

    def forward(self, x, W, y, conv_param, handle, x_desc, filter_desc,
                conv_desc, y_desc, workspace):
        algo, workspace_size = cupy.cudnn._get_algorithm_fwd(
            x, W, y, conv_param, handle, x_desc, filter_desc,
            conv_desc, y_desc, workspace.size, workspace.ptr)
        return algo;

    def backward_filter(self, x, gy, gW, conv_param, handle, x_desc, gy_desc,
                        conv_desc, filter_desc, workspace):
        algo, size = cupy.cudnn._get_algorithm_bwd_filter(
            x, gy, gW, conv_param, handle, x_desc, gy_desc, conv_desc,
            filter_desc, workspace.size, workspace.ptr)
        return algo

    def backward_data(self, W, x, y, conv_param, handle, filter_desc, x_desc,
                      conv_desc, y_desc, workspace):
        algo, size = cupy.cudnn._get_algorithm_bwd_data(
            W, x, y, conv_param, handle, filter_desc, x_desc,
            conv_desc, y_desc, workspace.size, workspace.ptr)
        return algo
