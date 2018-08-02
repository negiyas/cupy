import sys

from cupy.cuda import memory_hook
import inspect


class CallStackHook(memory_hook.MemoryHook):
    """Memory hook that prints call stack information.

    This memory hook identifies GPU memory consumption of each DNN layer.
    It shows the source files and lines calling Chainer
    ``malloc``/``free`` methods involved in the hooked functions
    at postprocessing/preocessing time (that is, just after/before each
    method is called).

    Example:
        The basic usage is to use it with ``with`` statement.

        Code example::

            >>> import cupy
            >>> from cupy.cuda import memory_hooks
            >>>
            >>> cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
            >>> with memory_hooks.CallStackHook():
            ...     x = cupy.array([1, 2, 3])
            ...     del x  # doctest:+SKIP

        Output example::
        CALLSTACK MALLOC 2 8388608 0x3effab600000 0x3effd466f0f0 \
                           convolution_2d.py:googlenet.py:37
        CALLSTACK MALLOC 2 102760448 0x3effa4a00000 0x3effd466feb8 \
                           relu.py:googlenet.py:38
        CALLSTACK MALLOC 2 25690112 0x3effa3000000 0x3effd712bf28 max_pooling_2d.py:googlenet.py:41
        CALLSTACK MALLOC 2 25690112 0x3effa1600000 0x3effd460e438 local_response_normalization.py:googlenet.py:42
        CALLSTACK MALLOC 2 25690112 0x3eff9fc00000 0x3effd460e710 local_response_normalization.py:googlenet.py:42
        CALLSTACK MALLOC 2 16384 0x3effe9d1f800 0x3effd460e940 convolution_2d.py:googlenet.py:46
        CALLSTACK MALLOC 2 32768 0x3effe9d23800 0x3effd460e9e8 convolution_2d.py:googlenet.py:46

        where the output format is space-separated CSV and
        The first column is ``CALLSTACK``,
        The second column is ``MALLOC`` or ``FREE`,
	The third column is the CUDA Device ID,
        The forth column is the memory size (rounded),
        The fifth column is the memory pointer,
        The sixth column is the cupy.cuda.memory.PooledMemory object ID, and
        The seventh column is layer information consisting of file name of layer
            function, its caller's source file and line number, and
        The eighth column is source files and lines of all callers (optional)

    Attributes:
        file: Output file_like object that redirect to.
        flush: If ``True``, this hook forcibly flushes the text stream
            at the end of print. The default is ``False``.
        full: If ``True``, the eighth column is shown, otherwise not shown.
            The default is ``False``.

    """

    name = 'CallStackHook'

    def __init__(self, file=sys.stdout, flush=False, full=False):
        self.file = file
        self.flush = flush
        self.full = full

    def _print(self, msg):
        self.file.write(msg)
        self.file.write('\n')
        if self.flush:
            self.file.flush()

    def _stack(self):
        stack = inspect.stack()
        ret = "UNKNOWN"
        for i in range(len(stack) - 2, 1, -1):
            info = inspect.getframeinfo(stack[i][0])
            pinfo = inspect.getframeinfo(stack[i - 1][0])
            if ((info.function == "__call__") or \
                ((pinfo.function == "to_gpu") and \
                 (pinfo.filename.split("/")[-1] == "link.py"))):
                ret = pinfo.function + ":" + pinfo.filename.split("/")[-1] + \
                      ":" + info.filename.split("/")[-1] + ":" + str(info.lineno)
                break 
        if self.full:
            ret += " "
            for i in range(2, len(stack) - 1):
                info = inspect.getframeinfo(stack[i][0])
                ret += info.function + ":" + info.filename.split("/")[-1] + \
                       ":" + str(info.lineno) + ","
        return ret

    def malloc_postprocess(self, **kwargs):
        msg = 'CALLSTACK MALLOC %d %d %s %s %s'
        msg %= (kwargs['device_id'], kwargs['mem_size'],
                hex(kwargs['mem_ptr']), hex(kwargs['pmem_id']), self._stack())
        self._print(msg)

    def free_preprocess(self, **kwargs):
        msg = 'CALLSTACK FREE %d %d %s %s %s'
        msg %= (kwargs['device_id'], kwargs['mem_size'],
                hex(kwargs['mem_ptr']), hex(kwargs['pmem_id']), self._stack())
        self._print(msg)
