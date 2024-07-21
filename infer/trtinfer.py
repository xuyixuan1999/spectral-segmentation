import ctypes
import numpy as np
import tensorrt as trt
from typing import Optional, List
from cuda import cuda, cudart



def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))
        
# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[dict], outputs: List[dict], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem["host_mem"].free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        
        
class TrtInfer:
    def __init__(self, engine_file, batch_size=1):
        self.engine_file = engine_file
        self.batch_size = batch_size
        
        self.engine = None
        self.context = None
        
        self.stream = None
        self.inputs = None
        self.outputs = None
        self.bindings = None

        self.logger = trt.Logger(trt.Logger.ERROR)        
        # deserialize tensorrt engine
        with open(self.engine_file, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        assert self.engine, "Engine should not be None"
        
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine, self.batch_size)
        
        self.context = self.engine.create_execution_context()
        assert self.context, "Context should not be None"
        
        for inp in self.inputs:
            self.context.set_input_shape(inp["name"], inp["shape"])
        
    def _allocate_buffers(self, engine: trt.ICudaEngine, batch_size=1):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        
        for name in tensor_names:
            shape = engine.get_tensor_shape(name)
            if shape[0] == -1:
                shape[0] = batch_size
            # 检查shape中的每一个元素是否大于等于0
            shape_valid = np.all([s >= 0 for s in shape[1:]])
            if not shape_valid:
                raise ValueError(f"Binding {name} has dynamic shape, " +\
                "but no profile was specified.")
            size = trt.volume(shape) 
            if engine.has_implicit_batch_dimension:
                size *= engine.max_batch_size
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            
            # 分配主机和设备内存
            bindingMem = HostDeviceMem(size, dtype)
            bindings.append(bindingMem.device)
            # 将主机和设备内存添加到列表中
            binding = {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "host_mem": bindingMem,
            }
            # 将主机内存添加到输入或输出列表中
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            # if engine.binding_is_input(name):
                inputs.append(binding)
            else:
                outputs.append(binding)
        return inputs, outputs, bindings, stream
    
    def mem_cpy_host_to_device(self, inputs):
        # assert len(inputs) == len(self.inputs), "Input number mismatch"
        if inputs[0].shape[0] != self.batch_size:
            self.batch_size = inputs[0].shape[0]
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine, self.batch_size)
            for inp in self.inputs:
                self.context.set_input_shape(inp["name"], inp["shape"])            
            
        for i, inp in enumerate(inputs):
            memcpy_host_to_device(self.inputs[i]["host_mem"].device, np.ascontiguousarray(inp))
    
    def mem_cpy_device_to_host(self):
        outputs = [np.zeros(out["shape"], out["dtype"]) for out in self.outputs]
        for i, out in enumerate(outputs):
            memcpy_device_to_host(np.ascontiguousarray(out), self.outputs[i]["host_mem"].device)
        return outputs
    
    def infer(self):
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        cudart.cudaStreamSynchronize(self.stream)
    
    def forward(self, inputs):
        '''
        inputs: list of numpy array
        outputs: list of numpy array
        '''
        self.mem_cpy_host_to_device(inputs)
        self.infer()
        return self.mem_cpy_device_to_host()

if __name__ == "__main__":
    import time
    import cv2
    import sys
    from PIL import Image
    sys.path.append('/root/spectral-segmentation')
    from utils.utils import resize_mat
    batch_size = 1
    test_times = 1000
    
    infer = TrtInfer(engine_file='/root/spectral-segmentation/infer/engine/unet_25band.trt', batch_size=batch_size)
    # inputs1 = np.random.randn(4, 25, 416, 416).astype(np.float32)
    img_path = '/root/spectral-segmentation/datasets/2021-10-11_75m_45_1_018.png'
    old_img = Image.open('/root/spectral-segmentation/datasets/spectral-dataset-multi/JPEGImages/2021-4-23_75m_45_1_008.jpg')
    org_w, org_h = old_img.size
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_data = np.zeros((org_h, org_w, 25), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            img_data[:, :, i*5+j] = img[i::5, j::5]
    
    # img = np.array(old_img, dtype=np.uint8)
    img, nw, nh = resize_mat(img_data, (416, 416))

    img = np.array(img, dtype=np.float32) / 255.0
    img_data = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    img_data = np.repeat(img_data, batch_size, axis=0)
    # inputs2 = np.random.randn(4, 3, 416, 416).astype(np.float32)
    infer.mem_cpy_host_to_device([img_data])
    # warm up
    for i in range(10):
        infer.infer()
    
    
    # t1 = time.time()
    # for i in range(test_times):
    #     infer.infer()
    # t2 = time.time()
    output = infer.mem_cpy_device_to_host()[0][0]
    
    # save to the txt
    # with open('output/output.txt', 'w') as f:
    #     output1 = output.reshape(-1)
    #     for i in range(len(output1)):
    #         f.write('%.6f ' % output1[i] + '\n')
    
    output = output[int((416 - nh) // 2) : int((416 - nh) // 2 + nh), \
                    int((416 - nw) // 2) : int((416 - nw) // 2 + nw)]
    
    output = cv2.resize(output, (org_w, org_h), interpolation = cv2.INTER_LINEAR)
    
    output = output.argmax(axis=-1)
    
    colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
               (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
               (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
               (128, 64, 12)]
    
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(output, [-1])], [org_h, org_w, -1])
    
    image   = Image.fromarray(np.uint8(seg_img))
    #------------------------------------------------#
    #   将新图与原图及进行混合
    #------------------------------------------------#
    # image   = Image.blend(old_img, image, 0.7)
    # image   = Image.fromarray(np.uint8(seg_img))
    image.save('output/test.jpg')
    # save in txt 
    # with open('output.txt', 'w') as f:
    #     output = output[0].reshape(-1)
    #     for i in range(len(output)):
    #         f.write(str(output[i]) + ' ')
    # np.savetxt('output.txt', output[0].reshape(-1)[:1000], fmt='%.6f', delimiter='\n')
    # print("Infer time of %d iterations: %.3f ms" % (test_times, (t2 - t1) * 1000 / test_times))
    # print(output[0].shape)
    
        