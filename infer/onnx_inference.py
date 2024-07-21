from typing import Any
import onnxruntime
import onnx
import numpy as np

class OnnxInference:
    def __call__(self, onnx_file, providers=['CPUexcutionProvider']) -> Any:
        self.model = onnxruntime.InferenceSession(onnx_file, providers=providers)
        print("onnx model loaded")
    
    def __call__(self, input_array) -> Any:
        try:
            if len(input_array.shape) == 3:
                array = input_array[np.newaxis, :, :, :]
            else:
                array = input_array.copy()
            output = self.model.run(None, {self.model.get_inputs()[0].name: array})
        
        except Exception as e:
            print(e)
            return None