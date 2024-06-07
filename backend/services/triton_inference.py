import tritonclient.http as httpclient
from tritonclient.utils import *
import numpy as np


def triton_inference(
        image_data: np.ndarray,
        model_name: str = "vit_b_onnx",
        async_execution: bool = False,
        service_name: str = 'spacewalker-triton',
        port: int = 8000
    ):
    """Infer results using a running triton instance"""
    with httpclient.InferenceServerClient(f"{service_name}:{port}") as client:
        input0_data = image_data
        # input0_data = np.random.rand(3, 1024, 1024).astype(np.float32)
        inputs = [
            httpclient.InferInput(
                "input", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data)

        outputs = [
            httpclient.InferRequestedOutput("output"),
        ]
        
        if async_execution:
            # async
            response = client.async_infer(model_name, inputs, request_id=str(1), outputs=outputs).get_result()
        else:
            # sync
            response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        output0_data = response.as_numpy("output")

        return output0_data