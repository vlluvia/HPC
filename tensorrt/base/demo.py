

import os

import numpy as np
import tensorrt as trt
from cuda import cudart

trtFile = './model.plan'
data =  np.arange(3*4*5, dtype=np.float32).reshape(3, 4, 5)

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.exists(trtFile):
        with open(trtFile, "rb") as f:
            engineString = f.read()

        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")

    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
        profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
        config.add_optimization_profile(profile)

        identityLayer = network.add_identity(inputTensor)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:                                          # write the serialized netwok into a .plan file
            f.write(engineString)
            print("Succeeded saving .plan file!")


    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
    return
    print("Succeeded building engine!")

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_model(i) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.creat_execution_context()
    context.set_input_shape(lTensorName[0], [3, 4, 5])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(context.get_tensor_dtype(lTensorName[i]))))

    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])


    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)
