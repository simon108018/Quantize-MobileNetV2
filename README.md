# Quantize-MobileNetV2

### 0. 可先將h5、tflite放在cifar10/models裡
### 1.執行train.py，可訓練模型。(裡面使用tensorflow.keras.applications.mobilenet_v2來訓練模型，可略過不訓練)
### 2.接著執行quantize.py，將model與q_model轉換成float16與int8
### 3.利用predict.py 評估模型。

## 問題:
### 1.在converter的時候，float16是可以成功轉換成tflite，int8不行。
#### RuntimeError: tensorflow/lite/kernels/pad.cc:123 op_context.dims <= reference_ops::PadKernelMaxDimensionCount() was not true.Node number 0 (PAD) failed to prepare.
### 2.在執行predict.py時，fp16_MobileNetV2成功評估，但fp16_q_MobileNetV2出現以下error，int8型態的因為轉tflite就已經不成功了，無法測試。
####  File "C:/Users/user/Desktop/python/ML/Quantize-MobileNetV2/predict.py", line 60, in run_tflite_model
####    interpreter.allocate_tensors()
####  File "C:\Users\user\.conda\envs\tf2_3_gpu\lib\site-packages\tensorflow\lite\python\interpreter.py", line 243, in allocate_tensors
####    return self._interpreter.AllocateTensors()
####  RuntimeError: tensorflow/lite/kernels/quantize.cc:111 affine_quantization->scale->size == 1 was not true.Node number 1 (QUANTIZE) failed to prepare.
