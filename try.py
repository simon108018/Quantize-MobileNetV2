import tensorflow as tf
from getdata import get_data
import tensorflow_model_optimization as tfmot
import pathlib
import numpy as np

# gpu設定
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

quantize_model = tfmot.quantization.keras.quantize_model
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply
quantize_scope = tfmot.quantization.keras.quantize_scope

# 取得資料
classes = 10
input_shape = (224, 224)
data = get_data(dataname='cifar10', batch_size=32, reshape=input_shape)
data.run()
train_data = data.train_data
valid_data = data.valid_data
data = get_data(dataname='cifar10', batch_size=None, reshape=input_shape)
data.run()
test_data = data.test_data
print('已取得 train_data、valid_data、test_data')




def conv_block(x, filters, kernel=(1, 1), stride=(1, 1)):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    return x


def depthwise_res_block(x, filters, kernel, stride, t, resdiual=False):

    input_tensor = x
    exp_channels = x.shape[-1]*t  #扩展维度

    x = conv_block(x, exp_channels, (1,1), (1,1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, padding='same', strides=stride)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    x = tf.keras.layers.Conv2D(filters, (1,1), padding='same', strides=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if resdiual:
        x = tf.keras.layers.add([x, input_tensor])

    return x


def inverted_residual_layers(x, filters, stride, t, n):
    x = depthwise_res_block(x, filters, (3, 3), stride, t, False)

    for i in range(1, n):
        x = depthwise_res_block(x, filters, (3, 3), (1, 1), t, True)

    return x


def MovblieNetV2(classes):

    img_input = tf.keras.layers.Input(shape=(224, 224, 3))

    x = conv_block(img_input, 32, (3,3), (2,2))

    x = tf.keras.layers.DepthwiseConv2D((3,3), padding='same', strides=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    x = inverted_residual_layers(x, 16, (1, 1), 1, 1)
    x = inverted_residual_layers(x, 24, (2, 2), 6, 1)
    x = inverted_residual_layers(x, 32, (2, 2), 6, 3)
    x = inverted_residual_layers(x, 64, (2, 2), 6, 4)
    x = inverted_residual_layers(x, 96, (1, 1), 6, 3)
    x = inverted_residual_layers(x, 160, (2, 2), 6, 3)
    x = inverted_residual_layers(x, 320, (1, 1), 6, 1)

    x = conv_block(x, 1280, (1, 1), (2, 2))

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Conv2D(classes, (1, 1), padding='same')(x)
    x = tf.keras.layers.Reshape((classes,))(x)
    x = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(img_input, x)

    return model


# model = MovblieNetV2(10)


img_input = tf.keras.layers.Input(shape=(224, 224, 3))
x = conv_block(img_input, 32, (3, 3), (2, 2))
x = tf.keras.layers.Reshape((1, 1, 112*112*32))(x)
x = tf.keras.layers.Conv2D(classes, (1, 1), padding='same')(x)
x = tf.keras.layers.Reshape((classes,))(x)
x = tf.keras.layers.Activation('softmax')(x)
model = tf.keras.Model(img_input, x)

model.summary()

q_model = quantize_model(model)
q_model.summary()


# 將model轉換float16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_float16_model = converter.convert()
print('已成功將model轉換\'tflite_float16_model\'')
# 儲存tflite model

tflite_models_dir = pathlib.Path("cifar10/try/models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"fp16_MobileNetV2.tflite"
tflite_model_file.write_bytes(tflite_float16_model)
print('成功生成\'fp16_MobileNetV2.tflite\'')

# 將q_model轉換float16
converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.target_spec.supported_types = [tf.float16]
# 以下放入的data只是為了提供Converter測量轉換的範圍
def representative_dataset_gen():
    for input in test_data.take(100):
        yield [input[0]]
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
q_tflite_float16_model = converter.convert()
print('已成功將model轉換\'tflite_float16_q_model\'')
# 儲存tflite model

tflite_models_dir = pathlib.Path("cifar10/try/models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"fp16_q_MobileNetV2.tflite"
tflite_model_file.write_bytes(q_tflite_float16_model)
print('成功生成\'fp16_q_MobileNetV2.tflite\'')



# 將model轉換int8

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 以下放入的data只是為了提供Converter測量轉換的範圍
def representative_dataset_gen():
    for input in test_data.batch(1).take(100):
        yield [input[0]]
converter.representative_dataset = representative_dataset_gen
# Ensure that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.int8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int_model = converter.convert()
print('成功轉換\'tflite_int_model\'')

tflite_models_dir = pathlib.Path("cifar10/try/models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"int8_MobileNetV2.tflite"
tflite_model_file.write_bytes(tflite_int_model)
print('成功生成\'int8_MobileNetV2.tflite\'')


# 將qmodel轉換int8

converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 以下放入的data只是為了提供Converter測量轉換的範圍
def representative_dataset_gen():
    for input in test_data.batch(1).take(100):
        yield [input[0]]
converter.representative_dataset = representative_dataset_gen
# q_tflite_model = converter.convert()
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.int8]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
q_tflite_int_model = converter.convert()
print('成功轉換\'tflite_int_q_model\'')

tflite_models_dir = pathlib.Path("cifar10/try/models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"int8_q_MobileNetV2.tflite"
tflite_model_file.write_bytes(q_tflite_int_model)
print('成功生成\'int8_q_MobileNetV2.tflite\'')