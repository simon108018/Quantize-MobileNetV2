import os, sys
from tensorflow import keras
import tensorflow as tf
from getdata import get_data
from tensorflow.keras.utils import plot_model
import pathlib
import tensorflow_model_optimization as tfmot
# import warnings
# warnings.filterwarnings("ignore")

# 處理quantize functions
quantize_model = tfmot.quantization.keras.quantize_model
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply
quantize_scope = tfmot.quantization.keras.quantize_scope

# gpu設定
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 取得資料
input_shape = (224, 224)
data = get_data(dataname='cifar10', batch_size=32, reshape=input_shape)
data.run()
train_data = data.train_data
valid_data = data.valid_data
test_data = data.test_data

# 取得model
model_dir = 'cifar10/models'
if not os.path.isfile(model_dir + "/quantiled_MobileNetV2.h5"):
    print('無法在{model_dir}找到\'quantiled_MobileNetV2.h5\'。'.format(model_dir=model_dir))
    print('將讀取\'MobileNetV2.h5\'，並quantize此模型。')
    # 讀取base model
    if not os.path.isfile(model_dir + "/MobileNetV2.h5"):
        print('沒有成功找到\'MobileNetV2.h5\'，請確定已將檔案放置{model_dir}'.format(model_dir=model_dir))
        sys.exit()
    model = keras.models.load_model(model_dir + "/MobileNetV2.h5")

    # quantize_model
    q_model = quantize_model(model)
    # 優化器、損失函數，指標函數
    q_model.compile(keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()])

    # 回調函數
    log_dir = os.path.join('cifar10', 'quantiled-MobileNetV2')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')
    model_mckp = keras.callbacks.ModelCheckpoint(model_dir + "/quantiled_MobileNetV2.h5", monitor='val_categorical_accuracy', mode='max')
    plot_model(q_model, to_file='model.png')
    ask = input('是否要開始訓練此模型?(請輸入\'y\'or\'n\')')
    while True:
        if ask == 'y':
            print('因為在測試而已，只使用epochs=1。')
            q_history = q_model.fit(train_data, initial_epoch=0, epochs=1, validation_data=valid_data, callbacks=[model_cbk, model_mckp])
            break
        elif ask == 'n':
            break
        else:
            ask = input('請輸入\'y\'or\'n\'，注意大小寫。')
else:
    model = keras.models.load_model(model_dir + "/MobileNetV2.h5")
    with quantize_scope():
        model_dir = 'cifar10/models'
        q_model = tf.keras.models.load_model(model_dir + "/quantiled_MobileNetV2.h5")
        plot_model(q_model, to_file='model.png')



# 將model轉換float16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 以下放入的data只是為了提供Converter測量轉換的範圍
def representative_dataset_gen():
    for input in test_data.batch(1).take(100):
        yield [input[0]]
converter.representative_dataset = representative_dataset_gen
tflite_float16_model = converter.convert()
print('已成功將model轉換\'tflite_float16_model\'')
# 儲存tflite model

tflite_models_dir = pathlib.Path("cifar10/models")
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
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 以下放入的data只是為了提供Converter測量轉換的範圍
def representative_dataset_gen():
    for input in test_data.batch(1).take(100):
        yield [input[0]]
converter.representative_dataset = representative_dataset_gen
q_tflite_float16_model = converter.convert()
print('已成功將model轉換\'tflite_float16_q_model\'')
# 儲存tflite model

tflite_models_dir = pathlib.Path("cifar10/models")
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
# q_tflite_model = converter.convert()
# Ensure that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.int8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int_model = converter.convert()
print('成功轉換\'tflite_int_model\'')

tflite_models_dir = pathlib.Path("cifar10/models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"int8_MobileNetV2.tflite"
tflite_model_file.write_bytes(tflite_int_model)
print('成功生成\'int8_MobileNetV2.tflite\'')


# 將qmodel轉換int8

converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
# converter.target_spec.supported_ops = [tf.int8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
q_tflite_int_model = converter.convert()
print('成功轉換\'tflite_int_q_model\'')

tflite_models_dir = pathlib.Path("cifar10/models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"int8_MobileNetV2.tflite"
tflite_model_file.write_bytes(q_tflite_int_model)
print('成功生成\'int8_q_MobileNetV2.tflite\'')