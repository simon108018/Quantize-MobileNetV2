import os, sys
import tensorflow as tf
from getdata import get_data
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import pylab
import time
import tensorflow_datasets as tfds
# import logging
# logging.getLogger("tensorflow").setLevel(logging.DEBUG)
# assert float(tf.__version__[:3]) >= 2.3

# tflite檔案路徑
tflite_models_dir = pathlib.Path("cifar10/try/models")
path = {'a': tflite_models_dir/"fp16_MobileNetV2.tflite",
        'b': tflite_models_dir/"fp16_q_MobileNetV2.tflite",
        'c': tflite_models_dir/"int8_MobileNetV2.tflite",
        'd': tflite_models_dir/"int8_q_MobileNetV2.tflite"}
type = {'a': "fp16",
        'b': "fp16-q",
        'c': "int8",
        'd': "int8-q"}

# 選擇測試檔案

ask = input('請選擇想測試的檔案。'
            '\n\'a\':\'fp16_MobileNetV2\''
            '\n\'b\':\'fp16_q_MobileNetV2\''
            '\n\'c\':\'int8_MobileNetV2\''
            '\n\'d\':\'int8_q_MobileNetV2\''
            '目前只有\'a\'可行')

filepath = path[ask]
datatype = type[ask]
# 取得資料
input_shape = (224, 224)
data = get_data(dataname='cifar10', batch_size=None, reshape=input_shape)
data.run()
train_data = data.train_data
valid_data = data.valid_data
test_data = data.test_data

# 對test_data轉成numpy，並取出images與labels
test_data_np = tfds.as_numpy(test_data)
test_images = []
test_labels = []
for data in test_data_np:
    test_images.append(data[0])
    test_labels.append(np.argmax(data[1])) # 轉成數字

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
    global test_images
    global test_labels

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    Time_start = time.time()
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_labels[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        Time_test = time.time()
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        Time_test = time.time() - Time_test

        predictions[i] = output.argmax()
        if len(test_image_indices)==1:
            print(Time_test)
        else:
            # if (i + 1) % 500 == 0:
            Time_mid = time.time()
            t = Time_mid - Time_start
            h = int(t / 3600)
            m = int(t / 60) % 60
            s = int(t % 60)
            a = round(100 * (i + 1) / len(test_image_indices), 2)
            b = "■" * int(a//5)
            c = "  "*(20-int(a//5))
            print("\r[{b}{c}]已評估{a:.2f}%,d{h} h {m} m {s} s".format(a=a, b=b, c=c, h=h, m=m, s=s), end="")
    return predictions

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
    global test_images
    global test_labels

    # test_image_indices = range(test_images.shape[0])
    test_image_indices = range(len(test_images))
    predictions = run_tflite_model(tflite_file, test_image_indices)
    accuracy = (np.sum(test_labels==predictions) * 100) / len(test_images)

    print('\n%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
    global test_labels

    predictions = run_tflite_model(tflite_file, [test_image_index])
    plt.figure()
    plt.imshow(test_images[test_image_index])
    template = model_type + " Model \n True:{true}, Predicted:{predict}"
    _ = plt.title(template.format(true=str(test_labels[test_image_index]), predict=str(predictions[0])))
    plt.grid(False)
    plt.show()


# Change this to test a different image
test_image_index = input('請輸入0-9999中的一個數字測試圖片，輸入其他值則跳出迴圈。')
while True:
    if test_image_index in np.arange(10000).astype('str'):

        test_model(filepath, int(test_image_index), model_type=datatype)
        test_image_index = input('請輸入0-9999中的一個數字測試圖片，輸入其他值則跳出迴圈。')
    else:
        break
ask = input('是否要評估整個模型，需要花費一點時間。(\'y\'or\'n\')')
if ask == 'y':
    print('正在評估模型')
    tStart = time.time()
    evaluate_model(filepath, model_type="Float")
    tEnd = time.time()
    t = tEnd-tStart
    print("It cost {h} hour {m} min {s} sec".format(h=int(t/3600),m=int((t%3600)/60), s=int(t%60)))