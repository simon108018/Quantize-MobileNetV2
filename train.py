import os, sys
import tensorflow as tf
from getdata import get_data
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


# gpu設定
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 設定路徑與建立資料夾
model_dir = 'cifar10/models'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# 取得資料
input_shape = (224, 224)
data = get_data(dataname='cifar10', batch_size=32, reshape=input_shape)
data.run()
train_data = data.train_data
valid_data = data.valid_data
test_data = data.test_data
print('已取得 train_data、valid_data、test_data')

# 取得mobil_net model
if not os.path.isfile(model_dir + "/MobileNetV2.h5"):
    print('無法在{model_dir}找到\'MobileNetV2.h5\'，請將此檔案放在{model_dir}'.format(model_dir=model_dir))
    ask = input('是否要忽略此訊息，並直接生成model?(請輸入\'y\'or\'n\')')
    while True:
        if ask == 'y':
            base_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape+(3,))
            # 利用.output解開封包
            x = base_model.output
            x = keras.layers.Dense(10, activation="softmax")(x)
            model = tf.keras.Model(inputs=base_model.input, outputs=x)
            break
        elif ask == 'n':
            sys.exit()
        else:
            ask = input('請輸入\'y\'or\'n\'，注意大小寫。')

else:
    model = keras.models.load_model(model_dir + "/MobileNetV2.h5")
    print('成功讀取\'MobileNetV2.h5\'')
plot_model(model, to_file='model.png')
print(model.summary())
# compile
model.compile(keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy()])

# 回調函數
log_dir = os.path.join('cifar10', 'MobileNetV2')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + "/MobileNetV2.h5", monitor='val_categorical_accuracy', mode='max')

# train
ask = input('是否要開始訓練此模型?(請輸入\'y\'or\'n\')')
while True:
    if ask == 'y':
        print('因為在測試而已，只使用epochs=1。')
        history = model.fit(train_data, initial_epoch=0, epochs=1, validation_data=valid_data, callbacks=[model_cbk, model_mckp])
        break
    elif ask == 'n':
        sys.exit()
    else:
        ask = input('請輸入\'y\'or\'n\'，注意大小寫。')