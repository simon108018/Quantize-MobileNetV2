import tensorflow as tf
import tensorflow_datasets as tfds



#設定訓練資料
class get_data:
    def __init__(self, dataname = 'cifar10', batch_size = 32, reshape=(224,224)):
        self.train_data, self.info = tfds.load(dataname, split="train[10%:]", with_info=True)
        self.valid_data = tfds.load(dataname, split="train[:10%]")
        self.test_data = tfds.load(dataname, split="test")
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.sh = reshape[0]
        self.sw = reshape[1]
        self.batch_size = batch_size

    # 對影像進行縮放
    def parse_aug_fn(self, dataset):
        def zoom(x, sh=self.sh, sw=self.sw):
            x = tf.image.resize(x, (sh, sw))
            return x

        # 影像標準化
        x = tf.cast(dataset['image'], tf.float32) / 255.
        # 影像放大到224*224
        x = zoom(x)
        y = tf.one_hot(dataset['label'], 10)
        y = y

        return x, y
    def run(self):
        train_num = int(self.info.splits['train'].num_examples / 10) * 9
        train_data = self.train_data.shuffle(train_num)
        train_data = train_data.map(map_func=self.parse_aug_fn, num_parallel_calls=self.AUTOTUNE)
        self.train_data = train_data.prefetch(buffer_size=self.AUTOTUNE)
        if self.batch_size is not None:
            self.train_data = train_data.batch(self.batch_size)

        valid_data = self.valid_data.map(map_func=self.parse_aug_fn, num_parallel_calls=self.AUTOTUNE)
        self.valid_data = self.valid_data.prefetch(buffer_size=self.AUTOTUNE)
        if self.batch_size is not None:
            self.valid_data = valid_data.batch(self.batch_size)

        test_data = self.test_data.map(map_func=self.parse_aug_fn, num_parallel_calls=self.AUTOTUNE)
        self.test_data = test_data.prefetch(buffer_size=self.AUTOTUNE)
        if self.batch_size is not None:
            self.test_data = test_data.batch(self.batch_size)