import keras
import glob
import os
import numpy as np
from keras.utils import to_categorical
import cv2

class Class_Generator(keras.utils.Sequence):
    def __init__(self, src_path, input_shape, class_num, batch_size, is_train=False):
        folder_list = os.listdir(src_path)
        self.is_train = is_train
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.class_num = class_num
        self.x, self.y = [], []

        for i, folder in enumerate(folder_list):
            imgs = glob.glob(src_path+folder+"/*.JPEG")
            self.x += imgs
            self.y += list(np.full((len(imgs)), fill_value=i))

        self.on_epoch_end()

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        batch_index = self.index[idx*self.batch_size:(idx+1)*self.batch_size]
        for i in batch_index:
            batch_x.append(self.x[i])
            batch_y.append(self.y[i])
        out_x, out_y = self.data_gen(batch_x, batch_y)
        return out_x, out_y

    def __len__(self):
        return round(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        self.index = np.arange(len(self.x))
        if self.is_train:
            np.random.shuffle(self.index)

    def data_gen(self, x, y):
        input_x = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        # input_y = np.zeros((self.batch_size, self.class_num), dtype=np.float32)

        input_y = to_categorical(y, num_classes=self.class_num)

        for idx in range(len(x)):
            img = cv2.imread(x[idx])
            input_x[idx] = cv2.resize(img, (self.input_shape[0], self.input_shape[1])) / 255.0
            # input_y[idx] = one_hot_y[idx]

        return input_x, input_y

if __name__ == "__main__":
    src_path = "./datasets/train/"
    cg = Class_Generator(src_path)
