import keras
import os
import datetime
from generator import Class_Generator
from models.resnet import Resnet
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint


def train():
    '''model configuration'''
    input_shape = (64, 64, 3)
    class_num = 200
    epochs = 1000
    batch_size = 64
    lr = 1e-4
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = keras.metrics.CategoricalAccuracy()
    model_name = "resnet"

    '''call back'''
    log_dir = "./logs/%s/%s"%(model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    weight_save_dir = "./save_weights/"
    if not os.path.isdir(weight_save_dir):
        os.mkdir(weight_save_dir)
    weight_save_file = "%s/%s_{epoch:05d}.h5"%(weight_save_dir, model_name)

    '''get datasets'''
    train_gen = Class_Generator("./datasets/train/", input_shape, class_num, batch_size, is_train=True)
    valid_gen = Class_Generator("./datasets/test/", input_shape, class_num, batch_size, is_train=False)

    '''train'''
    rs = Resnet(input_shape=input_shape, class_num=class_num, layer_num=34)
    model = rs.resnet()
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=epochs,
                        max_queue_size=20, workers=4,
                        callbacks=[TensorBoard(log_dir),
                         ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
                         ModelCheckpoint(weight_save_file, monitor="val_loss", save_best_only=True)])

if __name__ == "__main__":
    train()