import keras
import os
import datetime
import imgaug.augmenters as iaa
from generator import Class_Generator
from models.resnet import Resnet
from models.vgg16 import VGGNET16
from models.densenet import Densenet
from Utils.cyclical_learning_rate import CyclicLR
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint


def train():
    '''model configuration'''
    input_shape = (64, 64, 3)
    class_num = 200
    epochs = 1000
    batch_size = 16
    weight_decay = 1e-4
    lr = 1e-4
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = keras.metrics.CategoricalAccuracy()
    model_name = "densenet"

    '''Augmentation'''
    augs = [
        iaa.Fliplr(0.5),

        iaa.SomeOf((1,2),[
            iaa.MultiplyAndAddToBrightness(),
            iaa.GammaContrast()
        ]),

        iaa.SomeOf((0,1), [
            iaa.Sometimes(0.7, iaa.AdditiveGaussianNoise()),
            iaa.Sometimes(0.7, iaa.GaussianBlur())
        ]),

        iaa.SomeOf((0,8),[
            iaa.ShearX(),
            iaa.ShearY(),
            iaa.ScaleX(),
            iaa.ScaleY(),
            iaa.TranslateX(),
            iaa.TranslateY(),
            iaa.Sometimes(0.3, iaa.Affine()),
            iaa.Sometimes(0.3, iaa.PerspectiveTransform()),
        ]),

        iaa.SomeOf((0,1),[
            iaa.Sometimes(0.6, iaa.Dropout()),
            iaa.Sometimes(0.6, iaa.CoarseDropout()),
            iaa.Sometimes(0.6, iaa.Cutout())
        ])
    ]

    '''call back'''
    log_dir = "./logs/%s/%s"%(model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    weight_save_dir = "./save_weights/"
    if not os.path.isdir(weight_save_dir):
        os.mkdir(weight_save_dir)
    weight_save_file = "%s/%s_{epoch:05d}.h5"%(weight_save_dir, model_name)

    '''get datasets'''
    train_gen = Class_Generator("./datasets/train/", input_shape, class_num, batch_size, augs=augs, is_train=True)
    valid_gen = Class_Generator("./datasets/val/", input_shape, class_num, batch_size, augs= [], is_train=False)

    step_size = train_gen.__len__()
    print("Step size: ", step_size)

    '''train'''
    dn = Densenet(input_shape=input_shape, class_num=class_num, layer_num=121, weight_decay=weight_decay)
    model = dn.densenet()
    model.summary()
    model.load_weights("./save_weights/densenet_00039.h5")
    model.save("./test.h5")
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=epochs,
                        max_queue_size=20, workers=4, initial_epoch=39,
                        callbacks=[TensorBoard(log_dir),
                                   # ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
                                   CyclicLR(base_lr=1e-5, max_lr=1e-3, step_size=step_size*8, mode="triangular2"),
                                   ModelCheckpoint(weight_save_file, monitor="val_loss", save_best_only=True)])

if __name__ == "__main__":
    train()