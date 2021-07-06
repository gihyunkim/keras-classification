from keras.datasets import cifar100

def get_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    return x_train, y_train, x_test, y_test