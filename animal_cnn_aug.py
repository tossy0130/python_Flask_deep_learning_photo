from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

# import keras
from tensorflow import keras

import numpy as np

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

def main():
    X_train, X_test, y_train, y_test = np.load("./animal_aug.npy",  allow_pickle=True)
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)
    
def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

 #   opt = keras.optimizers.adam()
    opt = keras.optimizers.Adam()
 

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, batch_size=64, epochs=1)

    # モデルの保存
    model.save('./animal_cnn_aug.h5')

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])
    
def model_predict(model, X, y):
    result = model.predict(X)
    for i in range(X.shape[0]):
        print('predict: ', result[i].argmax())
        print('correct: ', y[i].argmax())
if __name__ == "__main__":
    main()