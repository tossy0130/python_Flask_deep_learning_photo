from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras,sys
import numpy as np
from PIL import Image

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

def build_model():
    # モデルのロード
    model = load_model('./animal_cnn_aug.h5')

    return model

def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)/255
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()
