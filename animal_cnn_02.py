from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

### 変更
# import keras
from tensorflow import keras


import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def main():
    
    # NumPy形式の配列データを読み込む
    X_train, X_test, y_train, y_test = np.load("./animal.npy", allow_pickle=True)
    
    # 255で割って0~1の値に正規化する
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    
    # to_categorical関数でone-hotベクトル形式に変換する
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    # トレーニングの実行
    model = model_train(X_train, y_train)
    
    # モデルの精度評価（未使用データの推定精度を計算する）
    model_eval(model, X_test, y_test)
    
def model_train(X, y) :
    # ニューラルネットワークを新規に生成する
    model = Sequential()
    
    # 2つの畳込み層を追加する。プーリング処理でデータ削減、 ドロップアウト処理を加える
    model.add(Conv2D(32, (3,3), padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 2つの畳込み層を追加する。プーリング処理でデータ削減、 ドロップアウト処理を加える 
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # データを直列に並べて全結合層を追加、 最終的に3つのノードにデータを出力
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    # 最適化アルゴリズムの指定（adam、 rmsprop、 sgdなどがある）
    #opt = keras.optimizers.adam()
    opt = keras.optimizers.Adam()
    
    # 損失関数の宣言（カテゴリカルクロスエントロピー）
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # トレーニングの実行
    model.fit(X, y, batch_size=32, epochs=100)
    
    # モデルの保存
    model.save('./animal_cnn.h5')
    
    return model

def model_eval(model, X, y) :
    scores = model.evaluate(X, y, verbose=1) # 評価関数に投入
    print('Test Loss: ', scores[0]) # スコアの先頭のデータを取得・表示
    print('Test Accuracy:' , scores[1]) # スコアの2番めのデータを取得・表示
    
if __name__ == "__main__" :
    main()