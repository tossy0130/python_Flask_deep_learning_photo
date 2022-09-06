from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

# トレーニング用の画像を格納する変数X_train, 正解ラベルY_train
# テスト画像を格納する変数X_test, 正解ラベルY_testを初期化します。

X_train = []
X_test = []
Y_train = []
Y_test = []

### 警告を表示しなくなる asarray 
# エラーメッセージ ::: 'dtype=object' when creating the ndarray, arr = np.asanyarray(arr)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

### 画像ファイルを順に読み込んでリストに格納し、最終的にNumPy配列に変換
# if文の中のelseのブロックでは画像を-20°から20°まで回転、および反転させたデータをリストに追加

# =============== テストデータ　回転　、　反転 ===============

for index, classlabel in enumerate(classes) :
    
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 200: break # 200個以上なら次のラベルへ
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        
        if i < num_testdata : # テストデータの個数以下の場合
            X_test.append(data)
            Y_test.append(index)
        else: #テストデータの個数以上の場合、増幅して学習データに追加
            for angle in range(-20, 21, 5):
                # ========= 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # ========= 反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)
                
X_train = np.array(X_train, dtype=object)
X_test = np.array(X_test, dtype=object)
y_train = np.array(Y_train, dtype=object)
y_test = np.array(Y_test, dtype=object)

# ==================== ファイルに出力
xy = (X_train, X_test, y_train, y_test)
np.save("./animal_aug.npy", xy)