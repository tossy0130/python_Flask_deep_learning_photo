from PIL import Image # PillowからImageクラスをインポート
import os, glob # OSにアクセスする関数をインポート
import numpy as np # NumPyをインポート
# from sklearn import model_section #トレーニングとテストデータを分割する関数

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

classes = ["monkey","boar","crow"]
num_classes = len(classes) # リストの長さ 取得
image_size = 50 # 画像データの変換サイズ。縦横ピクセル数

### 警告を表示しなくなる asarray 
# エラーメッセージ ::: 'dtype=object' when creating the ndarray, arr = np.asanyarray(arr)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

### imageに入っているリスト型データを, asarray関数で、
### NumPyの配列データ型に変換してdata変数に格納します。

X = [] #リスト型変数Xを初期化
Y = [] #リスト型変数Yを初期化

for index, classlabel in enumerate(classes) :
    
    # ディレクトリ名を生成
    photos_dir = "./" + classlabel
    
    # ファイル一覧を取得
    files = glob.glob(photos_dir + "/*.jpg")
    
    # 各ファイルをNumPy アレーに変換し、リストに追加
    for i, file in enumerate(files):
        # files から 1個づつ取り出し file に入れ　付番
        
        if i >= 200: break # 200 を超えたら次のラベルへループ
        image = Image.open(file) # Imageクラスのopen関数でファイルをオープン
        image = image.convert("RGB") # 配色データをRGBの順に揃える
        image = image.resize((image_size, image_size)) # 画像サイズを揃える
        data = np.asarray(image) # Numpy アレー に変換
        X.append(data)  #リストXの末尾に追加
        Y.append(index) #リストYの末尾に追加 

#X = np.array(X)
#Y = np.array(Y) 
X = np.array(X, dtype=object) # np.array エラー対策 ::: dtype=object を追加
Y = np.array(Y, dtype=object) # np.array エラー対策 ::: dtype=object を追加

#### 作成したXとYをトレーニング用と精度テスト用にスプリットします。
### 特に指定をしなければ、scikit learnのmodel_selection関数は3:1の割合, 
### つまりテストデータを25%としてデータセットを分割します。

# X_train, X_test, y_train, y_test = model_section.train_test_split(X, Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y)

xy = (X_train, X_test, y_train, y_test)

np.save("./animal.npy", xy)