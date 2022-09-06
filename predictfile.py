import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

### 画像認識用
from keras.models import Sequential, load_model
import keras,sys
import numpy as np
from PIL import Image

### ターゲット　クラスラベルの初期化
classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

# ブラウザからアップロードした、データの保存用ディレクトリ,画像ファイルの拡張子の指定
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif']) #対応する拡張子一覧を生成

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER #アプリの設定に追加

### ファイル名をチェックする関数
def allowed_file(filename):
    # ファイル名がOKなら、拡張子を小文字にしてパスを返す 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST']) #ルートURLの定義
def upload_file(): #ルートが呼ばれた時に呼ばれる関数
    if request.method == 'POST': #データがブラウザからPOSTで送信された場合
        if 'file' not in request.files: #POSTリクエストにファイルが無い場合
            flash('ファイルがありません')
            return redirect(request.url) #元のページに戻す
        file = request.files['file'] #ファイルをリクエストから取得
        if file.filename == '': #ファイルが空なら
            flash('ファイルがありません')
            return redirect(request.url) #元のページに戻す
        if file and allowed_file(file.filename): #ファイルがあって拡張子がOKの場合
            filename = secure_filename(file.filename) #ファイル名をサニタイズ
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #ファイルをアップロードフォルダーに保存
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) #ファイルパスを生成
            model = load_model('./animal_cnn_aug.h5') #学習済みモデルをファイルから読み込む

            image = Image.open(filepath) 
            image = image.convert('RGB') #RGB変換
            image = image.resize((image_size, image_size)) #サイズを揃える
            data = np.asarray(image) / 255 #正規化
            X = [] 
            X.append(data) #XにNumPy配列にしたdataを追加
            X = np.array(X) #NumPyアレーに型変換

            result = model.predict([X])[0] #予測値の1つ目のデータを取得
            predicted = result.argmax() #ハイスコアの添字を取得
            percentage = int(result[predicted] * 100) #推定確率をパーセント形式に
            
            label = ""
            if classes[predicted] == 'monkey':
                label = "サル"
            elif classes[predicted] == 'boar':
                label = "イノシシ"
            else:
                label = "カラス"

            return "ラベル： " + label + ", 確率："+ str(percentage) + " %" #結果を出力
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>画像認識 TEST</title></head>
    <body>
    <h1>画像認識 テスト（サル, イノシシ, カラス）の画像を判定</h1>

    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''