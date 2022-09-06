from flickrapi import FlickrAPI #FlikcrAPIクライアントのクラス定義
from urllib.request import urlretrieve #コマンドラインからhttp通信をする関数
import os, time, sys #Pythonからシステムにアクセスする関数, タイマー関数

### API
key = "001826a2c20c0ba54cda4b1b7e99982b"
secret = "c9970ced0cce5872"
wait_time = 1 # リクエストを発行するインターバル

animalname = sys.argv[1] # コマンドランで2番目の引数を取得
savedir = "./" + animalname # 引数で与えた単語名で、フォルダ作成

# FlickrAPIのクライアントオブジェクトを生成し, search関数で検索を実行
flickr = FlickrAPI(key, secret, format='parsed-json')

result = flickr.photos.search(
    text = animalname, # 検索ワード
    per_page = 400, # 検索数上限
    media = 'photos', # データタイプ
    sort = 'relevance', # 結果表示順, 最新から
    safe_search = 1, # 子供に不適切な画像を除外
    extras = 'url_q, licence' # オプション。データURL, ライセンスタイプ
)

# 検索結果　取得   
photos = result['photos']

######
# 画像ファイルのurl_q（画像のダウンロードURL）を取得し、
# コマンドラインでURLを叩いてファイルを取得するurlretrieveを呼び出して、
# 1秒おきにダウンロードを実行
######
for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q'] # photoオブジェクトからダウンロードURLを取得
    filepath = savedir + '/' + photo['id'] + '.jpg' #ファイル名をフルパスで生成
    if os.path.exists(filepath) : continue # ファイルが存在すれば、次へ
    urlretrieve(url_q, filepath) # ファイルダウンロードを実行
    time.sleep(wait_time) #サーバー負荷を考慮して1秒あける




