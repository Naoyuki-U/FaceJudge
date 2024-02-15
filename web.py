import tensorflow as tf
import multiprocessing as mp

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
# from werkzeug import secure_filename
import os
import eval
import glob

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
app.config['DEBUG'] = True
# 投稿画像の保存先
UPLOAD_FOLDER = './static/images/default'

# ルーティング。/にアクセス時
@app.route('/')
def index():
  return render_template('index.html', result="", detect_img="", count=0)

# 画像投稿時のアクション
@app.route('/post', methods=['GET','POST'])
def post():
    if request.method == 'POST':
        if not request.files.get('file') == u'':
            # 前回アップロードファイル削除
            for p in glob.glob(f'{UPLOAD_FOLDER}/*', recursive=True):
                if os.path.isfile(p):
                    os.remove(p)
            # アップロードされたファイルを保存
            f = request.files.get('file')
            img_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(img_path)
            # eval.pyへアップロードされた画像を渡す
            result,detect_img,count = eval.evaluation(img_path, './cnn_test_model.h5')
        else:
            result = []
            detect_img = ""
            count = 0
        # return render_template('result.html', result=result)
        return render_template('index.html', result=result, detect_img=detect_img, count=count)
    else:
        # エラーなどでリダイレクトしたい場合
        return redirect(url_for('index'))

if __name__ == '__main__':
  app.debug = True
  app.run(host='localhost')