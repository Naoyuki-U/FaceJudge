
from flask import Flask, render_template, request, redirect, url_for

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

@app.route('/')
def sample_form1():
    return render_template('sampleform.html')


@app.route('/sampleform')
def sample_form():
    return render_template('sampleform.html')

@app.route('/sampleform-post', methods=['POST'])
def sample_form_temp():
    print('POSTデータ受け取ったので処理します')
    return 'POST受け取ったよ'

if __name__ == '__main__':
  app.run(host='localhost')