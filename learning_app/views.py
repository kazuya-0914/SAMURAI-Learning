import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import Any
from django.http import HttpResponse
from PIL import Image
from io import BytesIO
from io import StringIO

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing

from django.shortcuts import render
from django.views.generic import TemplateView

# 画像出力用
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja # 日本語化(Python3.12以降 japanize_matplotlibはエラーが発生します)

# --- コンテキスト
class BaseContext:
    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        # テンプレートファイルにparamsデータを渡す
        context = super().get_context_data(**kwargs)
        params = {
            'title': '機械学習 | Django基礎',
            'heading': self.heading,
            'val': self.val, # 通常の出力
            'pre': self.pre, # <pre>で出力
            'url': self.url, # 画像出力
        }
        context.update(params)
        return context

# --- トップページ
class TopView(BaseContext, TemplateView):
    template_name = "top.html"
    heading = '機械学習'
    val = ''
    pre = ''
    url = ''
    
# --- NumPy   
class NumPyView(BaseContext, TemplateView):
    template_name = "numpy.html"
    heading = 'NumPyとは'
    val = ''
    pre = ''
    url = '/image/'

# --- NumPy（画像読み込み用）
def image_view(request):
    # 画像ファイルの絶対パスを取得
    image_path = os.path.join('static', 'images', 'camera.jpg')
    # 画像を読み込む
    im = Image.open(image_path)
    im = im.resize((im.width //2, im.height //2))
    # PIL形式からNumPy形式に変換
    im_np = np.asarray(im)
    negative_im_np = 255 - im_np
    negative_im = Image.fromarray(negative_im_np)

    buffer = BytesIO()
    negative_im.save(buffer, format="JPEG")
    image_data = buffer.getvalue()

    return HttpResponse(image_data, content_type="image/jpeg")

# --- Matplotlib
class MatplotlibView(BaseContext, TemplateView):
    # 教材コード（編集箇所）
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([100, 120, 150, 200, 160])
    # label = ["昆布", "うめ" , "鮭", "カルビ", "すじこ"]

    # 画像をバイト配列に保存
    # plt.figure(figsize=(12, 10))

    # プロット（編集箇所）
    # plt.title('おにぎりの具ごとの値段') # matplotlib_fontja使用
    # plt.bar(x, y, tick_label=label)

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read())

    # コンテキスト
    template_name = "matplotlib.html"
    heading = 'Matplotlibとは'
    val = ''
    pre = ''
    url = 'data:image/png;base64,' + urllib.parse.quote(string)

# --- pandas
class PandasView(BaseContext, TemplateView):
    # 教材コード
    category_df = pd.read_csv('learning_app/category.csv')
    df = pd.read_csv('learning_app/sample_pandas_6.csv')
    df = pd.merge(df, category_df[['商品番号', 'カテゴリー']], how='inner', on='商品番号')

    # コンテキスト
    template_name = "pandas.html"
    heading = 'pandasとは'
    # val = ''
    val = df.to_html
    pre = ''
    # pre = category_df.to_string()
    # pre = df['単価'].apply(tax).to_string() # to_string()を記載しないとif文でエラーが発生
    url = ''

# --- scikit-learn
class ScikitLearnView(BaseContext, TemplateView):
    # 教材コード
    dataset = load_wine()
    dataset.feature_names
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    model = DecisionTreeClassifier(random_state=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    X_real = np.array([
        [13, 1.6, 2.2, 16, 118, 2.6, 2.9, 0.21, 1.6, 5.8, 0.92, 3.2, 1011],
        [12, 2.8, 2.2, 18, 100, 2.5, 2.3, 0.25, 2.0, 2.2, 1.15, 3.3, 1000],
        [14, 4.1, 2.7, 24, 101, 1.6, 0.7, 0.53, 1.4, 9.4, 0.61, 1.6, 560]
    ])

    # コンテキスト
    template_name = "scikit-learn.html"
    heading = 'scikit-learnとは'
    val = model.score(X_test, y_test)
    pre = model.predict(X_real).tolist()
    # pre = y_pred.tolist() # 配列系はtolist()を記載しないとif文でエラーが発生
    # pre = train_test_split(X, y , test_size=0.3, random_state=5)
    url = ''

# --- seaborn
class SeabornView(BaseContext, TemplateView):
    # 教材コード
    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['Price'] = dataset.target
    feature_names_JPN = ['所得', '築年数', '部屋数', '寝室数', '地域人口', '世帯人数', '緯度', '経度', '住宅価格']
    df.columns = feature_names_JPN

    # dfのheadをHTML形式で取得
    val = f"{df.shape}{df.describe().to_html()}"

    # df.info()の出力をキャプチャ
    buffer = StringIO()
    df.info(buf=buffer)
    pre = buffer.getvalue()

    # 画像をバイト配列に保存
    # plt.figure(figsize=(12, 10))

    # プロット（編集箇所）
    # df.hist(bins=30, figsize=(12, 10))
    sns.stripplot(x='地域人口', data=df)

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read())

    # コンテキスト
    template_name = "seaborn.html"
    heading = 'seabornとは'
    val = val
    pre = pre
    url = 'data:image/png;base64,' + urllib.parse.quote(string)