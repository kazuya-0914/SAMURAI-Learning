from django.views.generic import TemplateView
from .base import BaseContext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 画像出力用
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja # 日本語化(Python3.12以降 japanize_matplotlibはエラーが発生します)

# --- 回帰（１）
class Regression_1_View(BaseContext, TemplateView):
    # 教材コード
    df = pd.read_csv('learning_app/california_housing_cleansing.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    val = f"{df.shape}{df.head().to_html()}"

    # ndarrayへの変換
    X = df.drop(columns=['住宅価格']).to_numpy()
    y = df['住宅価格'].to_numpy()

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # val = X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # modelに代入し学習
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 予測モデルの評価
    # val = f"{model.score(X_train, y_train)}<br>"
    # val += str(model.score(X_test, y_test))

    # 予測
    X_new = np.array([
        [8, 41, 500, 37, -120, 1, 0.2],
        [2, 10, 2000, 38, -122, 1.5, 0.5],
        [1, 25, 1000, 38, -121, 2, 1],
    ])
    # pre = str(model.predict(X_new))

    # 住宅価格の予測
    val = model.intercept_
    pre = str(model.coef_)

    # プロット用のデータ
    x_labels = ['所得', '築年数', '地域人口', '緯度', '経度', '部屋数/人', '寝室数/人']
    y_values = model.coef_

    # 画像をバイト配列に保存
    plt.figure(figsize=(12, 10))

    # プロット（編集箇所）
    sns.barplot(x=x_labels, y=y_values, palette="Set2")

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read()).decode('utf-8')  # Base64エンコード

    # コンテキスト
    template_name = "common.html"
    heading = '回帰の手法を学ぼう(1)'
    val = val
    pre = pre
    url = 'data:image/png;base64,' + urllib.parse.quote(string)