from django.views.generic import TemplateView
from .base import BaseContext

import pandas as pd
from sklearn.datasets import fetch_california_housing
from io import StringIO

# 画像出力用
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja # 日本語化(Python3.12以降 japanize_matplotlibはエラーが発生します)

# --- seaborn
class SeabornView(BaseContext, TemplateView):
    # 教材コード
    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['Price'] = dataset.target
    feature_names_JPN = ['所得', '築年数', '部屋数', '寝室数', '地域人口', '世帯人数', '緯度', '経度', '住宅価格']
    df.columns = feature_names_JPN

    # 築年数が52以外のデータのみを抽出
    df = df[df['築年数'] != 52]

    # 住宅価格が5.000010以外のデータのみを抽出
    df = df[df['住宅価格'] != 5.000010]

    df['世帯数'] = df['地域人口'] / df['世帯人数']
    df['全部屋数'] = df['部屋数'] * df['世帯数']
    df['全寝室数'] = df['寝室数'] * df['世帯数']
    df['部屋数/人'] = df['全部屋数']/df['地域人口']
    df['寝室数/人'] = df['全寝室数']/df['地域人口']
    # val = df.head().to_html()
    df = df.drop(columns = ['部屋数', '寝室数', '世帯人数', '世帯数', '全部屋数', '全寝室数'])
    val = df.shape

    # df.info()の出力をキャプチャ
    buffer = StringIO()
    df.info(buf=buffer)
    pre = buffer.getvalue()
    # pre = ''

    # プロット（編集箇所）
    df.hist(figsize=(12, 10), bins=30)

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read())

    # コンテキスト
    heading = 'seabornとは'
    val = val
    pre = pre
    url = 'data:image/png;base64,' + urllib.parse.quote(string)

    # 前処理済データの保存
    # df.to_csv('learning_app/california_housing_cleansing.csv')