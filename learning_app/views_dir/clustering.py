from django.views.generic import TemplateView
from .base import BaseContext

from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd
import seaborn as sns

# 画像出力用
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja # 日本語化(Python3.12以降 japanize_matplotlibはエラーが発生します)

# --- クラスタリング
class ClusteringView(BaseContext, TemplateView):
    # 教材コード
    df = pd.read_csv('learning_app/california_housing_cleansing.csv')
    df = df.drop(columns=['Unnamed: 0'])
    # val = df.head().to_html()
    # val += str(df.shape)

    # 各要素の標準化
    scaler = StandardScaler()
    X = df.to_numpy()
    scaler.fit(X)
    X_scaled  = scaler.transform(X)
    # pre = X_scaled.tolist()

    # データセットのクラスタリング
    model = KMeans(n_clusters=4, random_state=0)
    model.fit(X_scaled)
    df['クラスター'] = model.labels_
    val = df.groupby('クラスター').mean().to_html()
    # val = df.head().to_html()
    # pre = model.labels_.tolist()

    # 画像をバイト配列に保存
    # plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))  # 2行2列のサブプロット

    # プロット（編集箇所）
    # sns.countplot(x='クラスター', data=df, hue='クラスター', palette="Set2")
    sns.scatterplot(x='経度', y='緯度', hue='クラスター', data=df, ax=ax[0, 0])
    ax[0, 0].set_title('経度 vs 緯度')

    df_cluster2 = df.query('クラスター == 2')
    sns.scatterplot(x='経度', y='緯度', data=df_cluster2, ax=ax[0, 1], color='red')
    ax[0, 1].set_title('経度 vs 緯度')

    sns.histplot(x='地域人口', data=df, bins=50, ax=ax[1, 0])
    ax[1, 0].set_title('地域人口')

    sns.scatterplot(x='経度', y='緯度', data=df_cluster2.query('地域人口 > 5000'), ax=ax[1, 1], color='green')
    ax[1, 1].set_title('経度 vs 緯度 (地域人口 > 5000)')

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read()).decode('utf-8')  # Base64エンコード
    buffer.close()

    # コンテキスト
    heading = 'クラスタリングの手法を学ぼう'
    val = val
    pre = ''
    url = 'data:image/png;base64,' + urllib.parse.quote(string)