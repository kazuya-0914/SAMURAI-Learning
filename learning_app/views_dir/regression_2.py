from django.views.generic import TemplateView
from .base import BaseContext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

# --- 回帰（２）
class Regression_2_View(BaseContext, TemplateView):
    # 教材コード
    df = pd.read_csv('learning_app/california_housing_cleansing.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    # val = df.head().to_html()

    # 多重共線性の対処
    val = df.drop(columns=['住宅価格']).corr().to_html()

    # ndarrayへの変換
    # X = df.drop(columns=['住宅価格']).to_numpy()
    # y = df['住宅価格'].to_numpy()
    X = df[['所得', '築年数', '地域人口', '緯度', '部屋数/人']].to_numpy()
    y = df['住宅価格'].to_numpy()

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # StandardScalerクラスのインスタンス化
    scaler = StandardScaler()

    # 標準化の変換モデルの生成
    scaler.fit(X_train)

    # スケールの変換
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 変換前の状態を表示
    # df_X_train = pd.DataFrame(X_train, columns=['所得', '築年数', '地域人口', '緯度', '経度', '部屋数', '寝室数'])
    # val = f"変換前{df_X_train.head().to_html()}"

    # 変換後の状態を表示
    # df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=['所得', '築年数', '地域人口', '緯度', '経度', '部屋数', '寝室数'])
    # val += f"変換後{df_X_train_scaled.head().to_html()}"

    # 平均値と標準偏差を表示
    # val += f"平均値と標準偏差{df_X_train_scaled.describe().to_html()}"

    # テストデータのスケール変換
    X_test_scaled = scaler.transform(X_test)

    # 変換前の状態を表示
    # df_X_test= pd.DataFrame(X_test, columns=['所得', '築年数', '地域人口', '緯度', '経度', '部屋数', '寝室数'])
    # val = f"変換前{df_X_test.head().to_html()}"

    # 変換後の状態を表示
    # df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=['所得', '築年数', '地域人口', '緯度', '経度', '部屋数', '寝室数'])
    # val += f"変換後{df_X_test_scaled.head().to_html()}"

    # 平均値と標準偏差を表示
    # val += f"平均値と標準偏差{df_X_test_scaled.describe().to_html()}"

    # modelに代入し学習
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 予測モデルの評価
    val += f"{model.score(X_train_scaled, y_train)}<br>"
    val += f"{model.score(X_test_scaled, y_test)}<br>"

    # 予測
    '''
    X_new = np.array([
        [8, 41, 500, 37, -120, 1, 0.2],
        [2, 10, 2000, 38, -122, 1.5, 0.5],
        [1, 25, 1000, 38, -121, 2, 1],
    ])
    '''
    X_new = np.array([
        [8, 41, 500, 38, 2],
        [10, 10, 1000, 40, 1],
        [7.5, 25, 3500, 39, 3],
    ])

    X_new_scaled = scaler.transform(X_new)
    # pre = str(X_new_scaled)
    # pre += "\n------\n"
    pre = str(model.predict(X_new_scaled))

    # 住宅価格の予測
    # val += f"{model.intercept_}"
    pre += "\n------\n"
    pre += str(model.coef_)
    pre += "\n------\n"
    pre += str(model.intercept_)

    # プロット用のデータ（棒グラフ）
    # x_labels = ['所得', '築年数', '地域人口', '緯度', '経度', '部屋数/人', '寝室数/人']
    x_labels = ['所得', '築年数', '地域人口', '緯度', '部屋数/人']
    y_values = model.coef_

    # 画像をバイト配列に保存
    plt.figure(figsize=(12, 10))
    # fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1行2列のサブプロット

    # プロット（編集箇所）
    sns.barplot(x=x_labels, y=y_values, hue=x_labels, palette="Set2")

    # 最初の散布図（部屋数/人 vs 寝室数/人）
    # sns.scatterplot(x='部屋数/人', y='寝室数/人', data=df, ax=ax[0], color='blue')
    # ax[0].set_title('部屋数/人 vs 寝室数/人')

    # 2つ目の散布図（経度 vs 緯度）
    # sns.scatterplot(x='経度', y='緯度', data=df, ax=ax[1], color='red')
    # ax[1].set_title('経度 vs 緯度')

    # 凡例を追加
    # plt.legend()

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read()).decode('utf-8')  # Base64エンコード

    # コンテキスト
    heading = '回帰の手法を学ぼう(2)'
    val = val
    pre = pre
    url = 'data:image/png;base64,' + urllib.parse.quote(string)