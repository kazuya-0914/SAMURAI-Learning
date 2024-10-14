from django.views.generic import TemplateView
from .base import BaseContext

from typing import Any
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from io import StringIO  # df.info()の出力をキャプチャするために必要

import pandas as pd

# 画像出力用
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja # 日本語化(Python3.12以降 japanize_matplotlibはエラーが発生します)

# --- 分類
class CategorizeView(BaseContext, TemplateView):
    # 教材コード
    dataset = load_breast_cancer()

    # データの内容の確認
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['class'] = dataset.target
    # val = df.head().to_html()
    # val += str(df.shape)

    # pre = str(dataset.DESCR)

    # df.info()の出力をキャプチャ
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()  # info()の出力を文字列として取得
    # pre = info_str

    # 学習データとテストデータへの分割
    X = df.drop(columns=['class']).to_numpy()
    y = df['class'].to_numpy()

    # 比率7:3で学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # pre = X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # 予測モデルの学習
    # model = DecisionTreeClassifier(random_state=0)
    # 条件分岐の構造に対する制約の設定
    model = DecisionTreeClassifier(
        max_depth=2,
        max_leaf_nodes=3,
        min_samples_leaf=10,
        random_state=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # pre = y_pred.tolist()

    # 目的変数のテストデータ
    # pre = y_test.tolist()

    # 評価指標の出力
    # pre = classification_report(y_test, y_pred)

    # 検査データ
    df_X_new = pd.read_csv('learning_app/data_breastcancer.csv') # 最初に「/」を記載するとエラー発生
    X_new = df_X_new.to_numpy()
    val = df_X_new.head().to_html()
    # pre = model.predict(X_new).tolist()

    # 判定根拠の可視化
    names = dataset.feature_names
    names_list = names.tolist()
    pre = export_text(model, decimals=3, feature_names=names_list)

    # 画像プロット（編集箇所）
    df.hist(figsize=(12, 10), bins=30)

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read())

    # コンテキスト
    heading = '分類の手法を学ぼう'
    val = val
    pre = pre
    url = ''
    # url = 'data:image/png;base64,' + urllib.parse.quote(string)