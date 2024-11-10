from django.views.generic import TemplateView
from .base import BaseContext

from typing import Any
from io import StringIO  # df.info()の出力をキャプチャするために必要
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

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

# --- 時系列分析（1）
class Analysis_1_View(BaseContext, TemplateView):
    # 教材コード
    df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv')

    # df.info()の出力をキャプチャ
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()  # info()の出力を文字列として取得
    # pre = info_str
    pre = ''

    # 予測モデルの学習
    model = Prophet()
    model.fit(df)

    # 予測
    future = model.make_future_dataframe(periods=36, freq='MS')
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    # val = ''

    # 予測の評価
    cutoffs = pd.to_datetime(['1954-12-01', '1955-12-01', '1956-12-01', '1957-12-01'])
    df_cv = cross_validation(model, horizon = '1096 days', cutoffs=cutoffs)
    df_p = performance_metrics(df_cv, monthly=True)
    # val = df_p.head().to_html()
    val = ''

    # 画像をバイト配列に保存
    # plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1行2列のサブプロット

    # プロット（編集箇所）
    time = pd.to_datetime(df['ds'])
    # sns.lineplot(x=time, y=df['y'])
    # sns.lineplot(x='horizon', y='coverage', data=df_p)

    sns.lineplot(x='horizon', y='mse', data=df_p, ax=ax[0], color='red')
    ax[0].set_title('mseの推移')

    sns.lineplot(x='horizon', y='coverage', data=df_p, ax=ax[1], color='green')
    ax[1].set_title('coverageの推移')

    # fig_forecast = model.plot(forecast)

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    # fig_forecast.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read()).decode('utf-8')  # Base64エンコード

    # コンテキスト
    heading = '時系列分析の手法を学ぼう(1)'
    val = val
    pre = pre
    url = 'data:image/png;base64,' + urllib.parse.quote(string)
    # url = ''