from django.shortcuts import render
from django.views.generic import TemplateView

from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from io import StringIO  # df.info()の出力をキャプチャするために必要

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

# --- コンテキスト
class BaseContext:
    template_name = "common.html"
    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        # テンプレートファイルにparamsデータを渡す
        context = super().get_context_data(**kwargs)
        params = {
            'title': '機械学習 | Django基礎',
            'heading': self.heading,
            'current_path': self.request.path, # パスの取得
            'val': self.val, # 通常の出力
            'pre': self.pre, # <pre>で出力
            'url': self.url, # 画像出力
        }
        context.update(params)
        return context

# --- 時系列分析（1）
class TimeSeries_1_View(BaseContext, TemplateView):
    # 教材コード

    val = ''
    pre = ''

    # コンテキスト
    heading = '時系列分析の手法を学ぼう(1)'
    val = val
    pre = pre
    url = ''
    # url = 'data:image/png;base64,' + urllib.parse.quote(string)