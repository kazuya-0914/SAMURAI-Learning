from django.views.generic import TemplateView
from .base import BaseContext

import numpy as np
import matplotlib.pyplot as plt

# 画像出力用
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja # 日本語化(Python3.12以降 japanize_matplotlibはエラーが発生します)

# --- Matplotlib
class MatplotlibView(BaseContext, TemplateView):
    # 教材コード（編集箇所）
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([100, 120, 150, 200, 160])
    label = ["昆布", "うめ" , "鮭", "カルビ", "すじこ"]

    # 画像をバイト配列に保存
    plt.figure(figsize=(12, 10))

    # プロット（編集箇所）
    plt.title('おにぎりの具ごとの値段') # matplotlib_fontja使用
    plt.bar(x, y, tick_label=label)

    # プロット実行
    plt.tight_layout()

    # メモリ上に画像を保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read())

    # コンテキスト
    template_name = "common.html"
    heading = 'Matplotlibとは'
    val = ''
    pre = ''
    url = 'data:image/png;base64,' + urllib.parse.quote(string)