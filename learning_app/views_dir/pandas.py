from django.views.generic import TemplateView
from .base import BaseContext

import pandas as pd

# --- pandas
class PandasView(BaseContext, TemplateView):
    # 教材コード
    category_df = pd.read_csv('learning_app/category.csv')
    df = pd.read_csv('learning_app/sample_pandas_6.csv')
    df = pd.merge(df, category_df[['商品番号', 'カテゴリー']], how='inner', on='商品番号')

    # コンテキスト
    template_name = "common.html"
    heading = 'pandasとは'
    # val = ''
    val = df.to_html
    pre = ''
    # pre = category_df.to_string()
    # pre = df['単価'].apply(tax).to_string() # to_string()を記載しないとif文でエラーが発生
    url = ''