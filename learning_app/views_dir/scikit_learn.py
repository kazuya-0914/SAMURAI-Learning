from django.views.generic import TemplateView
from .base import BaseContext

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    template_name = "common.html"
    heading = 'scikit-learnとは'
    val = model.score(X_test, y_test)
    pre = model.predict(X_real).tolist()
    # pre = y_pred.tolist() # 配列系はtolist()を記載しないとif文でエラーが発生
    # pre = train_test_split(X, y , test_size=0.3, random_state=5)
    url = ''