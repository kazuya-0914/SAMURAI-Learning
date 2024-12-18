from django.urls import path
from .views_dir.top import TopView
from .views_dir.numpy import NumPyView, image_view
from .views_dir.matplotlib import MatplotlibView
from .views_dir.pandas import PandasView
from .views_dir.scikit_learn import ScikitLearnView
from .views_dir.seaborn import SeabornView
from .views_dir.regression_1 import Regression_1_View
from .views_dir.regression_2 import Regression_2_View
from .views_dir.categorize import CategorizeView
from .views_dir.clustering import ClusteringView
from .views_dir.analysis_1 import Analysis_1_View
from . import views

urlpatterns = [
    path('', TopView.as_view(), name='top'),
    path('numpy/', NumPyView.as_view(), name='numpy'),
    path('image/', image_view, name='image'),
    path('matplotlib/', MatplotlibView.as_view(), name='matplotlib'),
    path('pandas/', PandasView.as_view(), name='pandas'),
    path('scikit-learn/', ScikitLearnView.as_view(), name='scikit-learn'),
    path('seaborn/', SeabornView.as_view(), name='seaborn'),
    path('regression-1/', Regression_1_View.as_view(), name='regression-1'),
    path('regression-2/', Regression_2_View.as_view(), name='regression-2'),
    path('categorize/', CategorizeView.as_view(), name='categorize'),
    path('clustering/', ClusteringView.as_view(), name='clustering'),
    path('analysis-1/', Analysis_1_View.as_view(), name='analysis-1'),
    path('analysis-2/', views.Analysis_2_View.as_view(), name='analysis-2'),
]