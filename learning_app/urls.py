from django.urls import path
from . import views

urlpatterns = [
    path('', views.TopView.as_view(), name='top'),
    path('numpy/', views.NumPyView.as_view(), name='numpy'),
    path('image/', views.image_view, name='image'),
    path('matplotlib/', views.MatplotlibView.as_view(), name='matplotlib'),
    path('pandas/', views.PandasView.as_view(), name='pandas'),
    path('scikit-learn/', views.ScikitLearnView.as_view(), name='scikit-learn'),
    path('seaborn/', views.SeabornView.as_view(), name='seaborn'),
]