from django.views.generic import TemplateView
from .base import BaseContext

# --- トップページ
class TopView(BaseContext, TemplateView):
    template_name = "top.html"
    heading = '機械学習'
    val = ''
    pre = ''
    url = ''