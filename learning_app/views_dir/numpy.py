from django.views.generic import TemplateView
from .base import BaseContext

from django.http import HttpResponse
import numpy as np
from PIL import Image
from io import BytesIO
import os

# --- NumPy   
class NumPyView(BaseContext, TemplateView):
    heading = 'NumPyとは'
    val = ''
    pre = ''
    url = '/image/'

# --- NumPy（画像読み込み用）
def image_view(request):
    # 画像ファイルの絶対パスを取得
    image_path = os.path.join('static', 'images', 'camera.jpg')
    # 画像を読み込む
    im = Image.open(image_path)
    im = im.resize((im.width //2, im.height //2))
    # PIL形式からNumPy形式に変換
    im_np = np.asarray(im)
    negative_im_np = 255 - im_np
    negative_im = Image.fromarray(negative_im_np)

    buffer = BytesIO()
    negative_im.save(buffer, format="JPEG")
    image_data = buffer.getvalue()

    return HttpResponse(image_data, content_type="image/jpeg")