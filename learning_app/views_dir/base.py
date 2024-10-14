from typing import Any

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