
import torch.nn as nn
from midas import midas_net   # midas.py に定義されているベースクラス

class DepthEstimationModel(midas_net):
    def __init__(self, **kwargs):
        """
        midas_net の引数をそのまま渡せるように **kwargs を受け取ります
        例: variant='dpt_large', non_negative=True など
        """
        super().__init__(**kwargs)
        # 必要ならここで追加のヘッドや微調整層を定義

    # forward は継承された midas_net のまま