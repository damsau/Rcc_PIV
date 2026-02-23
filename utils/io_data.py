# モジュール
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os.path as osp


class ParticleImageDataset(Dataset):
    def __init__(self, root, filename, pivstep_skip, ref_skip_low, ref_skip_high):
        """
        root: 粒子画像のデータディレクトリ（例: /images）
        """
        super().__init__()

        print("\n粒子画像データを読み込みます...")

        # 動画ファイルを取得
        self.import_path = osp.join(root, filename)
        cap = cv2.VideoCapture(self.import_path)
        if not cap.isOpened():
            print("動画ファイルが開けませんでした．")
            exit()

        # 動画情報を取得
        self.img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frm_rate = cap.get(cv2.CAP_PROP_FPS)
        self.frm_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # パラメータ設定
        self.ref_skip_low = ref_skip_low
        self.ref_skip_high = ref_skip_high
        self.pivstep_skip = pivstep_skip
        if self.ref_skip_low < self.ref_skip_high:
            print("解析間隔の設定が間違っています．")
            exit()
        self.dt = {
            "high": (1 / self.frm_rate) * (ref_skip_high + 1),
            "low": (1 / self.frm_rate) * (ref_skip_high + 1),
        }

        # 0始まりのインデックス
        self.start = ref_skip_low + 1
        self.end = self.frm_num - (ref_skip_low + 1) - 1
        self.pivstep_skip = pivstep_skip
        self.pivstep_num = int((self.end - self.start + 1) / (self.pivstep_skip + 1))

        # 全フレームを一括読み込み
        img_list = []
        while True:
            _ret, _frame = cap.read()
            if not _ret:
                break

            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)  # グレースケールに変換
            img_list.append(_frame)

        cap.release()

        self.img_tensor = torch.from_numpy(np.array(img_list)).float()  # Tensorに変換
        self.img_tensor = self.img_tensor.unsqueeze(
            1
        )  # 次元変換 (B, H, W) -> (B, C=1, H, W)
        del img_list

    def __len__(self):
        return self.pivstep_num

    def __getitem__(self, idx, skip_low=None, skip_high=None):
        if skip_low == None:
            skip_low = self.ref_skip_low
        if skip_high == None:
            skip_high = self.ref_skip_high

        get_idx = [
            idx - 1 - skip_low,
            idx - 1 - skip_high,
            idx,
            idx + 1 + skip_high,
            idx + 1 + skip_low,
        ]

        return self.img_tensor[get_idx]
