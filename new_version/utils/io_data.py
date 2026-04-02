# モジュール
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os.path as osp


class ParticleImageDataset(Dataset):
    def __init__(self, root, filename, pivstep_skip, buffer_num):
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
        self.pivstep_skip = pivstep_skip
        self.buffer_num = buffer_num
        if self.ref_skip_low < self.ref_skip_high:
            print("解析間隔の設定が間違っています．")
            exit()
        self.dt = {
            "high": (1 / self.frm_rate) * (ref_skip_high + 1),
            "low": (1 / self.frm_rate) * (ref_skip_low + 1),
        }

        # 0始まりのインデックス
        self.start = ref_skip_low + 1
        self.end = self.frm_num - (ref_skip_low + 1) - 1
        self.pivstep_skip = pivstep_skip
        self.pivstep_num = int((self.end - self.start + 1) / (self.pivstep_skip + 1))

        cap.release()

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

        cap = cv2.VideoCapture(self.import_path)

        frames = []
        for i in get_idx:
            # 指定したフレーム番号へシーク（ジャンプ）
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                # 万が一読み込めなかった場合の安全対策（真っ黒の画像を返すなど）
                print("画像が読み込めてません")
                frame = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        cap.release()

        # numpy配列をPyTorchテンソルに変換: (5, H, W) -> (5, 1, H, W)
        img_tensor = torch.from_numpy(np.array(frames)).float().unsqueeze(1)

        return img_tensor
