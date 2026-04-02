# モジュール
import numpy as np
import cv2
import torch
import os.path as osp


class ParticleImageDataset:
    def __init__(self, root, filename, n_buffer=2):
        """
        root: 粒子画像のデータディレクトリ（例: /images）
        filename: 粒子画像動画のファイル名
        buffer_num: 保持する粒子画像の枚数
        """
        super().__init__()

        # 動画ファイルを取得
        self.IMPORT_PATH = osp.join(root, filename)
        cap = cv2.VideoCapture(self.IMPORT_PATH)
        if not cap.isOpened():
            print("動画ファイルが開けませんでした．")
            exit()

        # 動画情報を取得
        self.WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FRAMERATE = cap.get(cv2.CAP_PROP_FPS)
        self.N_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # パラメータ設定
        self.N_BUFFER = n_buffer

        cap.release()

    def get_images(self, i_frame=0):
        if i_frame < 0 or i_frame > self.N_FRAME:
            print("!!! 解析範囲外のフレームです !!!")
            exit

        # 読み込むフレーム番号の設定
        list_frame_number = [
            i_frame + i
            for i in range(-int(self.N_BUFFER / 2), int(self.N_BUFFER / 2) + 1)
        ]

        cap = cv2.VideoCapture(self.IMPORT_PATH)

        images = []
        for i in range(len(list_frame_number)):
            # 指定したフレーム番号へシーク（ジャンプ）
            cap.set(cv2.CAP_PROP_POS_FRAMES, list_frame_number[i])
            ret, image = cap.read()

            if not ret:
                # 万が一読み込めなかった場合の安全対策（真っ黒の画像を返すなど）
                print("画像が読み込めてません")
                image = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            images.append(image)

        cap.release()

        # numpy配列をPyTorchテンソルに変換: (buffer, H, W) -> (buffer, 1, H, W)
        images = torch.from_numpy(np.array(images)).float().unsqueeze(1)

        return list_frame_number, images
