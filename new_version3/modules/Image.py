import matplotlib.pyplot as plt
import yaml
import torch
import tifffile as tf
from pathlib import Path
import pprint as pp
import glob
import sys


class PivImages:
    def __init__(self, dir=None, file_name=None):
        """
        PIV画像のクラス
        Args:
            dir: 画像ファイルがあるディレクトリ
            file_name: 画像ファイルの名前
        """
        if dir == None:
            print("PIV解析画像のディレクトリが指定されていません")
            sys.exit()
        if file_name == None:
            print("PIV解析画像の名前が指定されていません")
            sys.exit()

        file = Path(dir / file_name)
        print(f"\n■ 画像ファイル（{file}）を読み込みます")

        self.IMAGE_FILE_LIST = sorted(glob.glob(pathname=str(file)))
        self.IMAGE_FILE_NUM = len(self.IMAGE_FILE_LIST)

        if self.IMAGE_FILE_NUM == 0:
            print("PIV解析画像が指定された場所に存在しません")
            sys.exit()

        # ファイル形式の確認
        file_suffix = file.suffix
        if file_suffix != ".tif":
            print("ファイル拡張子が対応してません")
            sys.exit()

        # テスト読み込み
        self.FRAME_INDEX = 0
        image_tmp = tf.imread(self.IMAGE_FILE_LIST[self.FRAME_INDEX]).astype(np.float32)
        print(f"データ型: {image_tmp.dtype}")
        print(f"サイズ・形状: {image_tmp.shape}")
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = image_tmp.shape

        # if True:
        if False:
            # 画像を表示するためのウィンドウ（フィギュア）を作成
            plt.figure(figsize=(8, 6))

            # 画像データを表示
            # cmap='gray': グレースケールで表示する
            # vmin, vmax: 表示する輝度の最小値と最大値を指定（省略も可能）
            plt.imshow(image_tmp, cmap="gray")

            # カラーバー（輝度値と色の対応表）を追加（任意）
            plt.colorbar(label="Intensity")

            # タイトルを追加（任意）
            plt.title(f"Frame Index: {self.FRAME_INDEX}")

            # 画面に表示する（この行を実行しないとウィンドウが開きません）
            plt.show()

    def get_image(self):
        """
            一枚の画像を読み込む
        Args:
            target_frame_num: ターゲット画像のフレーム番号
        """
        image = None
        try:
            image = tf.imread(self.IMAGE_FILE_LIST[self.FRAME_INDEX]).astype(np.float32)
        except IndexError:
            print(
                f"フレーム番号{self.FRAME_INDEX}が適切でないか画像が存在しないため，読み込みに失敗しました．"
            )
            return None

        if image is not None:
            return image
