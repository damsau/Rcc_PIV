# モジュール
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import tifffile as tf
from pathlib import Path
import pprint as pp
import glob
import sys


def main():
    # デバイス設定
    print("\n■ デバイスと環境の設定を確認します...")
    print(f"> PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"> CUDA Available: Yes")
        print(f"> GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"> CUDA Available: No")
        print(f"> Warning: Training URAFT on CPU will be very slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Device: {device}")

    # ディレクトリとファイル
    program_path = Path(__file__)
    file_name = program_path.name
    print(f"\n実行ファイル名：{file_name}")
    work_dir = program_path.parent.parent
    print(f"作業ディレクトリ：{work_dir}")

    # パラメータ読み込みテスト
    print("\nパラメータの読み込みテスト")
    pivparams = PivParams(dir=work_dir / "parameters")

    # 画像読み込みテスト
    print("\n解析画像の読み込みテスト")
    pivimages = PivImages(
        dir=work_dir / "images" / "0.1rps_001", file_name="0.1rps_100_Fps100_us*.tif"
    )
    pivimages.FRAME_INDEX = 100
    image_tmp = pivimages.get_image()


if __name__ == "__main__":
    main()
