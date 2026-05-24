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


class PivParams:
    def __init__(self, dir=None, file_name=Path("default_params.yaml")):
        """
        PIVパラメータのクラス
        Args:
            dir: パラメータファイルがあるディレクトリ
            file_name: パラメータファイルの名前
        """
        if dir == None:
            print("パラメータのディレクトリが指定されていません")
            sys.exit()

        # パラメータファイルの読み込み
        params_file = dir / file_name
        print(f"\n■ パラメータファイル（{params_file}）を読み込みます")
        with open(params_file, "r", encoding="utf-8") as params_file:
            params = yaml.safe_load(params_file)

        print("読み込んだパラメータ一覧：")
        pp.pprint(params)

        # 読み込んだパラメータの保存
        self.DATA_NAME = params["data_name"]
        self.PIXEL_TO_MM = params["pixel_to_mm"]
        self.FRAMERATE = params["framerate"]
        self.IMG_EXT = params["img_ext"]
        self.START = params["start"]
        self.END = params["end"]
        self.SKIP = params["skip"]
        self.N_RCC = params["n_rcc"]
        self.IW_SIZE = params["iw_size"]
        self.SW_SIZE = params["sw_size"]
        self.MARGIN = params["margin"]
        self.OVERLAP = params["overlap"]
        self.ERROR_THRESHOLD = params["error_threshold"]
        self.BUFFER = params["buffer"]
        self.DEVICE = None
        self.IMAGE_HEIGHT = None
        self.IMAGE_WIDTH = None
        self.N_WINDOW = None
        self.N_X = None
        self.N_Y = None
        self.IW_POSIT = None
        self.CM_SIZE = None
