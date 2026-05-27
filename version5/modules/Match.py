# モジュール
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn.functional as F
import tifffile as tf
from pathlib import Path
import pprint as pp
import glob
import sys


# 変数の形状と保存場所を確認する関数
def check_variable(var_name, var):
    print("[変数チェック]")
    print(f"    変数名: {var_name}")
    print(f"    次元: {var.shape}")
    print(f"    場所: {var.device}\n")


# 画像配列を確認する関数
def check_image(image):
    if len(image.shape) != 2:
        print("画像配列が２次元ではありません")
        return
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()


class PivMatch:
    def __init__(self, PivParams):
        """
        画像相関を計算するクラス
        """
        self.IMAGE_HEIGHT = PivParams.IMAGE_HEIGHT
        self.IMAGE_WIDTH = PivParams.IMAGE_WIDTH
        self.N_RCC = PivParams.N_RCC
        self.IW_SIZE = PivParams.IW_SIZE
        self.SW_SIZE = PivParams.SW_SIZE
        self.MARGIN = PivParams.MARGIN
        self.OVERLAP = PivParams.OVERLAP
        self.DEVICE = PivParams.DEVICE
        self.N_WINDOW = PivParams.N_WINDOW
        self.N_X = PivParams.N_X
        self.N_Y = PivParams.N_Y
        self.CM_SIZE = [a - b + 1 for a, b in zip(self.SW_SIZE, self.IW_SIZE)]

    def get_correlation_map(
        self, n_window, iw_size, sw_size, interrogation_images, search_images
    ):
        """
        相関係数配列を取得
        """

        # 検査画像の次元調整
        interrogation_images = interrogation_images.reshape(
            n_window, 1, iw_size, iw_size
        )

        # 各検査画像の平均値を計算
        interrogation_images_mean = torch.mean(
            interrogation_images, dim=(-2, -1), keepdim=True
        )

        # 検査画像を中心化（平均を0に）
        interrogation_images_zero_mean = (
            interrogation_images - interrogation_images_mean
        )

        # 探査画像の次元調整
        search_images = search_images.reshape(1, n_window, sw_size, sw_size)

        # 相関係数の配列サイズ
        cm_size = sw_size - iw_size + 1

        # 相関係数の分子を計算
        correlation_coef_nume = F.conv2d(
            search_images, interrogation_images_zero_mean, groups=n_window
        ).reshape(n_window, cm_size, cm_size)

        # 検査画像の標準偏差
        interrogation_images_sq_sum = torch.sum(
            interrogation_images_zero_mean**2, dim=(-2, -1), keepdim=True
        )
        interrogation_images_std = torch.sqrt(
            torch.clamp(interrogation_images_sq_sum, min=1e-8)
        )

        # 探査画像の標準偏差
        ones_filter = torch.ones(n_window, 1, iw_size, iw_size, device=self.DEVICE)
        search_images_sum = F.conv2d(search_images, ones_filter, groups=n_window)
        search_images_sq_sum = F.conv2d(search_images**2, ones_filter, groups=n_window)
        search_images_var = search_images_sq_sum - search_images_sum**2 / (
            iw_size * iw_size
        )
        search_images_std = torch.sqrt(torch.clamp(search_images_var, min=1e-8))

        # 相関係数の分母を計算
        correlation_coef_deno = (
            interrogation_images_std.reshape(1, n_window, 1, 1) * search_images_std
        )

        # 相関係数の計算
        correlation_coef = correlation_coef_nume / correlation_coef_deno

        # 相関係数の範囲を-1~1に修正
        correlation_map = torch.clamp(correlation_coef, min=-1, max=1).reshape(
            n_window, cm_size, cm_size
        )

        return correlation_map

    def get_peak_positions_and_values(self, n_window, correlation_map):
        """
        相関係数配列のピークの位置を取得
        """
        # 相関係数がピークとなるときの相関配列上の位置を取得
        # 相関係数配列の大きさ取得
        cm_size = correlation_map.shape[1]

        # 次元の調整
        correlation_map_ = correlation_map.reshape(n_window, -1)

        # 最大値とその位置の取得
        max_value, max_posit = torch.max(correlation_map_, dim=-1)

        max_posit_y = max_posit // cm_size
        max_posit_x = max_posit % cm_size

        # 相関係数がピークである周囲9点の相関係数を取得
        # 最大値が相関係数の配列境界に位置することを防ぐ
        pad_y = torch.clamp(max_posit_y, 1, cm_size - 2)
        pad_x = torch.clamp(max_posit_x, 1, cm_size - 2)

        # 3x3の相対オフセット
        dy = torch.tensor([-1, 0, 1], device=correlation_map.device)
        dx = torch.tensor([-1, 0, 1], device=correlation_map.device)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")

        # 全窓に対して，最大点を基準にした3x3の絶対座標を計算
        grid_y_abs = pad_y.view(n_window, 1, 1) + grid_y.view(1, 3, 3)
        grid_x_abs = pad_x.view(n_window, 1, 1) + grid_x.view(1, 3, 3)

        # アドバンスド・インデックス用の一括窓インデックス
        batch_idx = torch.arange(n_window, device=correlation_map.device).view(
            n_window, 1, 1
        )

        peak_vals = correlation_map[batch_idx, grid_y_abs, grid_x_abs]

        peak_posit = torch.stack([max_posit_y, max_posit_x], dim=-1)

        return peak_posit, peak_vals
