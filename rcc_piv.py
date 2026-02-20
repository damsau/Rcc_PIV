# モジュール
import yaml
import argparse
import pprint as pp
import os.path as osp
import torch
from torch.utils.data import DataLoader

# 自作モジュール
from utils import io_data as id
from utils import tools
from modules import piv_modules as pm

# デバイス設定
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA Available: No")
    print(f"Warning: Training URAFT on CPU will be very slow.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# RCC-PIV解析のメイン関数
def rcc_piv(params):
    print("PIV解析を実行します...")
    # パラメータ取得
    io_params = params["file_io"]
    piv_params = params["rcc_piv"]

    # Datasetの作成
    particle_images = id.ParticleImageDataset(
        root=osp.join(io_params["workdir_path"], "images"),
        filename=piv_params["img_name"] + piv_params["img_ext"],
        ref_skip_low=piv_params["ref_skip_low"],
        ref_skip_high=piv_params["ref_skip_high"],
        pivstep_skip=piv_params["pivstep_skip"],
    )

    # 解析範囲の設定
    if piv_params["start"] == "None" and piv_params["end"] == "None":
        start = particle_images.start
        end = particle_images.end
        skip = particle_images.pivstep_skip
    else:
        start = piv_params["start"]
        end = piv_params["end"]
        skip = piv_params["pivstep_skip"]

    # piv解析のループ
    for piv_step in range(start, end + 1, skip + 1):
        print(f"piv_step: {piv_step} / {end}")

        # 解析するペア画像を取得
        imgs = particle_images[piv_step].to(device)

        # 変数の初期化
        rcc_num = piv_params["rcc_num"]
        displacement = None
        iw_center_old = None

        # 再帰処理ループ
        for rcc_step in range(rcc_num):
            print(f"rcc_step: {rcc_step + 1} / {rcc_num}")

            # 検査画像・探査画像の設定
            windows = pm.WindowSetup(
                piv_params,
                rcc_step,
                particle_images.img_width,
                particle_images.img_height,
                imgs,
            )

            # 速度点数を取得
            ny, nx = windows.get_dimention()

            # 検査画像の取得（次元: [1, 1, ny, nx, iw_size, iw_size]）
            interrogation_imgs = windows.get_iw_img()
            iw_lt, iw_rb = windows.get_iw_lt_and_rb()
            iw_center = windows.get_iw_centers()
            print("interrogation_img.shape: ", interrogation_imgs.shape)

            # オフセット変位の計算
            offset = windows.get_offset(displacement, iw_center_old, iw_center)
            print(offset)

            # 探査画像の取得（次元: [time, 1, ny, nx, sw_size, sw_size]）
            sw_lt, sw_rb = windows.get_sw_lt_and_rb(offset=None)
            search_imgs = windows.get_sw_img(sw_lt)
            print("search_imgs.shape: ", search_imgs.shape)

            # 画像相関（輝度値の共分散）を計算
            matcher = pm.PatternMatch(interrogation_imgs, search_imgs)
            correlation_map = matcher.get_correlation_map()
            print("correlation_map.shape: ", correlation_map.shape)
            averaged_correlation_map = matcher.average_correlation_map(correlation_map)
            print("averaged_correlation_map.shape: ", averaged_correlation_map.shape)

            # 最大相関値の位置と値を取得
            peak_vals_high, peak_idx_high = matcher.get_max_correlation_data(
                averaged_correlation_map[0, ...]
            )  # 高速域
            peak_vals_low, peak_idx_low = matcher.get_max_correlation_data(
                averaged_correlation_map[1, ...]
            )  # 高速域
            print("peak_vals_high.shape: ", peak_vals_high.shape)
            print("peak_idx_high.shape: ", peak_idx_high.shape)

            # 変位を計算
            displacement = windows.get_displacement(peak_idx_high, iw_lt, sw_lt)
            print("displacement.shape", displacement.shape)

            # 検査領域の中心座標を保存
            iw_center_old = iw_center


if __name__ == "__main__":
    # パラメータの読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", type=str, default="-")
    args = parser.parse_args()
    with open(args.params_file, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        pp.pprint(params, sort_dicts=False)

    # piv解析実行
    rcc_piv(params)
