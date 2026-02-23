# モジュール
import yaml
import argparse
import pprint as pp
import os.path as osp
import torch
from torch.utils.data import DataLoader
import numpy as np

# 自作モジュール
from utils import io_data as id
from utils import tools
from modules import piv_modules as pm

# デバイス設定
print("\nデバイスと環境の設定を確認します...")
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
def rcc_piv(params, mode="analysis"):
    print("\nPIV解析を実行します...")
    # パラメータ取得
    io_params = params["file_io"]
    print(io_params)
    piv_params = params["rcc_piv"]
    WORKDIR_PATH = io_params["workdir_path"]

    # Datasetの作成
    ParticleImages = id.ParticleImageDataset(
        root=osp.join(WORKDIR_PATH, "images"),
        filename=piv_params["img_name"] + piv_params["img_ext"],
        ref_skip_low=piv_params["ref_skip_low"],
        ref_skip_high=piv_params["ref_skip_high"],
        pivstep_skip=piv_params["pivstep_skip"],
    )

    # 解析範囲の設定
    if piv_params["start"] == "None" and piv_params["end"] == "None":
        start = ParticleImages.start
        end = ParticleImages.end
        skip = ParticleImages.pivstep_skip
    else:
        start = piv_params["start"]
        end = piv_params["end"]
        skip = piv_params["pivstep_skip"]

    print("\n解析範囲を設定しました...")
    print(f"start: {start + 1} [frame]")
    print(f"end: {end + 1} [frame]")
    print(f"skip: {skip} [frame]")

    # piv解析のループ
    print("\nPIV解析を開始します...")
    list_sn_ratio = []
    list_error_ratio = []
    for piv_step in range(start, end + 1, skip + 1):
        print(f"\n--- piv_step: {piv_step + 1} / {end + 1} ---")

        # 解析するペア画像を取得
        if mode == "analysis":
            imgs = ParticleImages[piv_step].to(device)
        else:
            rng = np.random.default_rng()  # 乱数ジェネレータ
            piv_step = rng.integers(start, end + 1)
            imgs = ParticleImages[piv_step].to(device)

        # 変数の初期化
        rcc_num = piv_params["rcc_num"]
        displacement = None
        iw_center_old = None

        # 再帰処理ループ
        for rcc_step in range(rcc_num):
            print(f"---> rcc_step: {rcc_step + 1} / {rcc_num}")

            # 検査画像・探査画像の設定
            Rcc = pm.RecursivCorrelationPIV(
                piv_params,
                rcc_step,
                ParticleImages.img_width,
                ParticleImages.img_height,
                imgs,
            )

            # 速度点数を取得
            ny, nx = Rcc.get_dimention()
            print(f"\t速度点数 (ny, nx): ({ny}, {nx})")

            # 検査画像の取得（次元: [1, 1, ny, nx, iw_size, iw_size]）
            interrogation_imgs = Rcc.get_iw_img()
            iw_lt, iw_rb = Rcc.get_iw_lt_and_rb()
            iw_center = Rcc.get_iw_centers()

            # オフセット変位の計算
            offset = Rcc.get_offset(displacement, iw_center_old, iw_center)

            # 探査画像の取得（次元: [time, 1, ny, nx, sw_size, sw_size]）
            sw_lt, sw_rb = Rcc.get_sw_lt_and_rb(offset=offset)
            search_imgs = Rcc.get_sw_img(sw_lt)

            # 画像相関（輝度値の共分散）を計算
            matcher = pm.PatternMatch(interrogation_imgs, search_imgs)
            correlation_map = matcher.get_correlation_map()
            averaged_correlation_map = matcher.average_correlation_map(correlation_map)

            # 最大相関値の位置と値を取得
            peak_vals, peak_idx = matcher.get_max_correlation_data(
                averaged_correlation_map
            )

            if mode == "optimize":
                # sn比を計算
                sn_ratio = matcher.evaluate_sn_ratio(
                    averaged_correlation_map, mask_radius=matcher.map_w / 2
                )
                sn_ratio_mean = sn_ratio.mean().item()
                print(f"SN比の平均: {sn_ratio_mean:5e}")
                list_sn_ratio.append(sn_ratio_mean)

            # 変位を計算
            displacement = Rcc.get_displacement(peak_idx, iw_lt, sw_lt)

            # 誤ベクトル除去
            if rcc_step < rcc_num - 1:
                displacement[0, ...], _ = Rcc.correct_errors(
                    displacement[0, ...].float(),
                    error_threshold=piv_params["error_threshold"][rcc_step],
                )
                displacement[1, ...], _ = Rcc.correct_errors(
                    displacement[1, ...].float(),
                    error_threshold=piv_params["error_threshold"][rcc_step],
                )

            # 検査領域の中心座標を保存
            iw_center_old = iw_center.clone().detach()

        # サブピクセル解析
        subpixel_displacement = Rcc.get_subpixel_displacement(peak_vals)

        # サブピクセル変位を加算
        displacement = displacement + subpixel_displacement

        # 低速域用の変位を用いるフラグを取得
        mask_flag = Rcc.get_mask_flag(
            displacement, threshold=piv_params["mask_threshold"]
        )

        # 時間刻みの取得
        dt = ParticleImages.dt

        # 変位を速度に変換 [pixel/sec]
        velocity = Rcc.get_velocity(displacement, mask_flag, dt)

        # 誤ベクトルの修正
        corrected_velocity, error_flag = Rcc.correct_errors(
            velocity, error_threshold=piv_params["error_threshold"][rcc_step]
        )
        error_ratio = error_flag.float().mean()
        print(f"誤ベクトル率: {error_ratio.item()}")
        if mode == "optimize":
            list_error_ratio.append(error_ratio.item())

        # キャリブレーション&座標変換
        corrected_velocity = corrected_velocity * piv_params["pixel_to_mm"]
        corrected_velocity = torch.flip(corrected_velocity, dims=[0])
        corrected_velocity[:, :, 0] = -corrected_velocity[:, :, 0]
        iw_center = torch.flip(iw_center, dims=[0])
        iw_center[:, :, 0] = Rcc.img_height - iw_center[:, :, 0]

        if mode == "analysis":
            tools.plot_vector(
                iw_center[:, :, 1],
                iw_center[:, :, 0],
                corrected_velocity,
                output_filepath=osp.join(
                    WORKDIR_PATH, f"results/velocity/velocity_{piv_step:06d}.svg"
                ),
            )

    if mode == "optimize":
        return np.array(list_error_ratio), np.array(list_sn_ratio)


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
