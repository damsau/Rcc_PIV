from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys

# 自作モジュール
import modules.Image as Image
import modules.Param as Param
import modules.Window as Window
import modules.Match as Match
import utils.tools as tools

# グローバル変数
WORKING_DIR = Path("/home/masuda/02_experiment/20260423/0.1rps_001")
PARAMETERS_DIR = WORKING_DIR / Path("parameter")
PARAMETERS_FILE_NAME = Path("default_params.yaml")
# IMAGES_DIR = WORKING_DIR / Path("images")
# IMAGES_FILE_NAME = Path("8rps_3000_Fps150_us*.tif")
IMAGES_DIR = Path("/mnt/b/experiment/data/20260423/water/0.1rps_001")
IMAGES_FILE_NAME = Path("0.1rps_100_Fps100_us*.tif")
DATA_DIR = WORKING_DIR / Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PICS_DIR = DATA_DIR / Path("pics")
PICS_DIR.mkdir(parents=True, exist_ok=True)
VELOCITY_DIR = DATA_DIR / Path("velocity")
VELOCITY_DIR.mkdir(parents=True, exist_ok=True)


# GPUデバイス設定
print("\n■ デバイスと環境の設定を確認します")
print(f"> PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"> CUDA Available: Yes")
    print(f"> GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"> CUDA Available: No")
    print(f"> Warning: Training URAFT on CPU will be very slow.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Device: {DEVICE}")

# カラーコードの定義
BG_RED = "\033[1;101m"
BG_MAGENDA = "\033[1;105m"
BG_BLACK = "\033[1;100m"
BG_YELLOW = "\033[43m"
BG_WHITE = "\033[47m"
BG_BLUE = "\033[1;104m"
RESET = "\033[0m"


def load_images(PivImages, idx_target, idx_backref, idx_forref):
    """
    解析に使用する画像を読み込む
    """
    # ターゲット画像
    PivImages.FRAME_INDEX = idx_target
    target_image = PivImages.get_image()

    # 参照画像
    backref_images = []
    for idx in idx_backref:
        PivImages.FRAME_INDEX = idx
        image_tmp = PivImages.get_image()
        backref_images.append(image_tmp)
    backref_images = np.array(backref_images)
    forref_images = []
    for idx in idx_forref:
        PivImages.FRAME_INDEX = idx
        image_tmp = PivImages.get_image()
        forref_images.append(image_tmp)
    forref_images = np.array(forref_images)

    return (target_image, backref_images, forref_images)


def create_offset(
    PivParams, n_x_old, n_y_old, displacement, n_x_new, n_y_new, iw_posit_new
):
    """
    既知の変位からオフセットを計算する
    """

    # 次元を調整
    disp_map = displacement.reshape(n_y_old, n_x_old, 2).permute(2, 0, 1).unsqueeze(0)

    # 新しい検査領域の中心座標を計算
    iw_center_y = (iw_posit_new[:, 0] + iw_posit_new[:, 2]) / 2.0
    iw_center_x = (iw_posit_new[:, 1] + iw_posit_new[:, 3]) / 2.0

    # オフセットの座標の設定
    new_grid_x = (iw_center_x[:] / (PivParams.IMAGE_WIDTH - 1)) * 2.0 - 1.0
    new_grid_y = (iw_center_y[:] / (PivParams.IMAGE_HEIGHT - 1)) * 2.0 - 1.0
    grid = torch.stack([new_grid_x, new_grid_y], dim=-1).view(1, -1, 1, 2)

    # 補間
    sampled_disp = F.grid_sample(
        disp_map, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    print(f"sampled_dispサイズ: {sampled_disp.shape}")

    offset = sampled_disp.squeeze(3).squeeze(0).t()

    return offset


def correct_displacement(n_x, n_y, displacement, threshold=2.0):
    """
    変位の誤ベクトルを修正
    """
    # 次元調整
    disp = displacement.reshape(n_y, n_x, 2).permute(2, 0, 1).unsqueeze(0)
    print(f"disp.shapeの次元: {disp.shape}")

    # 誤ベクトル検知処理
    # 端を一列増やす
    disp_padded = F.pad(disp, (1, 1, 1, 1), mode="replicate")

    # 3x3の窓を全速度点で抽出
    disp_windows = F.unfold(disp_padded, kernel_size=3).view(1, 2, 9, n_y, n_x)

    # 中央値ベクトルを計算
    filtered_disp, _ = torch.median(disp_windows, dim=2, keepdim=True)

    # 判定対象のベクトルを抽出
    target_disp = disp_windows[:, :, 4:5, :, :]

    # target_dispと周囲ベクトルの中央値の差の絶対値を計算
    target_disp_diff_abs = torch.norm(target_disp - filtered_disp, dim=1, keepdim=True)

    # 周囲8点の変位ベクトルを抽出
    neighbor_idx = [0, 1, 2, 3, 5, 6, 7, 8]
    neighbors = disp_windows[:, :, neighbor_idx, :, :]

    # 周囲と中央値の差の絶対値の中央値を計算
    filtered_disp_diff_abs = torch.norm(neighbors - filtered_disp, dim=1, keepdim=True)
    r_m, _ = torch.median(filtered_disp_diff_abs, dim=2, keepdim=True)

    # 閾値判定
    threshold_map = target_disp_diff_abs / (r_m + 0.1)
    error_flag = (threshold_map >= threshold).reshape(1, 1, n_y, n_x)

    # 誤ベクトル修正
    # 距離の重み付き線形補完用のカーネルを定義
    kernel_weight = torch.tensor(
        [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]], device=displacement.device
    )
    kernel = kernel_weight.reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)

    # 正しいベクトルの判定
    valid_flag = (~error_flag).float().expand(1, 2, n_y, n_x)

    # 正しい変位
    disp_valid = disp * valid_flag

    # 畳込みを使って，重み付き和を計算
    disp_valid_pad = F.pad(disp_valid, (1, 1, 1, 1), mode="constant", value=0.0)
    weighted_sum = F.conv2d(disp_valid_pad, kernel, groups=2)

    # 周囲の重みの合計を計算
    valid_flag_pad = F.pad(valid_flag, (1, 1, 1, 1), mode="constant", value=0.0)
    weight_sum = F.conv2d(valid_flag_pad, kernel, groups=2)

    # 線形補完したベクトルを計算
    interpolated_disp = weighted_sum / (weight_sum + 1e-8)

    # 誤ベクトル判定された点だけ，置き換える
    error_flag = error_flag.expand(1, 1, n_y, n_x)
    correct_dips = torch.where(error_flag, interpolated_disp, disp)

    # 次元調整
    correct_dips = correct_dips.squeeze(0).permute(1, 2, 0).reshape(n_x * n_y, 2)

    print(
        f"誤ベクトル率: {torch.sum(error_flag)} / {n_x*n_y} ({torch.sum(error_flag)/(n_x*n_y)*100:.2f} %)"
    )

    return correct_dips


def estimate_subpixel_displacement(peak_vals):
    """
    変位のサブピクセル補完を計算
    """
    peak_vals_positive = torch.clamp(peak_vals, min=1e-5)  # 負の値を修正

    log_peak_vals_positive = torch.log(peak_vals_positive)  # 対数を計算

    # y方向
    Rm1_y = log_peak_vals_positive[..., 0, 1]  # 上
    R0_y = log_peak_vals_positive[..., 1, 1]  # 中心
    Rp1_y = log_peak_vals_positive[..., 2, 1]  # 下
    Rm1_x = log_peak_vals_positive[..., 1, 0]  # 左
    R0_x = log_peak_vals_positive[..., 1, 1]  # 中心
    Rp1_x = log_peak_vals_positive[..., 1, 2]  # 右

    # パーツの計算
    nom_y = Rm1_y - Rp1_y
    den_y = 2.0 * (Rm1_y - 2.0 * R0_y + Rp1_y)
    nom_x = Rm1_x - Rp1_x
    den_x = 2.0 * (Rm1_x - 2.0 * R0_x + Rp1_x)

    eps = 1e-7
    den_y = torch.where(
        torch.abs(den_y) < eps, torch.tensor(eps, device=den_y.device), den_y
    )
    den_x = torch.where(
        torch.abs(den_x) < eps, torch.tensor(eps, device=den_x.device), den_x
    )

    # サブピクセル変位の計算
    subpixel_displacement_y = nom_y / den_y
    subpixel_displacement_x = nom_x / den_x
    subpixel_displacement_y = torch.clamp(subpixel_displacement_y, -1.0, 1.0)
    subpixel_displacement_x = torch.clamp(subpixel_displacement_x, -1.0, 1.0)

    return torch.stack([subpixel_displacement_y, subpixel_displacement_x], dim=-1)


def output_grid_position(n_x, n_y, grid_position, output_filepath):
    n_y_pos, n_x_pos, _ = grid_position.shape

    if n_x_pos != n_x or n_y_pos != n_y:
        print("指定された次元とデータの次元が異なります")
        sys.exit()
    else:
        np.save(output_filepath, grid_position)


def output_velocity(n_x, n_y, velocity, output_filepath):
    """
    流速をファイル出力する
    """
    if velocity.shape[0] == 0:
        print("流速データはありません")
        return
    else:
        print("過去の流速データを出力します")
        _, n_y_vel, n_x_vel, _ = velocity.shape
        if n_x_vel != n_x or n_y_vel != n_y:
            print("指定された次元とデータの次元が異なります")
            sys.exit()
        else:
            np.save(output_filepath, velocity)


def piv_exe():
    """
    PIVを実行する
    """
    print(f"\n{BG_MAGENDA}[PIV解析の初期設定]{RESET}")

    # パラメータクラスの初期設定
    PivParams = Param.PivParams(dir=PARAMETERS_DIR, file_name=PARAMETERS_FILE_NAME)
    PivParams.DEVICE = DEVICE  # デバイス

    # 画像クラスの初期設定
    PivImages = Image.PivImages(dir=IMAGES_DIR, file_name=IMAGES_FILE_NAME)
    PivParams.IMAGE_HEIGHT = PivImages.IMAGE_HEIGHT  # 画像高さ
    PivParams.IMAGE_WIDTH = PivImages.IMAGE_WIDTH  # 画像幅

    # 窓クラスの初期設定
    PivWindow = Window.PivWindows(PivParams)
    PivParams.N_WINDOW = PivWindow.N_WINDOW  # 全窓数
    PivParams.N_X = PivWindow.N_X  # x方向の窓数
    PivParams.N_Y = PivWindow.N_Y  # y方向の窓数
    PivParams.IW_POSIT = PivWindow.IW_POSIT  # 検査窓の座標

    # 格子の位置データを出力
    grid_position = (
        PivWindow.IW_CENTER_LBO_YP[PivParams.N_RCC - 1].detach().cpu().numpy()
    )
    output_grid_position(
        PivParams.N_X[PivParams.N_RCC - 1],
        PivParams.N_Y[PivParams.N_RCC - 1],
        grid_position * PivParams.PIXEL_TO_MM,
        output_filepath=VELOCITY_DIR / Path(f"grid_position_mm.npy"),
    )

    # 相関クラスの初期設定
    PivMatch = Match.PivMatch(PivParams)
    PivParams.CM_SIZE = PivMatch.CM_SIZE

    print(f"\n{BG_MAGENDA}[PIV解析 開始]{RESET}")

    print(f"\n{BG_RED}■ 解析範囲{RESET}")
    print(
        f"{PivParams.START}から{PivParams.END + 1}まで（{PivParams.SKIP - 1}枚飛ばし）"
    )

    for idx_frame in range(PivParams.START, PivParams.END + 1, PivParams.SKIP):
        print(
            f"\n{BG_BLUE}=== 解析中のフレーム番号: {idx_frame} / {PivParams.END} ==={RESET}"
        )

        print(f"\n{BG_RED}■ 画像の読み込み{RESET}")
        # フレームインデックスの設定
        idx_target = idx_frame
        idx_list_backref = [
            idx_frame - j - 1
            for j in range(PivParams.BUFFER)
            if PivParams.START <= idx_frame - j - 1 <= PivParams.END
        ]
        # idx_list_backref.reverse()
        idx_list_forref = [
            idx_frame + j + 1
            for j in range(PivParams.BUFFER)
            if PivParams.START <= idx_frame + j + 1 <= PivParams.END
        ]
        print(f"ターゲット画像のフレーム番号: {idx_target}")
        print(f"参照画像のフレーム番号（前）: {idx_list_backref}")
        print(f"参照画像のフレーム番号（後）: {idx_list_forref}")

        # 時間刻みの設定
        dt_list_backref = [
            (idx_target - j) * PivParams.FRAMERATE ** (-1) for j in idx_list_backref
        ]
        dt_list_forref = [
            (j - idx_target) * PivParams.FRAMERATE ** (-1) for j in idx_list_forref
        ]

        # 画像読み込み
        target_image, backref_images, forref_images = load_images(
            PivImages, idx_target, idx_list_backref, idx_list_forref
        )
        target_image_gpu = torch.from_numpy(target_image).to(DEVICE)
        backref_images_gpu = torch.from_numpy(backref_images).to(DEVICE)
        forref_images_gpu = torch.from_numpy(forref_images).to(DEVICE)

        # 参照するフレームの数の取得
        n_backref_images = len(idx_list_backref)
        n_forref_images = len(idx_list_forref)
        idx_ref_max = max(n_backref_images, n_forref_images)
        print(f"過去フレーム数: {n_backref_images}")
        print(f"未来フレーム数: {n_forref_images}")

        # 流速データを保存する変数の初期化
        velocity_forref = []
        velocity_backref = []

        # 参照フレームのループ
        # targetフレームから近い順番に解析を進める
        for idx_ref in range(idx_ref_max):
            print(
                f"\n{BG_BLACK}====== 参照ループ: {idx_ref + 1} / {idx_ref_max} ======{RESET}"
            )

            # 過去フレームの解析
            if idx_ref < n_backref_images:
                print(
                    f"\n{BG_RED}■ 過去フレーム({idx_ref + 1} / {n_backref_images})の解析{RESET}"
                )

                # 再帰的処理のループ
                for idx_rcc in range(PivParams.N_RCC):
                    print(
                        f"\n{BG_BLACK}>>> 再帰ループ: {idx_rcc + 1} / {PivParams.N_RCC}{RESET}"
                    )

                    print(f"\n{BG_RED}■ 検査画像の設定{RESET}")
                    interrogation_images_gpu = PivWindow.get_interrogation_images(
                        PivParams.IW_SIZE[idx_rcc],
                        PivParams.OVERLAP[idx_rcc],
                        target_image_gpu,
                    )
                    print(
                        f"interrogation_images_gpuの次元: {interrogation_images_gpu.shape}"
                    )
                    print("検査画像を設定しました")

                    print(f"\n{BG_RED}■ オフセット設定{RESET}")
                    if idx_rcc == 0 and idx_ref == 0:
                        print("offsetを初期化します")
                        offset = None
                    elif idx_rcc == 0 and idx_ref > 0:
                        offset = create_offset(
                            PivParams,
                            PivParams.N_X[PivParams.N_RCC - 1],
                            PivParams.N_Y[PivParams.N_RCC - 1],
                            displacement,
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            PivParams.IW_POSIT[idx_rcc],
                        )
                        print("offsetを設定しました")
                    else:
                        offset = create_offset(
                            PivParams,
                            PivParams.N_X[idx_rcc - 1],
                            PivParams.N_Y[idx_rcc - 1],
                            displacement,
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            PivParams.IW_POSIT[idx_rcc],
                        )
                        print("offsetを設定しました")

                    print(f"\n{BG_RED}■ 探査画像の設定{RESET}")
                    search_images_gpu = PivWindow.get_search_images(
                        PivParams.IW_SIZE[idx_rcc],
                        PivParams.SW_SIZE[idx_rcc],
                        PivParams.N_WINDOW[idx_rcc],
                        PivParams.IW_POSIT[idx_rcc],
                        backref_images_gpu[idx_ref],
                        offset,
                    )
                    sw_posit = PivWindow.get_sw_posit()
                    print(f"search_images_gpuの次元: {search_images_gpu.shape}")
                    print(f"sw_positの次元: {sw_posit.shape}")
                    print("探査画像を設定しました")

                    print(f"\n{BG_RED}■ 画像相関計算{RESET}")
                    correlation_map = PivMatch.get_correlation_map(
                        PivParams.N_WINDOW[idx_rcc],
                        PivParams.IW_SIZE[idx_rcc],
                        PivParams.SW_SIZE[idx_rcc],
                        interrogation_images_gpu,
                        search_images_gpu,
                    )
                    print(f"correlation_mapの次元: {correlation_map.shape}")
                    # tools.show_3d_img_from_tensor(correlation_map[100, :, :])
                    print("画像相関を計算しました")
                    peak_posit, peak_vals = PivMatch.get_peak_positions_and_values(
                        PivParams.N_WINDOW[idx_rcc], correlation_map
                    )
                    print(f"peak_positの次元: {peak_posit.shape}")
                    print(f"peak_valsの次元: {peak_vals.shape}")
                    print("相関のピーク位置を取得しました")

                    print(f"\n{BG_RED}■ 変位計算{RESET}")
                    displacement_y = (
                        peak_posit[:, 0]
                        + sw_posit[:, 0]
                        - PivParams.IW_POSIT[idx_rcc][:, 0]
                    )
                    displacement_x = (
                        peak_posit[:, 1]
                        + sw_posit[:, 1]
                        - PivParams.IW_POSIT[idx_rcc][:, 1]
                    )
                    displacement = torch.stack([displacement_y, displacement_x], dim=-1)
                    print(f"displacementの次元: {displacement.shape}")
                    print("変位を計算しました")

                    print(f"\n{BG_RED}■ サブピクセル補完{RESET}")
                    subpixel_displacement = estimate_subpixel_displacement(peak_vals)
                    print(f"subpixel_displacementの次元: {subpixel_displacement.shape}")
                    print("サブピクセル変位を計算しました")
                    displacement = displacement + subpixel_displacement

                    if idx_rcc < PivParams.N_RCC - 1:
                        print(f"\n{BG_RED}■ 誤ベクトル修正{RESET}")
                        displacement = correct_displacement(
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            displacement,
                            threshold=2,
                        )

                    if idx_rcc == PivParams.N_RCC - 1:
                        print(f"\n{BG_RED}■ 誤ベクトル修正{RESET}")
                        displacement = correct_displacement(
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            displacement,
                            threshold=2,
                        )

                        print(f"\n■ 流速場計算")
                        velocity = (
                            -PivParams.PIXEL_TO_MM
                            * displacement.reshape(
                                PivParams.N_Y[idx_rcc], PivParams.N_X[idx_rcc], 2
                            )
                            / dt_list_backref[idx_ref]
                        )
                        # 左下原点，y軸正方向を上向きに
                        velocity = torch.flip(velocity, dims=[0])
                        velocity[:, :, 0] = -velocity[:, :, 0]

                        # 可視化
                        if True:
                            # if False:
                            tools.plot_vector(
                                PivParams.N_X[idx_rcc],
                                PivParams.N_Y[idx_rcc],
                                PivWindow.IW_CENTER_LBO_YP[idx_rcc]
                                * PivParams.PIXEL_TO_MM,
                                velocity,
                                output_filepath=PICS_DIR
                                / Path(
                                    f"velocity_target{idx_target}_backref{idx_list_backref[idx_ref]}.png"
                                ),
                            )

                        # numpy配列に変換して保存
                        velocity = velocity.detach().cpu().numpy()
                        velocity_backref.append(velocity)

            else:
                print(f"\n{BG_RED}■ 過去フレームの解析はしません{RESET}")

            # 未来フレームの解析
            if idx_ref < n_forref_images:
                print(
                    f"\n{BG_RED}■ 未来フレーム({idx_ref + 1} / {n_forref_images})の解析{RESET}"
                )

                # 再帰的処理のループ
                for idx_rcc in range(PivParams.N_RCC):
                    print(
                        f"\n{BG_BLACK}>>> 再帰ループ: {idx_rcc + 1} / {PivParams.N_RCC}{RESET}"
                    )

                    print(f"\n{BG_RED}■ 検査画像の設定{RESET}")
                    interrogation_images_gpu = PivWindow.get_interrogation_images(
                        PivParams.IW_SIZE[idx_rcc],
                        PivParams.OVERLAP[idx_rcc],
                        target_image_gpu,
                    )
                    print(
                        f"interrogation_images_gpuの次元: {interrogation_images_gpu.shape}"
                    )
                    print("検査画像を設定しました")

                    print(f"\n{BG_RED}■ オフセット設定{RESET}")
                    if idx_rcc == 0 and idx_ref == 0:
                        print("offsetを初期化します")
                        offset = None
                    elif idx_rcc == 0 and idx_ref > 0:
                        offset = create_offset(
                            PivParams,
                            PivParams.N_X[PivParams.N_RCC - 1],
                            PivParams.N_Y[PivParams.N_RCC - 1],
                            displacement,
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            PivParams.IW_POSIT[idx_rcc],
                        )
                        print("offsetを設定しました")
                    else:
                        offset = create_offset(
                            PivParams,
                            PivParams.N_X[idx_rcc - 1],
                            PivParams.N_Y[idx_rcc - 1],
                            displacement,
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            PivParams.IW_POSIT[idx_rcc],
                        )
                        print("offsetを設定しました")

                    print(f"\n{BG_RED}■ 探査画像の設定{RESET}")
                    search_images_gpu = PivWindow.get_search_images(
                        PivParams.IW_SIZE[idx_rcc],
                        PivParams.SW_SIZE[idx_rcc],
                        PivParams.N_WINDOW[idx_rcc],
                        PivParams.IW_POSIT[idx_rcc],
                        forref_images_gpu[idx_ref],
                        offset,
                    )
                    sw_posit = PivWindow.get_sw_posit()
                    print(f"search_images_gpuの次元: {search_images_gpu.shape}")
                    print(f"sw_positの次元: {sw_posit.shape}")
                    print("探査画像を設定しました")

                    print(f"\n{BG_RED}■ 画像相関計算{RESET}")
                    correlation_map = PivMatch.get_correlation_map(
                        PivParams.N_WINDOW[idx_rcc],
                        PivParams.IW_SIZE[idx_rcc],
                        PivParams.SW_SIZE[idx_rcc],
                        interrogation_images_gpu,
                        search_images_gpu,
                    )
                    print(f"correlation_mapの次元: {correlation_map.shape}")
                    print("画像相関を計算しました")
                    peak_posit, peak_vals = PivMatch.get_peak_positions_and_values(
                        PivParams.N_WINDOW[idx_rcc], correlation_map
                    )
                    print(f"peak_positの次元: {peak_posit.shape}")
                    print(f"peak_valsの次元: {peak_vals.shape}")
                    print("相関のピーク位置を取得しました")

                    print(f"\n{BG_RED}■ 変位計算{RESET}")
                    displacement_y = (
                        peak_posit[:, 0]
                        + sw_posit[:, 0]
                        - PivParams.IW_POSIT[idx_rcc][:, 0]
                    )
                    displacement_x = (
                        peak_posit[:, 1]
                        + sw_posit[:, 1]
                        - PivParams.IW_POSIT[idx_rcc][:, 1]
                    )
                    displacement = torch.stack([displacement_y, displacement_x], dim=-1)
                    print(f"displacementの次元: {displacement.shape}")
                    print("変位を計算しました")

                    print(f"\n■ サブピクセル補完")
                    subpixel_displacement = estimate_subpixel_displacement(peak_vals)
                    print(f"subpixel_displacementの次元: {subpixel_displacement.shape}")
                    print("サブピクセル変位を計算しました")

                    # 変位を修正
                    displacement = displacement + subpixel_displacement

                    if idx_rcc < PivParams.N_RCC - 1:
                        print(f"\n{BG_RED}■ 誤ベクトル修正{RESET}")
                        displacement = correct_displacement(
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            displacement,
                            threshold=2,
                        )

                    if idx_rcc == PivParams.N_RCC - 1:
                        print(f"\n{BG_RED}■ 誤ベクトル修正{RESET}")
                        displacement = correct_displacement(
                            PivParams.N_X[idx_rcc],
                            PivParams.N_Y[idx_rcc],
                            displacement,
                            threshold=2,
                        )

                        print(f"\n■ 流速場計算")
                        velocity = (
                            PivParams.PIXEL_TO_MM
                            * displacement.reshape(
                                PivParams.N_Y[idx_rcc], PivParams.N_X[idx_rcc], 2
                            )
                            / dt_list_forref[idx_ref]
                        )
                        # 左下原点，y軸正方向を上向きに
                        velocity = torch.flip(velocity, dims=[0])
                        velocity[:, :, 0] = -velocity[:, :, 0]

                        # 可視化
                        if True:
                            # if False:
                            tools.plot_vector(
                                PivParams.N_X[idx_rcc],
                                PivParams.N_Y[idx_rcc],
                                PivWindow.IW_CENTER_LBO_YP[idx_rcc]
                                * PivParams.PIXEL_TO_MM,
                                velocity,
                                output_filepath=PICS_DIR
                                / Path(
                                    f"velocity_target{idx_target}_forref{idx_list_forref[idx_ref]}.png"
                                ),
                            )

                        # numpy配列に変換して保存
                        velocity = velocity.detach().cpu().numpy()
                        velocity_forref.append(velocity)

            else:
                print(f"\n{BG_RED}■ 未来フレームの解析しません{RESET}")

        velocity_backref = np.array(velocity_backref)
        velocity_forref = np.array(velocity_forref)

        # 速度データを出力
        # 過去フレームの解析結果
        output_velocity(
            PivParams.N_X[PivParams.N_RCC - 1],
            PivParams.N_Y[PivParams.N_RCC - 1],
            velocity_backref,
            VELOCITY_DIR / Path(f"velocity_mmps_target{idx_target}_backref.npy"),
        )
        # 未来フレームの解析結果
        output_velocity(
            PivParams.N_X[PivParams.N_RCC - 1],
            PivParams.N_Y[PivParams.N_RCC - 1],
            velocity_forref,
            VELOCITY_DIR / Path(f"velocity_mmps_target{idx_target}_forref.npy"),
        )


def main():

    # PIV実行
    piv_exe()

    # 後処理


if __name__ == "__main__":
    main()
