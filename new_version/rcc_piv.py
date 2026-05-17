# モジュール
import yaml
import argparse
import pprint as pp
import os.path as osp
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 自作モジュール
from utils import io_data as id
from utils import tools
from modules import piv_modules as pm

FLAG_DEBUG = True


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
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.colorbar()

    plt.show()


def check_correlation_map(correlation_map, iy=0, ix=0):
    """
    相関係数の分布を３次元的に確認する関数
        correlation_map: 相関値配列（次元: [ny, nx, cm_size, cm_size]）
        iy, ix : 確認したい相関配列の位置
    """
    # 可視化する相関配列を取得
    Z = correlation_map[iy, ix].detach().cpu().numpy()

    # グリッド作成
    cm_size_y, cm_size_x = Z.shape
    x = np.arange(cm_size_x)
    y = np.arange(cm_size_y)
    X, Y = np.meshgrid(x, y)

    # 描画設定
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X, Y, Z, cmap="viridis", vmin=-1.0, vmax=1.0, edgecolor="none", alpha=0.9
    )

    # 見栄えの調整
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="correlation value")
    ax.set_title(f"Cross-Correlation Surface [ny={iy}, nx={ix}]")
    ax.set_xlabel("X Offset (pixel)")
    ax.set_ylabel("Y Offset (pixel)")
    ax.set_zlabel("Correlation")
    ax.legend()

    # 表示角度の調整 (仰角, 方位角)
    ax.view_init(elev=30, azim=45)

    # ピーク（最大値）の位置に赤いマーカーを打つ（PIV解析に便利です）
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    peak_y, peak_x = max_idx
    print(f"第1ピークの位置座標: ({peak_x}, {peak_y})")
    peak_z = Z[peak_y, peak_x]
    ax.scatter(
        peak_x, peak_y, peak_z, color="red", s=50, label=f"Peak ({peak_x}, {peak_y})"
    )

    plt.show()


def check_velocity(x_mesh, y_mesh, velocity):
    """
    速度場を確認する関数
    x_mesh, y_mesh: 速度定義点の位置座標のメッシュ
    velocity: 速度テンソル（次元: [ny, nx, 2]）
    """
    plt.style.use("PIV_results")

    x_mesh = x_mesh.detach().cpu().numpy()
    y_mesh = y_mesh.detach().cpu().numpy()
    velocity = velocity.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)

    # 軸ラベル
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # 描画範囲
    ax.set_xlim(np.min(x_mesh) - 50, np.max(x_mesh) + 50)
    ax.set_ylim(np.min(y_mesh) - 50, np.max(y_mesh) + 50)

    # 間引き間隔
    s = slice(None, None, 2)

    velocity_abs = np.sqrt(velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2)
    velocity_abs[velocity_abs == 0] = 1.0
    velocity_normalized_x = velocity[:, :, 1] / velocity_abs
    velocity_normalized_y = velocity[:, :, 0] / velocity_abs

    # 描画
    q = ax.quiver(
        x_mesh[s, s],
        y_mesh[s, s],
        velocity_normalized_x[s, s],
        velocity_normalized_y[s, s],
        velocity_abs[s, s],
        angles="xy",
        scale_units="xy",
        scale=0.02,
        pivot="mid",
        cmap=tools.CMAP_THERMAL,
        clim=(0, np.max(velocity_abs)),
    )

    # アスペクト比
    ax.set_aspect("equal")

    # カラーバーの設定
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(q, cax=cax, extend="max")
    cbar.set_label("velocity Mag.")

    fig.tight_layout()

    plt.show()
    plt.close(fig)


def check_streamline(x_mesh, y_mesh, velocity):
    """
    流線を確認する関数
    """

    plt.style.use("PIV_results")

    x_mesh = x_mesh.detach().cpu().numpy()
    y_mesh = y_mesh.detach().cpu().numpy()
    velocity = velocity.detach().cpu().numpy()

    velocity_abs = np.sqrt(velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2)
    velocity_abs[velocity_abs == 0] = 1.0
    velocity_normalized_x = velocity[:, :, 1] / velocity_abs
    velocity_normalized_y = velocity[:, :, 0] / velocity_abs

    fig, ax = plt.subplots(figsize=(4, 3), dpi=500)

    strm = ax.streamplot(
        x_mesh,
        y_mesh,
        velocity_normalized_x,
        velocity_normalized_y,
        color=velocity_abs,
        linewidth=2 * velocity_abs / velocity_abs.max(),
        cmap=tools.CMAP_THERMAL,
        density=1.5,
        arrowsize=0.5,
    )

    ax.set_title("Flow Structure (streamplot)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")

    # カラーバーの設定
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(strm.lines, cax=cax, extend="max")
    cbar.set_label("velocity Mag.")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


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


# RCC-PIV解析のメイン関数
def rcc_piv(params, mode="analysis"):
    print("\n[PIV解析 開始]")
    # \\\ パラメータ取得 \\\
    io_params = params["file_io"]
    piv_params = params["rcc_piv"]
    WORKDIR_PATH = io_params["workdir_path"]

    # \\\ Datasetの作成 \\\
    print("\n■ 粒子画像データを設定します...")
    n_buffer = piv_params["n_buffer"]
    filename = piv_params["img_name"] + piv_params["img_ext"]
    ParticleImages = id.ParticleImageDataset(
        root=osp.join(WORKDIR_PATH, "images"),
        filename=filename,
        n_buffer=n_buffer,
    )
    print(
        f"> 画像サイズ(width x height): ({ParticleImages.WIDTH} x {ParticleImages.HEIGHT}) [pixel]"
    )
    print(f"> フレームレート: {ParticleImages.FRAMERATE} [fps]")
    print(f"> フレーム数: {ParticleImages.N_FRAME}")
    # delta_t = 1 / ParticleImages.FRAMERATE
    delta_t = 1

    # \\\ 解析範囲の設定 \\\
    print("\n■ 解析範囲を設定します...")
    start = 0 if piv_params["start"] == "None" else piv_params["start"] - 1
    end = ParticleImages.N_FRAME if piv_params["end"] == "None" else piv_params["end"]
    skip = 1 if piv_params["skip"] == "None" else piv_params["skip"]
    print(f"> 開始位置: {start + 1}")
    print(f"> 終了位置: {end}")
    print(f"> 解析間隔: {skip}")

    # \\\ メインループ \\\
    print("\n■ PIV解析を開始します...")
    # list_sn_ratio = []  # SN比
    # list_error_ratio = []  # 誤ベクトル率
    # list_dynamic_range = []  # ダイナミックレンジ
    # list_spatial_variations = []  # 速度場の滑らかさ
    if mode == "analysis":
        for i_frame in range(start, end, skip):
            print(f"\n=== 解析中のフレーム番号: {i_frame + 1} / {end} ===")

            # \\\ 解析するペア画像を取得 \\\
            print(f"\n> 画像配列を取得...")
            list_frame_number, images = ParticleImages.get_images(i_frame=i_frame)
            print(f"読み込んだフレーム番号: {list_frame_number}")

            if (
                min(list_frame_number) < 0
                or max(list_frame_number) > ParticleImages.N_FRAME - 1
            ):
                print("!!! 解析に使用するフレームが足りないのでスキップします !!!")
                continue
            print(f"\n> 画像配列を{device}に転送...")
            images_gpu = images.to(device)  # 画像配列をGPUに送信

            template_image_gpu = images_gpu[
                int(n_buffer / 2) : int(n_buffer / 2) + 1, ...
            ]
            forward_images_gpu = images_gpu[int(n_buffer / 2) + 1 :, ...]
            backward_images_gpu = torch.flip(
                images_gpu[: int(n_buffer / 2), ...], dims=[0]
            )

            if FLAG_DEBUG:
                check_variable("images_gpu", images_gpu)
                check_variable("template_image_gpu", template_image_gpu)
                check_variable("forward_images_gpu", forward_images_gpu)
                check_variable("backward_images_gpu", backward_images_gpu)

            # \\\ 変数の初期化 \\\
            n_rcc = piv_params["n_rcc"]  # 再帰的処理回数
            #     displacement = None
            #     iw_center_old = None

            # \\\ 再帰ループ \\\
            for i_rcc in range(n_rcc):
                print(f"> 再帰処理 {i_rcc + 1}回目 / {n_rcc} 回")

                # \\\ 検査画像の設定 \\\
                InterrogationWindow = pm.InterrogationWindowSetting(
                    image_width=ParticleImages.WIDTH,
                    image_height=ParticleImages.HEIGHT,
                    image=template_image_gpu,
                    iw_size=piv_params["iw_size"][i_rcc],
                    margin=piv_params["margin"][i_rcc],
                    overlap=piv_params["overlap"][i_rcc],
                    device=device,
                )

                interrogation_image_gpu = (
                    InterrogationWindow.get_image()
                )  # 検査画像の取得

                nx, ny = InterrogationWindow.dimension()  # 速度定義点の数を取得
                print(f"> 速度定義点の数: (nx, ny) = ({nx}, {ny})")

                iw_position_lt_gpu, iw_position_rb_gpu = (
                    InterrogationWindow.get_iw_position()
                )  # 検査画像のピクセル位置の取得
                iw_position_center_gpu = (iw_position_lt_gpu + iw_position_rb_gpu) / 2
                if FLAG_DEBUG:
                    check_variable("iw_position_lt_gpu", iw_position_lt_gpu)

                # /// 探査画像の設定 ///
                SearchWindow = pm.SearchWindowSetting(
                    image_width=ParticleImages.WIDTH,
                    image_height=ParticleImages.HEIGHT,
                    image=forward_images_gpu[0:1, ...],
                    # image=backward_images_gpu[0:1, ...],
                    iw_position_lt=iw_position_lt_gpu,
                    iw_size=piv_params["iw_size"][i_rcc],
                    sw_size=piv_params["sw_size"][i_rcc],
                    device=device,
                )

                sw_position_lt_gpu, sw_position_rb_gpu = (
                    SearchWindow.get_sw_position()
                )  # 探査画像のピクセル位置の取得
                sw_position_center_gpu = (sw_position_lt_gpu + sw_position_rb_gpu) / 2
                if FLAG_DEBUG:
                    check_variable("sw_position_lt_gpu", sw_position_lt_gpu)

                search_image_gpu = SearchWindow.get_sw_image()

                # パターンマッチングの設定
                PatternMatch = pm.PatternMatchSetting(
                    nx=nx,
                    ny=ny,
                    iw_size=piv_params["iw_size"][i_rcc],
                    interrogation_image=interrogation_image_gpu,
                    sw_size=piv_params["sw_size"][i_rcc],
                    search_image=search_image_gpu,
                    device=device,
                )

                # 相関配列の取得
                correlation_map_gpu = PatternMatch.get_correlation_map()

                peak_value, peak_position = PatternMatch.get_peak_position()
                if FLAG_DEBUG:
                    check_variable("peak_position", peak_position)

                # 変位計算
                pixel_displacement = (
                    peak_position + sw_position_lt_gpu - iw_position_lt_gpu
                )

                # 速度計算
                pixel_velocity = pixel_displacement / delta_t

                pixel_velocity_flipped = torch.flip(pixel_velocity, dims=[0])
                pixel_velocity_flipped[:, :, 0] = -pixel_velocity_flipped[:, :, 0]
                iw_position_center_gpu_flipped = torch.flip(
                    iw_position_center_gpu, dims=[0]
                )
                iw_position_center_gpu_flipped[:, :, 0] = (
                    ParticleImages.HEIGHT - iw_position_center_gpu_flipped[:, :, 0]
                )

                if FLAG_DEBUG:
                    check_velocity(
                        iw_position_center_gpu_flipped[..., 1],
                        iw_position_center_gpu_flipped[..., 0],
                        pixel_velocity_flipped,
                    )
                    check_streamline(
                        iw_position_center_gpu_flipped[..., 1],
                        iw_position_center_gpu_flipped[..., 0],
                        pixel_velocity_flipped,
                    )

                # if FLAG_DEBUG:
                #     check_variable("self.peak_position", peak_position)
                #     check_variable("self.peak_value", peak_value)

    #         Rcc = pm.RecursivCorrelationPIV(
    #             piv_params,
    #             rcc_step,
    #             ParticleImages.img_width,
    #             ParticleImages.img_height,
    #             imgs,
    #         )

    #         # 速度点数を取得
    #         ny, nx = Rcc.get_dimention()
    #         print(f"\t速度点数 (ny, nx): ({ny}, {nx})")

    #         # 検査画像の取得（次元: [1, 1, ny, nx, iw_size, iw_size]）
    #         interrogation_imgs = Rcc.get_iw_img()
    #         iw_lt, iw_rb = Rcc.get_iw_lt_and_rb()
    #         iw_center = Rcc.get_iw_centers()

    #         # オフセット変位の計算
    #         offset = Rcc.get_offset(displacement, iw_center_old, iw_center)

    #         # 探査画像の取得（次元: [time, 1, ny, nx, sw_size, sw_size]）
    #         sw_lt, sw_rb = Rcc.get_sw_lt_and_rb(offset=offset)
    #         search_imgs = Rcc.get_sw_img(sw_lt)

    #         # 画像相関（輝度値の共分散）を計算
    #         matcher = pm.PatternMatch(interrogation_imgs, search_imgs)
    #         correlation_map = matcher.get_correlation_map()
    #         averaged_correlation_map = matcher.average_correlation_map(correlation_map)

    #         # 最大相関値の位置と値を取得
    #         peak_vals, peak_idx = matcher.get_max_correlation_data(
    #             averaged_correlation_map
    #         )

    #         if mode == "optimize":
    #             # sn比を計算
    #             sn_ratio = matcher.evaluate_sn_ratio(
    #                 averaged_correlation_map, mask_radius=matcher.map_w / 2
    #             )
    #             sn_ratio_mean = sn_ratio.mean().item()
    #             print(f"SN比の平均: {sn_ratio_mean:5e}")
    #             list_sn_ratio.append(sn_ratio_mean)

    #         # 変位を計算
    #         displacement = Rcc.get_displacement(peak_idx, iw_lt, sw_lt)

    #         # 誤ベクトル除去
    #         if rcc_step < rcc_num - 1:
    #             displacement[0, ...], _ = Rcc.correct_errors(
    #                 displacement[0, ...].float(),
    #                 error_threshold=piv_params["error_threshold"][rcc_step],
    #             )
    #             displacement[1, ...], _ = Rcc.correct_errors(
    #                 displacement[1, ...].float(),
    #                 error_threshold=piv_params["error_threshold"][rcc_step],
    #             )

    #         # 検査領域の中心座標を保存
    #         iw_center_old = iw_center.clone().detach()

    #     # サブピクセル解析
    #     subpixel_displacement = Rcc.get_subpixel_displacement(peak_vals)

    #     # サブピクセル変位を加算
    #     displacement = displacement + subpixel_displacement

    #     # 低速域用の変位を用いるフラグを取得
    #     mask_flag = Rcc.get_mask_flag(
    #         displacement, threshold=piv_params["mask_threshold"]
    #     )

    #     # 時間刻みの取得
    #     dt = ParticleImages.dt

    #     # 変位を速度に変換 [pixel/sec]
    #     velocity = Rcc.get_velocity(displacement, mask_flag, dt)

    #     # 誤ベクトルの修正
    #     corrected_velocity, error_flag = Rcc.correct_errors(
    #         velocity, error_threshold=piv_params["error_threshold"][rcc_step]
    #     )
    #     error_ratio = error_flag.float().mean()
    #     print(f"誤ベクトル率: {error_ratio.item()}")
    #     if mode == "optimize":
    #         list_error_ratio.append(error_ratio.item())

    #     # キャリブレーション&座標変換
    #     corrected_velocity = corrected_velocity * piv_params["pixel_to_mm"]
    #     corrected_velocity = torch.flip(corrected_velocity, dims=[0])
    #     corrected_velocity[:, :, 0] = -corrected_velocity[:, :, 0]
    #     iw_center = torch.flip(iw_center, dims=[0])
    #     iw_center[:, :, 0] = Rcc.img_height - iw_center[:, :, 0]

    #     if mode == "analysis":
    #         tools.plot_vector(
    #             iw_center[:, :, 1],
    #             iw_center[:, :, 0],
    #             corrected_velocity,
    #             output_filepath=osp.join(
    #                 WORKDIR_PATH, f"results/velocity/velocity_{piv_step:06d}.svg"
    #             ),
    #         )

    #     if mode == "optimize":
    #         # 速度のダイナミックレンジを評価
    #         vel_x = corrected_velocity[0]
    #         vel_y = corrected_velocity[1]
    #         mag = torch.sqrt(vel_x**2 + vel_y**2)
    #         max_vel = torch.quantile(mag, 0.9)  # 外れ値を避けるため95パーセンタイル
    #         min_vel = torch.quantile(mag, 0.1)  # ノイズフロア

    #         # 速度のダイナミックレンジの近似
    #         dynamic_range_score = (max_vel / (min_vel + 1e-6)).item()
    #         list_dynamic_range.append(dynamic_range_score)

    #         # 隣接画素との差分（x方向とy方向）
    #         diff_u_x = torch.diff(vel_x, dim=1)
    #         diff_u_y = torch.diff(vel_x, dim=0)
    #         diff_v_x = torch.diff(vel_y, dim=1)
    #         diff_v_y = torch.diff(vel_y, dim=0)

    #         # 空間的な変動（滑らかさ）の指標：小さいほど良い
    #         spatial_fluctuation = (
    #             (
    #                 diff_u_x.pow(2).mean()
    #                 + diff_u_y.pow(2).mean()
    #                 + diff_v_x.pow(2).mean()
    #                 + diff_v_y.pow(2).mean()
    #             )
    #             .sqrt()
    #             .item()
    #         )

    #         list_spatial_variations.append(spatial_fluctuation)

    # if mode == "optimize":

    #     return (
    #         np.array(list_error_ratio),
    #         np.array(list_sn_ratio),
    #         np.array(list_dynamic_range),
    #         np.array(list_spatial_variations),
    #     )


if __name__ == "__main__":
    # パラメータの読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", type=str, default="-")
    args = parser.parse_args()
    with open(args.params_file, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        print("\n■ パラメータを読み込みます...")
        print("> 解析パラメータ:")
        pp.pprint(params, sort_dicts=False, indent=4)

    # piv解析実行
    rcc_piv(params)
