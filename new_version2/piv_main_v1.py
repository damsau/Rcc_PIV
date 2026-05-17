from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# 自作モジュール
import modules.Image as Image
import modules.Param as Param
import modules.Window as Window
import modules.Match as Match
import utils.tools as tools

# グローバル変数
WORKING_DIR = Path("/home/masuda/02_experiment/20260423/water_0.1rps")
PARAMETERS_DIR = WORKING_DIR / Path("parameters")
PARAMETERS_FILE_NAME = Path("default_params.yaml")
IMAGES_DIR = WORKING_DIR / Path("images/0.1rps_001")
IMAGES_FILE_NAME = Path("0.1rps_100_Fps100_us*.tif")

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


def piv_exe():
    """
    PIVを実行する
    """
    print("\n[PIV解析の初期設定]")

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

    # 相関クラスの初期設定
    PivMatch = Match.PivMatch(PivParams)
    PivParams.CM_SIZE = PivMatch.CM_SIZE

    print("\n[PIV解析 開始]")

    print("\n■ 解析範囲")
    print(
        f"{PivParams.START}から{PivParams.END + 1}まで（{PivParams.SKIP - 1}枚飛ばし）"
    )

    for idx_frame in range(PivParams.START, PivParams.END + 1, PivParams.SKIP):
        print(f"\n=== 解析中のフレーム番号: {idx_frame} / {PivParams.END} ===")

        print("\n■ 画像の読み込み")
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
        print(f"過去フレーム数: {n_backref_images}")
        print(f"未来フレーム数: {n_forref_images}")

        # オフセットの初期化
        backref_offset = None
        forref_offset = None

        for idx_rcc in range(PivParams.N_RCC):
            print(f"\n>>再帰ループ: {idx_rcc + 1} / {PivParams.N_RCC}")

            print("\n■ 検査画像の設定")
            interrogation_images_gpu = PivWindow.get_interrogation_images(
                PivParams.IW_SIZE[idx_rcc],
                PivParams.OVERLAP[idx_rcc],
                target_image_gpu,
            )
            print(f"interrogation_images_gpuサイズ: {interrogation_images_gpu.shape}")
            print(f"iw_positサイズ: {PivParams.IW_POSIT[idx_rcc].shape}")

            print("\n■ 探査画像の設定")
            # 過去フレームの探査画像
            backref_search_images_gpu = []
            backref_sw_posit = []
            for idx_backref in range(n_backref_images):
                search_images_gpu_tmp = PivWindow.get_search_images(
                    PivParams.IW_SIZE[idx_rcc],
                    PivParams.SW_SIZE[idx_rcc],
                    PivParams.N_WINDOW[idx_rcc],
                    PivParams.IW_POSIT[idx_rcc],
                    backref_images_gpu[idx_backref],
                    (
                        backref_offset
                        if backref_offset == None
                        else backref_offset[idx_backref]
                    ),
                )
                backref_search_images_gpu.append(search_images_gpu_tmp)
                backref_sw_posit_tmp = PivWindow.get_sw_posit()
                backref_sw_posit.append(backref_sw_posit_tmp)
            if n_backref_images == 0:
                backref_search_images_gpu = torch.tensor([])
                backref_sw_posit = torch.tensor([])
            else:
                backref_search_images_gpu = torch.stack(backref_search_images_gpu)
                backref_sw_posit = torch.stack(backref_sw_posit)
                print(
                    f"backref_search_images_gpuサイズ: {backref_search_images_gpu.shape}"
                )
                print(f"backref_sw_positサイズ: {backref_sw_posit.shape}")

            # 未来フレームの探査画像
            forref_search_images_gpu = []
            forref_sw_posit = []
            for idx_forref in range(n_forref_images):
                search_images_gpu_tmp = PivWindow.get_search_images(
                    PivParams.IW_SIZE[idx_rcc],
                    PivParams.SW_SIZE[idx_rcc],
                    PivParams.N_WINDOW[idx_rcc],
                    PivParams.IW_POSIT[idx_rcc],
                    forref_images_gpu[idx_forref],
                    (
                        forref_offset
                        if forref_offset == None
                        else forref_offset[idx_forref]
                    ),
                )
                forref_search_images_gpu.append(search_images_gpu_tmp)
                forref_sw_posit_tmp = PivWindow.get_sw_posit()
                forref_sw_posit.append(forref_sw_posit_tmp)
            if n_forref_images == 0:
                forref_search_images_gpu = torch.tensor([])
                forref_sw_posit = torch.tensor([])
            else:
                forref_search_images_gpu = torch.stack(forref_search_images_gpu)
                forref_sw_posit = torch.stack(forref_sw_posit)
                print(
                    f"forref_search_images_gpuサイズ: {forref_search_images_gpu.shape}"
                )
                print(f"forref_sw_positサイズ: {forref_sw_posit.shape}")

            print("\n■ 画像相関を計算")
            # 過去フレームとの相関
            backref_correlation_map = []
            for idx_backref in range(n_backref_images):
                correlation_map_tmp = PivMatch.get_correlation_map(
                    PivParams.N_WINDOW[idx_rcc],
                    PivParams.IW_SIZE[idx_rcc],
                    PivParams.SW_SIZE[idx_rcc],
                    interrogation_images_gpu,
                    backref_search_images_gpu[idx_backref],
                )
                backref_correlation_map.append(correlation_map_tmp)
            if n_backref_images == 0:
                backref_correlation_map = torch.tensor([])
            else:
                backref_correlation_map = torch.stack(backref_correlation_map)
                print(f"backref_correlation_mapサイズ: {backref_correlation_map.shape}")

            # 未来フレームとの相関
            forref_correlation_map = []
            for idx_forref in range(n_forref_images):
                correlation_map_tmp = PivMatch.get_correlation_map(
                    PivParams.N_WINDOW[idx_rcc],
                    PivParams.IW_SIZE[idx_rcc],
                    PivParams.SW_SIZE[idx_rcc],
                    interrogation_images_gpu,
                    forref_search_images_gpu[idx_forref],
                )
                forref_correlation_map.append(correlation_map_tmp)
            if n_forref_images == 0:
                forref_correlation_map = torch.tensor([])
            else:
                forref_correlation_map = torch.stack(forref_correlation_map)
                print(f"forref_correlation_mapサイズ: {forref_correlation_map.shape}")

            print("\n■ 変位を計算")
            # 過去フレームとの相関のピーク座標を取得
            backref_peak_posit = []
            for idx_backref in range(n_backref_images):
                peak_posit_tmp = PivMatch.get_peak_position(
                    PivParams.N_WINDOW[idx_rcc], backref_correlation_map[idx_backref]
                )
                backref_peak_posit.append(peak_posit_tmp)
            if n_backref_images == 0:
                backref_peak_posit = torch.tensor([])
            else:
                backref_peak_posit = torch.stack(backref_peak_posit)
                print(f"backref_peak_positサイズ: {backref_peak_posit.shape}")

            # 未来フレームとの相関のピーク座標を取得
            forref_peak_posit = []
            for idx_forref in range(n_forref_images):
                peak_posit_tmp = PivMatch.get_peak_position(
                    PivParams.N_WINDOW[idx_rcc], forref_correlation_map[idx_forref]
                )
                forref_peak_posit.append(peak_posit_tmp)
            if n_forref_images == 0:
                forref_peak_posit = torch.tensor([])
            else:
                forref_peak_posit = torch.stack(forref_peak_posit)
                print(f"forref_peak_positサイズ: {forref_peak_posit.shape}")

            # 過去フレームとの変位場
            backref_displacement_y = []
            backref_displacement_x = []
            for idx_backref in range(n_backref_images):
                displacement_y_tmp = (
                    backref_peak_posit[idx_backref, :, 0]
                    + backref_sw_posit[idx_backref, :, 0]
                    - PivParams.IW_POSIT[idx_rcc][:, 0]
                )

                displacement_x_tmp = (
                    backref_peak_posit[idx_backref, :, 1]
                    + backref_sw_posit[idx_backref, :, 1]
                    - PivParams.IW_POSIT[idx_rcc][:, 1]
                )

                backref_displacement_y.append(displacement_y_tmp)
                backref_displacement_x.append(displacement_x_tmp)
            if n_backref_images == 0:
                backref_displacement_y = torch.tensor([])
                backref_displacement_x = torch.tensor([])
            else:
                backref_displacement_y = torch.stack(backref_displacement_y)
                backref_displacement_x = torch.stack(backref_displacement_x)
            backref_displacement = torch.stack(
                [backref_displacement_y, backref_displacement_x], dim=-1
            )
            print(f"backref_displacementサイズ: {backref_displacement.shape}")

            # 未来フレームとの変位場
            forref_displacement_y = []
            forref_displacement_x = []
            for idx_forref in range(n_forref_images):
                displacement_y_tmp = (
                    forref_peak_posit[idx_forref, :, 0]
                    + forref_sw_posit[idx_forref, :, 0]
                    - PivParams.IW_POSIT[idx_rcc][:, 0]
                )

                displacement_x_tmp = (
                    forref_peak_posit[idx_forref, :, 1]
                    + forref_sw_posit[idx_forref, :, 1]
                    - PivParams.IW_POSIT[idx_rcc][:, 1]
                )

                forref_displacement_y.append(displacement_y_tmp)
                forref_displacement_x.append(displacement_x_tmp)
            if n_forref_images == 0:
                forref_displacement_y = torch.tensor([])
                forref_displacement_x = torch.tensor([])
            else:
                forref_displacement_y = torch.stack(forref_displacement_y)
                forref_displacement_x = torch.stack(forref_displacement_x)
            forref_displacement = torch.stack(
                [forref_displacement_y, forref_displacement_x], dim=-1
            )
            print(f"forref_displacementサイズ: {forref_displacement.shape}")

            if idx_rcc < PivParams.N_RCC - 1:
                print("\n■ オフセット計算")
                # 過去フレームでのオフセット計算
                backref_offset = []
                for idx_backref in range(n_backref_images):
                    offset_tmp = create_offset(
                        PivParams,
                        PivParams.N_X[idx_rcc],
                        PivParams.N_Y[idx_rcc],
                        backref_displacement[idx_backref],
                        PivParams.N_X[idx_rcc + 1],
                        PivParams.N_Y[idx_rcc + 1],
                        PivParams.IW_POSIT[idx_rcc + 1],
                    )
                    backref_offset.append(offset_tmp)
                if n_backref_images == 0:
                    backref_offset = torch.tensor([])
                else:
                    backref_offset = torch.stack(backref_offset)
                print(f"backref_offsetサイズ: {backref_offset.shape}")

                # 未来フレームでのオフセット計算
                forref_offset = []
                for idx_forref in range(n_forref_images):
                    offset_tmp = create_offset(
                        PivParams,
                        PivParams.N_X[idx_rcc],
                        PivParams.N_Y[idx_rcc],
                        forref_displacement[idx_forref],
                        PivParams.N_X[idx_rcc + 1],
                        PivParams.N_Y[idx_rcc + 1],
                        PivParams.IW_POSIT[idx_rcc + 1],
                    )
                    forref_offset.append(offset_tmp)
                if n_forref_images == 0:
                    forref_offset = torch.tensor([])
                else:
                    forref_offset = torch.stack(forref_offset)
                print(f"forref_offsetサイズ: {forref_offset.shape}")

                print("\n■ 速度ベクトルの取得")

            # 過去フレーム
            # for idx_backref in range(n_backref_images):
            #     tools.plot_vector(
            #         PivParams.N_X[idx_rcc],
            #         PivParams.N_Y[idx_rcc],
            #         PivParams.IW_POSIT[idx_rcc],
            #         backref_displacement[idx_backref],
            #     )

            # 未来フレーム
            # for idx_forref in range(n_forref_images):
            #     tools.plot_vector(
            #         PivParams.N_X[idx_rcc],
            #         PivParams.N_Y[idx_rcc],
            #         PivParams.IW_POSIT[idx_rcc],
            #         forref_displacement[idx_forref],
            #     )


def main():

    # PIV実行
    piv_exe()

    # 後処理


if __name__ == "__main__":
    main()
