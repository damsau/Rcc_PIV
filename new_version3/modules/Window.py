import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pprint as pp


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


class PivWindows:
    def __init__(self, PivParams):
        """
        検査領域クラス
        """
        print(f"再帰的相関法の階数: {PivParams.N_RCC}")

        self.IMAGE_HEIGHT = PivParams.IMAGE_HEIGHT
        self.IMAGE_WIDTH = PivParams.IMAGE_WIDTH
        self.PIXEL_TO_MM = PivParams.PIXEL_TO_MM
        self.N_RCC = PivParams.N_RCC
        self.IW_SIZE = PivParams.IW_SIZE
        self.SW_SIZE = PivParams.SW_SIZE
        self.MARGIN = PivParams.MARGIN
        self.OVERLAP = PivParams.OVERLAP
        self.DEVICE = PivParams.DEVICE

        # 各再帰ステップにおける検査領域の位置座標を計算
        self._cal_iw_position()

        # 全窓数を計算
        self.N_WINDOW = []
        for idx_rcc in range(self.N_RCC):
            n_window_tmp = self.N_Y[idx_rcc] * self.N_X[idx_rcc]
            self.N_WINDOW.append(n_window_tmp)

    def _cal_iw_position(self):
        """
        検査画像の位置座標の計算
        """
        self.IW_POSIT = []
        self.N_Y = []
        self.N_X = []
        for i_rcc in range(self.N_RCC):
            stride = int(self.IW_SIZE[i_rcc] * (1 - self.OVERLAP[i_rcc]))

            # 検査領域の左上座標のリストを作成
            iw_y_lt = torch.arange(
                0,
                self.IMAGE_HEIGHT - self.IW_SIZE[i_rcc] + 1,
                stride,
                device=self.DEVICE,
            )
            iw_x_lt = torch.arange(
                0,
                self.IMAGE_WIDTH - self.IW_SIZE[i_rcc] + 1,
                stride,
                device=self.DEVICE,
            )

            grid_y, grid_x = torch.meshgrid(iw_y_lt, iw_x_lt, indexing="ij")

            iw_y_lt = grid_y.flatten()
            iw_x_lt = grid_x.flatten()
            iw_y_rb = iw_y_lt + self.IW_SIZE[i_rcc]
            iw_x_rb = iw_x_lt + self.IW_SIZE[i_rcc]

            iw_posit_tmp = torch.stack(
                [iw_y_lt, iw_x_lt, iw_y_rb, iw_x_rb], dim=1
            )  # [左上y, 左上x, 右下y, 右下x]
            n_y_tmp, n_x_tmp = grid_y.shape

            self.IW_POSIT.append(iw_posit_tmp)
            self.N_Y.append(n_y_tmp)
            self.N_X.append(n_x_tmp)

        # 左下原点，y軸正を上方向にした時の検査領域の中心位置
        self.IW_CENTER_LBO_YP = []
        for idx_rcc in range(self.N_RCC):
            iw_posit_tmp = self.IW_POSIT[idx_rcc].reshape(
                self.N_Y[idx_rcc], self.N_X[idx_rcc], 4
            )
            iw_center_y = torch.flip(
                (
                    self.IMAGE_HEIGHT
                    - (iw_posit_tmp[:, :, 0] + iw_posit_tmp[:, :, 2]) / 2.0
                ),
                dims=[0],
            )
            iw_center_x = torch.flip(
                (iw_posit_tmp[:, :, 1] + iw_posit_tmp[:, :, 3]) / 2.0, dims=[0]
            )
            iw_center = torch.stack([iw_center_y, iw_center_x], dim=-1)
            self.IW_CENTER_LBO_YP.append(iw_center)

    def get_interrogation_images(self, iw_size, overlap, target_image):
        """
        検査画像の取得
        """
        # 検査画像のデバイス情報取得
        device = target_image.device

        # 各検査画像の間隔
        stride = int(iw_size * (1 - overlap))

        # 次元調整
        target_image_tensor = target_image.unsqueeze(0).unsqueeze(0)

        # 一括抽出
        iw_images = F.unfold(target_image_tensor, kernel_size=iw_size, stride=stride)

        # 整形
        iw_images = iw_images.squeeze(0).t().view(-1, iw_size, iw_size)

        return iw_images

    def get_search_images(
        self, iw_size, sw_size, n_window, iw_posit, reference_image, offset=None
    ):
        """
        探査画像の取得
        """
        # 検査画像の左上座標
        iw_y_lt = iw_posit[..., 0]
        iw_x_lt = iw_posit[..., 1]

        # 検査画像と探査画像を中心合わせで配置したときの余白
        margin = (sw_size - iw_size) / 2.0

        # オフセットの設定
        if offset == None:
            offset_y = 0.0
            offset_x = 0.0
        else:
            offset_y = offset[..., 0]
            offset_x = offset[..., 1]

        # 探査画像の左上座標
        sw_y_lt = iw_y_lt - margin + offset_y
        sw_x_lt = iw_x_lt - margin + offset_x

        # 探査領域サイズのローカルグリッドを作成
        local_axis_y = torch.arange(0, sw_size, device=self.DEVICE)
        local_axis_x = torch.arange(0, sw_size, device=self.DEVICE)
        local_mesh_y, local_mesh_x = torch.meshgrid(
            local_axis_y, local_axis_x, indexing="ij"
        )

        # グローバル座標に変換
        global_mesh_y = sw_y_lt[:, None, None] + local_mesh_y[None, :, :]
        global_mesh_x = sw_x_lt[:, None, None] + local_mesh_x[None, :, :]

        # grid_sample用に[-1, 1]の範囲に正規化
        norm_y = global_mesh_y / (self.IMAGE_HEIGHT - 1) * 2.0 - 1.0
        norm_x = global_mesh_x / (self.IMAGE_WIDTH - 1) * 2.0 - 1.0

        # gridの作成とメモリハック
        grid = torch.stack([norm_x, norm_y], dim=-1)
        grid = grid.view(1, n_window * sw_size, sw_size, 2)

        reference_image_tensor = reference_image.view(
            1, 1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH
        )

        # 一括抽出
        sampled_images = F.grid_sample(
            reference_image_tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        sw_images = sampled_images.view(n_window, sw_size, sw_size)

        # 探査画像の位置座標の計算
        sw_y_rb = sw_y_lt + sw_size
        sw_x_rb = sw_x_lt + sw_size

        self.sw_posit = torch.stack(
            [sw_y_lt, sw_x_lt, sw_y_rb, sw_x_rb], dim=1
        )  # [左上y, 左上x, 右下y, 右下x]

        return sw_images

    def get_sw_posit(self):
        return self.sw_posit
