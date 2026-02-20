import torch
import torch.nn.functional as F


class WindowSetup:
    def __init__(self, piv_params, rcc_step, img_width, img_height, imgs):
        """
        piv_params: PIV解析パラメータ
        img_width: 画像の幅
        img_height: 画像の高さ
        imgs: 解析画像
        """

        # パラメータ設定
        self.imgs_num = imgs.shape[0]  # 解析に使う画像の数
        self.iw_size = piv_params["iw_size"][rcc_step]  # 検査領域サイズ
        self.sw_size = piv_params["sw_size"][rcc_step]  # 探査領域サイズ
        self.margin_size = piv_params["margin_size"][rcc_step]  # 余白領域サイズ
        self.overlap_ratio = piv_params["overlap_ratio"][rcc_step]  # オーバーラップ率
        self.stride = int(self.iw_size * (1 - self.overlap_ratio))
        self.img_width = img_width
        self.img_height = img_height

        # 比較元画像
        self.template_img = imgs[2:3, 0:1, :, :]

        # 比較先画像
        self.reference_imgs = imgs[[0, 1, 3, 4], 0:1, :, :]
        self.ref_imgs_num = self.imgs_num - 1

        # 検査画像（interrogation_img）を生成
        self.interrogation_img = (
            (
                self.template_img[
                    :,
                    :,
                    self.margin_size : self.img_height - self.margin_size,
                    self.margin_size : self.img_width - self.margin_size,
                ]
                .unfold(2, self.iw_size, self.stride)
                .unfold(3, self.iw_size, self.stride)
            )
            .permute(2, 3, 1, 4, 5, 0)
            .squeeze(-1)
        )
        self.device = self.interrogation_img.device

        # 速度点数
        self.ny = self.interrogation_img.shape[0]
        self.nx = self.interrogation_img.shape[1]

    def get_dimention(self):
        """
        速度点数を取得
        """
        return self.ny, self.nx

    def get_iw_img(self):
        """
        検査画像を取得
        """
        return self.interrogation_img  # 次元: [ny, nx, 1, iw_size, iw_size]

    def get_iw_lt_and_rb(self):
        """
        検査画像の左上，右下の座標の取得
        """
        # インデックスグリッドの生成（ny, nx）
        jj = torch.arange(self.ny, device=self.device)
        ii = torch.arange(self.nx, device=self.device)
        grid_j, grid_i = torch.meshgrid(jj, ii, indexing="ij")

        # 左上座標の計算
        lt_y = self.margin_size + grid_j * self.stride
        lt_x = self.margin_size + grid_i * self.stride

        # 右下座標の計算
        rb_y = lt_y + self.iw_size
        rb_x = lt_x + self.iw_size

        iw_lt = torch.stack([lt_y, lt_x], dim=-1)
        iw_rb = torch.stack([rb_y, rb_x], dim=-1)

        return iw_lt, iw_rb

    def get_iw_centers(self):
        """
        検査画像の中心座標の取得
        """
        iw_lt, _ = self.get_iw_lt_and_rb()
        centers = iw_lt + self.iw_size / 2
        return centers

    def get_sw_lt_and_rb(self, offset=None):
        """
        探査画像の左上，右下の座標の取得
        offset: 次元[time, ny, nx, 2]のテンソル，Noneの場合は0
        """
        # 検査領域の左上の座標を取得
        iw_lt, _ = self.get_iw_lt_and_rb()
        iw_lt_y = iw_lt[..., 0]
        iw_lt_x = iw_lt[..., 1]

        # 検査領域と探査領域の中心を合わせるためのシフト量
        shift_y = (self.sw_size - self.iw_size) // 2
        shift_x = (self.sw_size - self.iw_size) // 2

        # オフセットの適用
        if offset is None:
            offset = torch.zeros(
                (
                    self.ref_imgs_num,
                    self.ny,
                    self.nx,
                    2,
                ),
                device=self.device,
            )
        sw_lt_y = iw_lt_y - shift_y + offset[..., 0]
        sw_lt_x = iw_lt_x - shift_x + offset[..., 1]

        # 整数値に丸める
        sw_lt_y = torch.round(sw_lt_y).long()
        sw_lt_x = torch.round(sw_lt_x).long()

        # 次元[ny, nx, 2]にスタック
        sw_lt = torch.stack([sw_lt_y, sw_lt_x], dim=-1)
        sw_rb = sw_lt + self.sw_size

        return sw_lt, sw_rb  # 次元: [ref_imgs_num, ny, nx, 2]

    def get_sw_img(self, sw_lt):
        """
        探査画像を取得
        sw_lt: 探査画像の左上の座標
        """
        n = self.ny * self.nx

        # 境界外アクセスを防ぐためのクランプ処理
        h_max, w_max = self.img_height, self.img_height
        sw_lt_y = (
            torch.clamp(sw_lt[..., 0], 0, h_max - self.sw_size)
            .long()
            .reshape(self.ref_imgs_num, n, 1, 1)
        )
        sw_lt_x = (
            torch.clamp(sw_lt[..., 1], 0, w_max - self.sw_size)
            .long()
            .reshape(self.ref_imgs_num, n, 1, 1)
        )

        # 各窓内の相対座標グリッドを作成（0, 1, 2, ..., sw_size - 1）
        rel_y = torch.arange(self.sw_size, device=self.device)
        rel_x = torch.arange(self.sw_size, device=self.device)
        rel_grid_y, rel_grid_x = torch.meshgrid(rel_y, rel_x, indexing="ij")
        rel_grid_y = rel_grid_y.reshape(1, 1, self.sw_size, self.sw_size)
        rel_grid_x = rel_grid_x.reshape(1, 1, self.sw_size, self.sw_size)

        # 全窓（n） x 窓内全ピクセル（sw*sw）の絶対座標を計算
        abs_y = sw_lt_y + rel_grid_y
        abs_x = sw_lt_x + rel_grid_x

        # 2次元座標を1次元インデックスに変換（idx = y*W + x）
        flat_idx = (abs_y * w_max + abs_x).reshape(self.ref_imgs_num, -1)

        # 一括サンプリング
        # flat_frame: [H*W], flat_idx: [n, sw, sw]
        # 戻り値: [n, sw, sw]
        reference_imgs_flat = self.reference_imgs.reshape(self.ref_imgs_num, -1)
        search_imgs = torch.gather(reference_imgs_flat, dim=1, index=flat_idx)

        return search_imgs.reshape(
            self.ref_imgs_num, self.ny, self.nx, 1, self.sw_size, self.sw_size
        )  # 次元: [ref_imgs_num, ny, nx, 1, iw_size, iw_size]

    def get_displacement(self, peak_idx, iw_lt, sw_lt):
        """
        pixel単位の変位を計算する
        peak_idx: 探査画像に対して相関が最大となる位置 [ny, nx, 2]
        iw_lt: 検査画像の左上のインデックス [ny, nx, 2]
        sw_lt: 探査画像の左上のインデックス [ref_imgs_num, ny, nx, 2]
        """
        displacement = peak_idx + sw_lt[int(self.ref_imgs_num / 2) :] - iw_lt

        return displacement

    def get_offset(self, displacement, iw_center_old, iw_center_new):
        """
        探査領域のオフセット変位を計算
        displacement: 前rcc_stepで計算した変位 [2, ny, nx, 2] （ny, nxは前rcc_stepの大きさ）
        """
        if displacement is None and iw_center_old is None:
            return None
        else:
            return 0


class PatternMatch:
    def __init__(self, interrogation_imgs, search_imgs):
        """
        画像相関を輝度値の共分散を用いて計算する
        interrogation_img: 検査画像群 [ny, nx, 1, iw_size, iw_size]
        search_imgs: 探査画像群 [ref_imgs_num, ny, nx, 1, sw_size, sw_size]
        """
        self.ref_imgs_num, self.ny, self.nx, self.channel_num, self.sw_size, _ = (
            search_imgs.shape
        )
        _, _, _, self.iw_size, _ = interrogation_imgs.shape
        self.n = self.ny * self.nx

        # 検査画像の輝度値の中心化
        interrogation_imgs_mean = torch.mean(
            interrogation_imgs, dim=(-2, -1), keepdim=True
        )
        interrogation_imgs_centered = (
            interrogation_imgs - interrogation_imgs_mean
        ).reshape(self.n, 1, self.iw_size, self.iw_size)

        # 探査画像の１次元化
        self.search_imgs_flat = search_imgs.reshape(
            self.ref_imgs_num * self.n, 1, self.sw_size, self.sw_size
        )

        # 共分散の計算（等価なE[interrogation_imgs_centered*search_imgs]を計算）
        # [ref_imgs_num*n, 1, sw_size, sw_size]と[n, 1, iw_size, iw_size]の相関を計算
        self.interrogation_imgs_centered = interrogation_imgs_centered.repeat(
            self.ref_imgs_num, 1, 1, 1
        )  # 次元をsearch_imgs_flatに合わせる

    def get_correlation_map(self):
        """
        相関配列を計算
        """
        # 入力テンソルxと重みテンソルwの設定
        x = self.search_imgs_flat.permute(1, 0, 2, 3)
        w = self.interrogation_imgs_centered

        correlation_map = F.conv2d(
            x,
            w,
            groups=self.ref_imgs_num * self.n,
        ) / (self.iw_size * self.iw_size)
        self.map_h, self.map_w = correlation_map.shape[-2:]
        return correlation_map.reshape(
            self.ref_imgs_num, self.ny, self.nx, self.map_h, self.map_w
        )

    def average_correlation_map(self, correlation_map):
        """
        相関配列を時間方向に平均
        """
        # 負の時間方向の相関マップのみを反転
        flip_idx = [0, 1]
        correlation_map[flip_idx] = torch.flip(correlation_map[flip_idx], dims=(-2, -1))

        # それぞれの時間刻み幅の相関配列を足す
        averaged_correlation_map_low = (
            correlation_map[0, ...] + correlation_map[3, ...]
        )  # 低速域用
        averaged_correlation_map_high = (
            correlation_map[1, ...] + correlation_map[2, ...]
        )  # 高速域用

        return torch.stack(
            [averaged_correlation_map_high, averaged_correlation_map_low], dim=0
        )

    def get_max_correlation_data(self, averaged_correlation_map):
        """
        平均相関配列から最大のとその周囲8点の相関値を取得
        averaged_correlation_map: 平均相関配列 [2, ny, nx, 1, map_h, map_w]
        """
        # 4次元に変換
        amap = averaged_correlation_map.squeeze(2)

        # 最大値の位置を取得
        flat_map = amap.reshape(self.ny, self.nx, -1)
        max_idx = torch.argmax(flat_map, dim=-1)

        # amap上の２次元座標に変換
        peak_idx_y = max_idx // self.map_h
        peak_idx_x = max_idx % self.map_w

        # 境界クランプ
        peak_idx_y = torch.clamp(peak_idx_y, 1, self.map_h - 2)
        peak_idx_x = torch.clamp(peak_idx_x, 1, self.map_w - 2)

        # 周囲3x3を抽出するためのオフセット
        offsets = torch.tensor([-1, 0, 1], device=amap.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")

        # インデックス計算 [ny, nx, 3, 3]
        idx_y = peak_idx_y.view(self.ny, self.nx, 1, 1) + dy
        idx_x = peak_idx_x.view(self.ny, self.nx, 1, 1) + dx
        idx_flat = (idx_y * self.map_w + idx_x).reshape(self.ny, self.nx, 9)

        # 値を抽出
        peak_vals = torch.gather(flat_map, dim=-1, index=idx_flat)

        return peak_vals.view(self.ny, self.nx, 3, 3), torch.stack(
            [peak_idx_y, peak_idx_x], dim=-1
        )
