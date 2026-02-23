import torch
import torch.nn.functional as F


class RecursivCorrelationPIV:
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
        peak_idx: 探査画像に対して相関が最大となる位置 [int(ref_imgs_num / 2), ny, nx, 2]
        iw_lt: 検査画像の左上のインデックス [ny, nx, 2]
        sw_lt: 探査画像の左上のインデックス [ref_imgs_num, ny, nx, 2]
        """

        displacement = peak_idx + sw_lt[int(self.ref_imgs_num / 2) :] - iw_lt

        return displacement

    def get_mask_flag(self, displacement, threshold=3.0):

        # 引数をキャスト
        disp_high = displacement[0, ...]
        disp_low = displacement[1, ...]

        # 高速域用での変位の絶対値を計算
        disp_high_mag = torch.norm(disp_high, dim=-1)

        # マスクの作成: 高速域用の変位がthreshold未満なら低速域用の変位を用いる
        flag_use_low = disp_high_mag < threshold
        flag_use_low = flag_use_low.unsqueeze(-1).expand_as(disp_low)

        return flag_use_low

    def get_subpixel_displacement(self, peak_vals):
        """
        サブピクセル変位を計算する
            peak_vals: 3x3の相関値配列
        """
        # 1. 共分散を用いているため，負の共分散を微小な正の値に置き換える
        peak_vals_non_negative = torch.clamp(peak_vals, min=1e-5)
        log_peak_vals_nn = torch.log(peak_vals_non_negative)

        # x, y方向のスライスを取得
        Rm1_y = log_peak_vals_nn[..., 0, 1]  # 上
        R0_y = log_peak_vals_nn[..., 1, 1]  # 中心
        Rp1_y = log_peak_vals_nn[..., 2, 1]  # 下
        Rm1_x = log_peak_vals_nn[..., 1, 0]  # 左
        R0_x = log_peak_vals_nn[..., 1, 1]  # 中心
        Rp1_x = log_peak_vals_nn[..., 1, 2]  # 右

        # サブピクセル変位の分子nomと分母denを計算
        nom_y = Rm1_y - Rp1_y
        den_y = 2.0 * (Rm1_y - 2.0 * R0_y + Rp1_y)
        nom_x = Rm1_x - Rp1_x
        den_x = 2.0 * (Rm1_x - 2.0 * R0_x + Rp1_x)

        # ゼロ除算回避
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
        subpixel_displacement_y = torch.clamp(
            subpixel_displacement_y, -1.0, 1.0
        ).unsqueeze(-1)
        subpixel_displacement_x = torch.clamp(
            subpixel_displacement_x, -1.0, 1.0
        ).unsqueeze(-1)

        return torch.cat((subpixel_displacement_y, subpixel_displacement_x), dim=-1)

    def get_offset(self, displacement, iw_center_old, iw_center):
        """
        探査領域のオフセット変位を計算
        displacement: 前rcc_stepで計算した変位 [2, ny, nx, 2] （ny, nxは前rcc_stepの大きさ）
        """
        if displacement is None and iw_center_old is None:
            return None
        else:
            # 引数を浮動小数に変換
            disp = displacement.float().permute(0, 3, 1, 2)
            iw_center_old = iw_center_old.float()
            iw_center = iw_center.float()

            # 新しい座標を古い座標を用いて正規化
            # 古い座標（iw_center_old）の四隅（最小値・最大値）を取得します
            y_min = iw_center_old[..., 0].min()
            y_max = iw_center_old[..., 0].max()
            x_min = iw_center_old[..., 1].min()
            x_max = iw_center_old[..., 1].max()

            # iw_centerのx, y座標をスケーリング
            y_norm = 2.0 * (iw_center[..., 0] - y_min) / (y_max - y_min) - 1.0
            x_norm = 2.0 * (iw_center[..., 1] - x_min) / (x_max - x_min) - 1.0

            # 形状が[1, ny', nx', 2]のグリッドを構築
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

            # バイリニア補間でサンプリング
            # 高速域用
            disp_interpolated_high = F.grid_sample(
                disp[0:1, ...],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            # 低速域用
            disp_interpolated_low = F.grid_sample(
                disp[0:1, ...],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)

            # offsetの設定
            offset = torch.cat(
                (
                    -disp_interpolated_low,
                    -disp_interpolated_high,
                    disp_interpolated_high,
                    disp_interpolated_low,
                ),
                dim=0,
            )
            return offset

    def get_velocity(self, displacement, mask_flag, dt):
        """
        変位から速度を計算する
        displacement: サブピクセル単位まで考慮した変位（各速度域での）
        mask_flag: 低速域用の解析結果を用いるflag
        dt: 時間刻み
        """
        dt_high = dt["high"]
        dt_low = dt["low"]

        velocity = torch.where(
            mask_flag, displacement[1, ...] / dt_low, displacement[0, ...] / dt_high
        )

        return velocity

    def correct_errors(self, velocity, error_threshold=1.0):
        """
        誤ベクトルを重み付き線形補完による修正を一括で行う関数
        velocity: 速度ベクトル [ny, nx, 2]
        error_threshold: 誤ベクトル判定の閾値
        """
        vel = velocity.permute(2, 0, 1).unsqueeze(0).float()  # 次元: [1, 2, ny, nx]

        # --- 誤ベクトル検知 ---
        # 端を１点づつコピーして増やす
        vel_padded = F.pad(velocity, (1, 1, 1, 1), mode="replicate")

        # 3x3の窓を全速度点分切り出す
        patches = F.unfold(vel_padded, kernel_size=3).view(1, 2, 9, self.ny, self.nx)

        # 中央値ベクトルを計算
        filtered_vec, _ = torch.median(patches, dim=2, keepdim=True)

        # 判定対象のベクトル
        center_vec = patches[:, :, 4:5, :, :]

        # center_vecと周囲のベクトルの中央値の差の絶対値(判定式の分子)を計算
        diff_vec_abs = torch.norm(center_vec - filtered_vec, dim=1, keepdim=True)

        # 周囲8点の速度ベクトルを抽出
        neighbor_idx = [0, 1, 2, 3, 5, 6, 7, 8]
        neighbors = patches[:, :, neighbor_idx, :, :]

        # 周囲と中央値の差の絶対値の中央値（判定式の分母のrm）を計算
        diff_arround_vec_abs = torch.norm(neighbors - filtered_vec, dim=1, keepdim=True)
        r_m, _ = torch.median(diff_arround_vec_abs, dim=2, keepdim=True)

        # 閾値判定
        threshold_map = diff_vec_abs / (r_m + 0.1)
        error_flag = (threshold_map >= error_threshold).squeeze()

        # --- 誤ベクトル修正 ---
        # 距離の重み付き線形補完用のカーネルを定義
        kernel_weight = torch.tensor(
            [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
            device=vel.device,
            dtype=vel.dtype,
        )
        kernel = kernel_weight.view(1, 1, 3, 3).repeat(2, 1, 1, 1)  # 次元: [2, 1, 3, 3]

        # 正しいベクトルをTrueとするflag
        valid_flag = (
            (~error_flag)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, 2, self.ny, self.nx)
        )

        # 誤ベクトルと判定されたベクトルを0に
        vel_valid = vel * valid_flag

        # 畳み込み(conv2d)をつかって，重み付き和を計算
        vel_valid_pad = F.pad(vel_valid, (1, 1, 1, 1), mode="constant", value=0.0)
        weighted_sum = F.conv2d(vel_valid_pad, kernel, groups=2)

        # 周囲の重みの計算
        flag_valid_pad = F.pad(valid_flag, (1, 1, 1, 1), mode="constant", value=0.0)
        weight_sum = F.conv2d(flag_valid_pad, kernel, groups=2)

        # 線形補完したベクトルを計算
        corrected_vel_vals = weighted_sum / (weight_sum + 1e-8)

        # 周囲8点がすべて誤ベクトルだった場合は，3x3中央値で代用
        corrected_vel_vals = torch.where(
            weight_sum > 0, corrected_vel_vals, filtered_vec.squeeze(2)
        )

        # flag_errorがTrueの点だけ，置き換える
        corrected_vel_vals = corrected_vel_vals.squeeze(0).permute(1, 2, 0)
        corrected_velocity = torch.where(
            error_flag.unsqueeze(-1), corrected_vel_vals, velocity
        )

        return corrected_velocity, error_flag


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
        # バッチサイズの取得
        ref_imgs_num = averaged_correlation_map.shape[0]

        # 1次元の要素を消去
        avg_corr_map = averaged_correlation_map.squeeze(3)

        # 最大値の位置を取得
        flat_map = avg_corr_map.reshape(ref_imgs_num, self.ny, self.nx, -1)  # 平坦化
        max_idx = torch.argmax(flat_map, dim=-1)

        # amap上の２次元座標に変換
        peak_idx_y = max_idx // self.map_w
        peak_idx_x = max_idx % self.map_w

        # 境界クランプ
        peak_idx_y = torch.clamp(peak_idx_y, 1, self.map_h - 2)
        peak_idx_x = torch.clamp(peak_idx_x, 1, self.map_w - 2)

        # 周囲3x3を抽出するためのオフセット
        offsets = torch.tensor([-1, 0, 1], device=avg_corr_map.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")

        # インデックス計算 [B, ny, nx, 3, 3]
        idx_y = peak_idx_y.view(ref_imgs_num, self.ny, self.nx, 1, 1) + dy
        idx_x = peak_idx_x.view(ref_imgs_num, self.ny, self.nx, 1, 1) + dx
        idx_flat = (idx_y * self.map_w + idx_x).reshape(
            ref_imgs_num, self.ny, self.nx, 9
        )

        # 値を抽出 [B, ny, nx, 9]
        peak_vals = torch.gather(flat_map, dim=-1, index=idx_flat)

        return peak_vals.view(ref_imgs_num, self.ny, self.nx, 3, 3), torch.stack(
            [peak_idx_y, peak_idx_x], dim=-1
        )

    def evaluate_sn_ratio(self, averaged_correlation_map, mask_radius=3):
        """
        Signal-Noise比を計算する
        averaged_correlation_map: 平均相関値配列 [2, ny, nx, map_h, map_w]
        """
        # 第1ピークの取得
        corr_map_flat = averaged_correlation_map.view(2, self.ny, self.nx, -1)
        R_min, _ = torch.min(corr_map_flat, dim=-1)
        R1, idx1 = torch.max(corr_map_flat, dim=-1)

        # 第1ピークの位置座標
        idx1_y = idx1 // self.map_w
        idx1_x = idx1 % self.map_w

        # 相関マップ全体に対する座標グリッドの作成
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.map_h, device=corr_map_flat.device),
            torch.arange(self.map_w, device=corr_map_flat.device),
            indexing="ij",
        )
        grid_y = grid_y.view(1, 1, 1, self.map_h, self.map_w)
        grid_x = grid_x.view(1, 1, 1, self.map_h, self.map_w)

        # 各ピクセルの第1ピークからの距離を計算
        dist_y = torch.abs(grid_y - idx1_y.view(2, self.ny, self.nx, 1, 1))
        dist_x = torch.abs(grid_x - idx1_x.view(2, self.ny, self.nx, 1, 1))

        # 第1ピーク付近を-infでマスク
        mask_flag = (dist_y <= mask_radius) & (dist_x <= mask_radius)
        masked_corr_map = averaged_correlation_map.clone()
        masked_corr_map[mask_flag] = float("-inf")

        # 第2ピークを取得
        masked_corr_map_flat = masked_corr_map.view(2, self.ny, self.nx, -1)
        R2, _ = torch.max(masked_corr_map_flat, dim=-1)
        R2 = torch.clamp(R2, min=1e-1)

        # SN比を計算
        sn_ratio = (R1 - R_min) / (R2 - R_min)

        return sn_ratio
