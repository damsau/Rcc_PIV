import numpy as np  # 科学計算用
import matplotlib as mpl  # グラフ用
import matplotlib.pyplot as plt  # グラフ用
import matplotlib.cm as cm  # グラフ用
import matplotlib.ticker as mt  # グラフ用
import glob  # ファイル探索用
import pprint as pp  # 標準出力をきれいにする用
from pathlib import Path
from numba import njit
from scipy.ndimage import gaussian_filter

# mplstyleのインポート
plt.style.use("jfm_thesis")

# グローバル変数
WORKING_DIR = Path("/home/masuda/02_experiment/20260423/16rps_001")
PARAMETERS_DIR = WORKING_DIR / Path("parameter")
PARAMETERS_FILE_NAME = Path("default_params.yaml")
IMAGES_DIR = Path("/mnt/b/experiment/data/20260423/water/16rps_001")
IMAGES_FILE_NAME = Path("16rps_4000_Fps150_us*.tif")
DATA_DIR = WORKING_DIR / Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PICS_DIR = DATA_DIR / Path("pics")
PICS_DIR.mkdir(parents=True, exist_ok=True)
VELOCITY_DIR = DATA_DIR / Path("velocity")
VELOCITY_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_TO_MM = 0.0260416666666  # キャリブレーション定数
INTERROGATION_WINDOW_SIZE = 10  # 検査窓サイズ


# 格子データを読み込む関数
def import_grid(imported_grid):
    print(f"\n{imported_grid}を読み込みます")

    # gridを取得
    grid = np.load(imported_grid)  # 次元[n_y, n_x, 2(y, x)]

    # 格子点数を取得
    n_y, n_x, _ = grid.shape
    print(f"速度点数: (N_X, N_X) = {n_x, n_y}")

    # 各座標軸と格子間隔を取得
    axis_x = grid[0, :, 1]
    axis_y = grid[:, 0, 0]
    delta_x = axis_x[1] - axis_x[0]
    delta_y = axis_y[1] - axis_y[0]

    return n_x, n_y, delta_x, delta_y, axis_x, axis_y


# 速度場データを一括読み込みする関数
def import_velocity(imported_filelist, n_x, n_y):
    print(f"\n{imported_filelist[0]}を読み込みます")
    n_data = len(imported_filelist)
    print(f"データ数: {n_data}")

    velocity_all = []

    for i in range(n_data):
        velocity_tmp = np.load(imported_filelist[i]).reshape(n_y, n_x, 2)
        velocity_all.append(velocity_tmp)

    velocity_all = np.array(velocity_all)

    return n_data, velocity_all


# 速度場データを平滑化する関数
def smooth_velocity_by_window(
    velocity,
    window_size,
    delta_x,
    delta_y,
    factor=1.0,
    mode="nearest",
):
    """
    PIV速度場を検査領域サイズ基準でGaussian平滑化する

    Parameters
    ----------
    velocity : ndarray, shape (n_y, n_x, 2)
        PIV速度場
    window_size : float
        検査領域サイズ．delta_x, delta_y と同じ単位
        例：mm
    delta_x : float
        x方向のベクトル間隔
    delta_y : float
        y方向のベクトル間隔
    factor : float
        平滑化幅の倍率
        factor=1.0 なら FWHM = window_size
        factor=0.5 なら FWHM = 0.5 * window_size
    mode : str
        境界処理．"nearest", "reflect" など

    Returns
    -------
    velocity_smooth : ndarray, shape (n_y, n_x, 2)
        平滑化後の速度場
    """

    velocity = np.asarray(velocity)

    if velocity.ndim != 3 or velocity.shape[2] != 2:
        raise ValueError(
            f"velocity must have shape (n_y, n_x, 2), got {velocity.shape}"
        )

    fwhm = factor * window_size

    sigma_phys = fwhm / 2.355

    sigma_y = sigma_phys / delta_y
    sigma_x = sigma_phys / delta_x

    velocity_smooth = np.zeros_like(velocity, dtype=float)

    velocity_smooth[:, :, 0] = gaussian_filter(
        velocity[:, :, 0],
        sigma=(sigma_y, sigma_x),
        mode=mode,
    )

    velocity_smooth[:, :, 1] = gaussian_filter(
        velocity[:, :, 1],
        sigma=(sigma_y, sigma_x),
        mode=mode,
    )

    return velocity_smooth


# 速度勾配テンソルを計算する関数
@njit
def cal_velocity_gradient_tensor(padded_u, padded_v, n_x, n_y, delta_x, delta_y):
    # 変数初期化
    dudx = np.zeros((n_y, n_x))
    dudy = np.zeros((n_y, n_x))
    dvdx = np.zeros((n_y, n_x))
    dvdy = np.zeros((n_y, n_x))

    for j in range(1, n_y + 1):
        for i in range(1, n_x + 1):
            # 値のコピー
            u_ij = padded_u[j, i]
            u_ip1j = padded_u[j, i + 1]
            u_im1j = padded_u[j, i - 1]
            u_ijp1 = padded_u[j + 1, i]
            u_ip1jp1 = padded_u[j + 1, i + 1]
            u_im1jp1 = padded_u[j + 1, i - 1]
            u_ijm1 = padded_u[j - 1, i]
            u_ip1jm1 = padded_u[j - 1, i + 1]
            u_im1jm1 = padded_u[j - 1, i - 1]
            v_ij = padded_v[j, i]
            v_ip1j = padded_v[j, i + 1]
            v_im1j = padded_v[j, i - 1]
            v_ijp1 = padded_v[j + 1, i]
            v_ip1jp1 = padded_v[j + 1, i + 1]
            v_im1jp1 = padded_v[j + 1, i - 1]
            v_ijm1 = padded_v[j - 1, i]
            v_ip1jm1 = padded_v[j - 1, i + 1]
            v_im1jm1 = padded_v[j - 1, i - 1]

            dudx[j - 1, i - 1] = (u_ip1j - u_im1j) / (2 * delta_x)
            dudy[j - 1, i - 1] = (
                u_ip1jp1 - u_ip1jm1 + 2 * (u_ijp1 - u_ijm1) + u_im1jp1 - u_im1jm1
            ) / (2 * delta_y)
            dvdx[j - 1, i - 1] = (
                v_ip1jp1 - v_im1jp1 + 2 * (v_ip1j - v_im1j) + v_ip1jm1 - v_im1jm1
            ) / (2 * delta_y)
            dvdy[j - 1, i - 1] = (v_ip1j - v_im1j) / (2 * delta_y)

    return dudx, dudy, dvdx, dvdy


# ２次元スカラー場をプロットする関数
def plot_scalar_field(
    scalar,
    x=None,
    y=None,
    title="Scalar field",
    xlabel="x",
    ylabel="y",
    cbar_label="value",
    cmap="viridis",
    vmin=None,
    vmax=None,
    aspect="equal",
    save_path=None,
    show=True,
):
    """
    2次元スカラー場をプロットする関数

    Parameters
    ----------
    scalar : ndarray, shape (n_y, n_x)
        プロットしたい2次元スカラー場
    x : ndarray, optional
        x座標．1次元配列 shape (n_x,) または2次元配列 shape (n_y, n_x)
    y : ndarray, optional
        y座標．1次元配列 shape (n_y,) または2次元配列 shape (n_y, n_x)
    title : str
        図のタイトル
    xlabel : str
        x軸ラベル
    ylabel : str
        y軸ラベル
    cbar_label : str
        カラーバーラベル
    cmap : str
        カラーマップ
    vmin, vmax : float, optional
        カラースケールの最小値・最大値
    aspect : str
        アスペクト比．"equal" or "auto"
    save_path : str or Path, optional
        保存先パス．Noneなら保存しない
    show : bool
        Trueならplt.show()する

    Returns
    -------
    fig, ax
        matplotlibのfigureとaxis
    """

    scalar = np.asarray(scalar)

    if scalar.ndim != 2:
        raise ValueError(f"scalar must be 2D array, but got shape {scalar.shape}")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)

    # x, y が指定されていない場合
    if x is None or y is None:
        im = ax.imshow(
            scalar,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
        )

    else:
        x = np.asarray(x)
        y = np.asarray(y)

        # x, y が1次元なら meshgrid にする
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y)
        elif x.ndim == 2 and y.ndim == 2:
            X, Y = x, y
        else:
            raise ValueError("x and y must be both 1D or both 2D arrays")

        im = ax.pcolormesh(
            X,
            Y,
            scalar,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_aspect(aspect)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    return fig, ax


# ハニング窓関数
def _hann_window_1d(n):
    """Hanning窓"""
    return np.hanning(n)


# 1次元のx方向スペクトルを計算する関数
def calc_1d_spectrum_x(
    field,
    dx,
    remove_mean=True,
    apply_window=True,
    use_radial_wavenumber=True,
):
    """
    2次元スカラー場から x方向1Dスペクトルを計算する

    Parameters
    ----------
    field : ndarray, shape (n_y, n_x)
        2次元スカラー場
    dx : float
        x方向格子間隔
    remove_mean : bool
        Trueなら各y列ごとにx方向平均を除去
    apply_window : bool
        Trueならx方向にHanning窓を適用
    use_radial_wavenumber : bool
        Trueなら波数を rad/unit にする．Falseなら cycle/unit

    Returns
    -------
    kx_pos : ndarray
        正のx方向波数
    spectrum_x : ndarray
        x方向1Dスペクトル
    """

    field = np.asarray(field, dtype=float)

    if field.ndim != 2:
        raise ValueError(f"field must be 2D array, got shape {field.shape}")

    n_y, n_x = field.shape
    f = field.copy()

    # NaN対策
    if np.isnan(f).any():
        mean_val = np.nanmean(f)
        f = np.nan_to_num(f, nan=mean_val)

    # 各yに対してx方向平均を除去
    if remove_mean:
        f = f - np.mean(f, axis=1, keepdims=True)

    # 窓関数
    if apply_window:
        wx = _hann_window_1d(n_x)
        f = f * wx[None, :]
        window_correction = np.mean(wx**2)
    else:
        window_correction = 1.0

    # x方向FFT
    f_hat = np.fft.rfft(f, axis=1)

    # パワースペクトル
    spectrum = np.abs(f_hat) ** 2 / n_x**2
    spectrum /= window_correction

    # y方向に平均
    spectrum_x = np.mean(spectrum, axis=0)

    # one-sided補正
    if n_x > 2:
        spectrum_x[1:-1] *= 2.0

    # 波数
    kx = np.fft.rfftfreq(n_x, d=dx)

    if use_radial_wavenumber:
        kx = 2.0 * np.pi * kx

    return kx, spectrum_x


# 1次元y方向スペクトルを計算する関数
def calc_1d_spectrum_y(
    field,
    dy,
    remove_mean=True,
    apply_window=True,
    use_radial_wavenumber=True,
):
    """
    2次元スカラー場から y方向1Dスペクトルを計算する

    Parameters
    ----------
    field : ndarray, shape (n_y, n_x)
        2次元スカラー場
    dy : float
        y方向格子間隔
    remove_mean : bool
        Trueなら各x列ごとにy方向平均を除去
    apply_window : bool
        Trueならy方向にHanning窓を適用
    use_radial_wavenumber : bool
        Trueなら波数を rad/unit にする．Falseなら cycle/unit

    Returns
    -------
    ky_pos : ndarray
        正のy方向波数
    spectrum_y : ndarray
        y方向1Dスペクトル
    """

    field = np.asarray(field, dtype=float)

    if field.ndim != 2:
        raise ValueError(f"field must be 2D array, got shape {field.shape}")

    n_y, n_x = field.shape
    f = field.copy()

    # NaN対策
    if np.isnan(f).any():
        mean_val = np.nanmean(f)
        f = np.nan_to_num(f, nan=mean_val)

    # 各xに対してy方向平均を除去
    if remove_mean:
        f = f - np.mean(f, axis=0, keepdims=True)

    # 窓関数
    if apply_window:
        wy = _hann_window_1d(n_y)
        f = f * wy[:, None]
        window_correction = np.mean(wy**2)
    else:
        window_correction = 1.0

    # y方向FFT
    f_hat = np.fft.rfft(f, axis=0)

    # パワースペクトル
    spectrum = np.abs(f_hat) ** 2 / n_y**2
    spectrum /= window_correction

    # x方向に平均
    spectrum_y = np.mean(spectrum, axis=1)

    # one-sided補正
    if n_y > 2:
        spectrum_y[1:-1] *= 2.0

    # 波数
    ky = np.fft.rfftfreq(n_y, d=dy)

    if use_radial_wavenumber:
        ky = 2.0 * np.pi * ky

    return ky, spectrum_y


# 1次元等方スペクトルを計算する関数
def calc_isotropic_spectrum_2d(
    field,
    dx,
    dy,
    n_bins=100,
    remove_mean=True,
    apply_window=True,
    use_radial_wavenumber=True,
    bin_method="mean",
):
    """
    2次元スカラー場から等方1Dスペクトルを計算する

    Parameters
    ----------
    field : ndarray, shape (n_y, n_x)
        2次元スカラー場
    dx : float
        x方向格子間隔
    dy : float
        y方向格子間隔
    n_bins : int
        波数bin数
    remove_mean : bool
        Trueなら全体平均を除去
    apply_window : bool
        Trueなら2D Hanning窓を適用
    use_radial_wavenumber : bool
        Trueなら波数を rad/unit にする．Falseなら cycle/unit
    bin_method : {"mean", "sum"}
        "mean"なら同じ波数帯で平均
        "sum"なら同じ波数帯で総和

    Returns
    -------
    k_bin_center : ndarray
        波数bin中心
    spectrum_iso : ndarray
        等方1Dスペクトル
    """

    field = np.asarray(field, dtype=float)

    if field.ndim != 2:
        raise ValueError(f"field must be 2D array, got shape {field.shape}")

    if bin_method not in ["mean", "sum"]:
        raise ValueError("bin_method must be 'mean' or 'sum'")

    n_y, n_x = field.shape
    f = field.copy()

    # NaN対策
    if np.isnan(f).any():
        mean_val = np.nanmean(f)
        f = np.nan_to_num(f, nan=mean_val)

    # 平均除去
    if remove_mean:
        f -= np.mean(f)

    # 2D窓関数
    if apply_window:
        wx = _hann_window_1d(n_x)
        wy = _hann_window_1d(n_y)
        window = wy[:, None] * wx[None, :]
        f = f * window
        window_correction = np.mean(window**2)
    else:
        window_correction = 1.0

    # 2D FFT
    f_hat = np.fft.fft2(f)

    # 2Dパワースペクトル
    spectrum_2d = np.abs(f_hat) ** 2 / (n_x * n_y) ** 2
    spectrum_2d /= window_correction

    # 波数
    kx = np.fft.fftfreq(n_x, d=dx)
    ky = np.fft.fftfreq(n_y, d=dy)

    if use_radial_wavenumber:
        kx = 2.0 * np.pi * kx
        ky = 2.0 * np.pi * ky

    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # flatten
    k_flat = K.ravel()
    s_flat = spectrum_2d.ravel()

    # k=0を除いてbin範囲を作る
    k_max = np.max(k_flat)
    bins = np.linspace(0.0, k_max, n_bins + 1)
    k_bin_center = 0.5 * (bins[:-1] + bins[1:])

    spectrum_iso = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    bin_index = np.digitize(k_flat, bins) - 1

    for i in range(k_flat.size):
        idx = bin_index[i]
        if 0 <= idx < n_bins:
            spectrum_iso[idx] += s_flat[i]
            counts[idx] += 1

    valid = counts > 0

    if bin_method == "mean":
        spectrum_iso[valid] /= counts[valid]

    spectrum_iso[~valid] = np.nan

    return k_bin_center, spectrum_iso


# 全てのスペクトルを計算する関数
def calc_all_1d_spectra(
    field,
    dx,
    dy,
    n_bins=100,
    remove_mean=True,
    apply_window=True,
    use_radial_wavenumber=True,
    isotropic_bin_method="mean",
):
    """
    x方向，y方向，等方1Dスペクトルをまとめて計算する

    Returns
    -------
    spectra : dict
        {
            "kx": kx,
            "spectrum_x": spectrum_x,
            "ky": ky,
            "spectrum_y": spectrum_y,
            "k_iso": k_iso,
            "spectrum_iso": spectrum_iso,
        }
    """

    kx, spectrum_x = calc_1d_spectrum_x(
        field,
        dx=dx,
        remove_mean=remove_mean,
        apply_window=apply_window,
        use_radial_wavenumber=use_radial_wavenumber,
    )

    ky, spectrum_y = calc_1d_spectrum_y(
        field,
        dy=dy,
        remove_mean=remove_mean,
        apply_window=apply_window,
        use_radial_wavenumber=use_radial_wavenumber,
    )

    k_iso, spectrum_iso = calc_isotropic_spectrum_2d(
        field,
        dx=dx,
        dy=dy,
        n_bins=n_bins,
        remove_mean=remove_mean,
        apply_window=apply_window,
        use_radial_wavenumber=use_radial_wavenumber,
        bin_method=isotropic_bin_method,
    )

    spectra = {
        "kx": kx,
        "spectrum_x": spectrum_x,
        "ky": ky,
        "spectrum_y": spectrum_y,
        "k_iso": k_iso,
        "spectrum_iso": spectrum_iso,
    }

    return spectra


# スペクトルを表示する関数
def plot_all_1d_spectra(
    spectra,
    title="1D spectra",
    ylabel="Spectrum",
):
    """
    x方向，y方向，等方スペクトルをまとめてlog-logプロットする
    """

    kx = spectra["kx"]
    sx = spectra["spectrum_x"]

    ky = spectra["ky"]
    sy = spectra["spectrum_y"]

    k_iso = spectra["k_iso"]
    s_iso = spectra["spectrum_iso"]

    fig, ax = plt.subplots(figsize=(3, 4), dpi=300)

    valid_x = np.isfinite(sx) & (sx > 0) & (kx > 0)
    valid_y = np.isfinite(sy) & (sy > 0) & (ky > 0)
    valid_iso = np.isfinite(s_iso) & (s_iso > 0) & (k_iso > 0)

    ax.plot(
        k_iso[valid_iso],
        10 ** (10) * k_iso[valid_iso] ** (-1),
        marker="None",
        color="black",
        linestyle="--",
    )

    ax.loglog(kx[valid_x], sx[valid_x], marker="None", label="x direction")
    ax.loglog(ky[valid_y], sy[valid_y], marker="None", label="y direction")
    ax.loglog(k_iso[valid_iso], s_iso[valid_iso], marker="None", label="isotropic")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.set_ylim(10 ** (-4), 10 ** (2))
    # ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()
    plt.show()

    return fig, ax


# 2次の速度構造関数を計算する関数
def calc_velocity_structure_function_uv(
    u,
    v,
    dx,
    dy,
    max_lag_x=None,
    max_lag_y=None,
    order=2,
    component="vector",
):
    """
    u, v が別々に与えられた2次元速度場から速度構造関数を計算する

    Parameters
    ----------
    u : ndarray, shape (n_y, n_x)
        x方向速度成分

    v : ndarray, shape (n_y, n_x)
        y方向速度成分

    dx : float
        x方向格子間隔

    dy : float
        y方向格子間隔

    max_lag_x : int or None
        x方向に何格子分までラグを取るか
        Noneなら n_x//2

    max_lag_y : int or None
        y方向に何格子分までラグを取るか
        Noneなら n_y//2

    order : int or float
        構造関数の次数
        order=2 なら二次構造関数

    component : {"vector", "u", "v"}
        "vector" : <|Δu_vec|^order>
        "u"      : <|Δu|^order>
        "v"      : <|Δv|^order>

    Returns
    -------
    result : dict
        {
            "r_x": x方向ラグ距離,
            "D_x": x方向構造関数,
            "count_x": x方向の平均点数,
            "r_y": y方向ラグ距離,
            "D_y": y方向構造関数,
            "count_y": y方向の平均点数,
        }
    """

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"u and v must be 2D arrays, got u={u.shape}, v={v.shape}")

    if u.shape != v.shape:
        raise ValueError(f"u and v must have same shape, got u={u.shape}, v={v.shape}")

    if component not in ["vector", "u", "v"]:
        raise ValueError("component must be 'vector', 'u', or 'v'")

    n_y, n_x = u.shape

    if max_lag_x is None:
        max_lag_x = n_x // 2

    if max_lag_y is None:
        max_lag_y = n_y // 2

    D_x = np.zeros(max_lag_x, dtype=float)
    D_y = np.zeros(max_lag_y, dtype=float)

    count_x = np.zeros(max_lag_x, dtype=int)
    count_y = np.zeros(max_lag_y, dtype=int)

    # -------------------------
    # x方向ラグ
    # -------------------------
    for lag in range(1, max_lag_x + 1):
        du = u[:, lag:] - u[:, :-lag]
        dv = v[:, lag:] - v[:, :-lag]

        if component == "vector":
            diff = np.sqrt(du**2 + dv**2)
        elif component == "u":
            diff = np.abs(du)
        else:  # component == "v"
            diff = np.abs(dv)

        valid = np.isfinite(diff)

        if np.any(valid):
            D_x[lag - 1] = np.mean(diff[valid] ** order)
            count_x[lag - 1] = np.sum(valid)
        else:
            D_x[lag - 1] = np.nan
            count_x[lag - 1] = 0

    # -------------------------
    # y方向ラグ
    # -------------------------
    for lag in range(1, max_lag_y + 1):
        du = u[lag:, :] - u[:-lag, :]
        dv = v[lag:, :] - v[:-lag, :]

        if component == "vector":
            diff = np.sqrt(du**2 + dv**2)
        elif component == "u":
            diff = np.abs(du)
        else:  # component == "v"
            diff = np.abs(dv)

        valid = np.isfinite(diff)

        if np.any(valid):
            D_y[lag - 1] = np.mean(diff[valid] ** order)
            count_y[lag - 1] = np.sum(valid)
        else:
            D_y[lag - 1] = np.nan
            count_y[lag - 1] = 0

    r_x = np.arange(1, max_lag_x + 1) * dx
    r_y = np.arange(1, max_lag_y + 1) * dy

    result = {
        "r_x": r_x,
        "D_x": D_x,
        "count_x": count_x,
        "r_y": r_y,
        "D_y": D_y,
        "count_y": count_y,
    }

    return result


import matplotlib.pyplot as plt
import numpy as np


# 速度構造関数を表示する関数
def plot_velocity_structure_function(
    sf,
    title="Velocity structure function",
    ylabel=r"$D_2(r)$",
):
    r_x = sf["r_x"]
    D_x = sf["D_x"]

    r_y = sf["r_y"]
    D_y = sf["D_y"]

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    valid_x = np.isfinite(D_x) & (D_x > 0) & (r_x > 0)
    valid_y = np.isfinite(D_y) & (D_y > 0) & (r_y > 0)

    ax.plot(
        r_x[valid_x],
        # 10 ** (4) * r_x[valid_x] ** (2 / 3),
        10 ** (4) * r_x[valid_x] ** (2),
        # 10 ** (6) * r_x[valid_x] ** (1),
        marker="None",
        color="black",
        linestyle="--",
    )

    ax.loglog(
        r_x[valid_x], D_x[valid_x], marker="None", color="red", label="x direction"
    )
    ax.loglog(
        r_y[valid_y], D_y[valid_y], marker="None", color="blue", label="y direction"
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()
    plt.show()

    return fig, ax


def main():
    # 格子データの読み込み
    imported_gridfile = VELOCITY_DIR / Path("grid_position_mm.npy")
    N_X, N_Y, DELTA_X, DELTA_Y, AXIS_X, AXIS_Y = import_grid(imported_gridfile)

    # 速度場データの読み込み
    imported_filelist = VELOCITY_DIR / Path("velocity_mmps_target*_median.npy")
    imported_filelist = sorted(glob.glob(str(imported_filelist)))
    n_data, velocity_all = import_velocity(imported_filelist, N_X, N_Y)

    # PIVデータの後処理/解析
    for idx_data in range(n_data):
        print(f"\n解析データ: {idx_data + 1} / {n_data}")

        # 速度場を検査窓サイズで平滑化
        velocity_smoothed = smooth_velocity_by_window(
            velocity_all[idx_data],
            INTERROGATION_WINDOW_SIZE * PIXEL_TO_MM,
            DELTA_X,
            DELTA_Y,
            factor=1.5,
            mode="nearest",
        )
        print(f"検査窓サイズ: {INTERROGATION_WINDOW_SIZE * PIXEL_TO_MM} [mm]")
        print(f"格子幅: {DELTA_X, DELTA_Y} [mm]")

        # 速度配列をパディング
        padded_u = np.pad(velocity_smoothed[:, :, 1], pad_width=1, mode="edge")
        padded_v = np.pad(velocity_smoothed[:, :, 0], pad_width=1, mode="edge")

        # 速度勾配の計算
        dudx, dudy, dvdx, dvdy = cal_velocity_gradient_tensor(
            padded_u, padded_v, N_X, N_Y, DELTA_X, DELTA_Y
        )

        # dudxの確認
        if False:
            vmax = np.nanmax(np.abs(dudx))
            plot_scalar_field(
                dudx,
                x=AXIS_X,
                y=AXIS_Y,
                title=r"$\pdv{u}{x}$",
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                cbar_label=r"$\pdv{u}{x}$ [1/s]",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )

        # dudyの確認
        if False:
            vmax = np.nanmax(np.abs(dudy))
            plot_scalar_field(
                dudy,
                x=AXIS_X,
                y=AXIS_Y,
                title=r"$\pdv{u}{y}$",
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                cbar_label=r"$\pdv{u}{y}$ [1/s]",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )

        # dvdxの確認
        if False:
            vmax = np.nanmax(np.abs(dvdx))
            plot_scalar_field(
                dvdx,
                x=AXIS_X,
                y=AXIS_Y,
                title=r"$\pdv{v}{x}$",
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                cbar_label=r"$\pdv{v}{x}$ [1/s]",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )

        # dvdyの確認
        if False:
            vmax = np.nanmax(np.abs(dvdy))
            plot_scalar_field(
                dvdy,
                x=AXIS_X,
                y=AXIS_Y,
                title=r"$\pdv{v}{y}$",
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                cbar_label=r"$\pdv{v}{y}$ [1/s]",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )

        # 歪み速度と渦度の計算
        strainrate = dvdx + dudy
        vorticity = dvdx - dudy

        # 歪み速度の確認
        if False:
            vmax = np.nanmax(np.abs(strainrate))
            plot_scalar_field(
                strainrate,
                x=AXIS_X,
                y=AXIS_Y,
                title=r"Strainrate",
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                cbar_label=r"$S$ [1/s]",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )

        # 渦度の確認
        if False:
            vmax = np.nanmax(np.abs(vorticity))
            plot_scalar_field(
                vorticity,
                x=AXIS_X,
                y=AXIS_Y,
                title=r"Vorticity",
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                cbar_label=r"$\Omega$ [1/s]",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )

        # 1次元スペクトルの計算
        # エンストロフィ
        enstrophy_spectra = calc_all_1d_spectra(
            vorticity**2,
            dx=DELTA_X,
            dy=DELTA_Y,
            n_bins=100,
            remove_mean=True,
            apply_window=True,
            use_radial_wavenumber=True,
        )

        # エンストロフィースペクトルの確認
        if False:
            plot_all_1d_spectra(
                enstrophy_spectra,
                title="Spectrum of enstrophy",
                ylabel="Power spectrum",
            )

        # 歪み速度の2乗
        sq_strainrate_spectra = calc_all_1d_spectra(
            strainrate**2,
            dx=DELTA_X,
            dy=DELTA_Y,
            n_bins=100,
            remove_mean=True,
            apply_window=True,
            use_radial_wavenumber=True,
        )

        # 2乗歪み速度スペクトルの確認
        if False:
            plot_all_1d_spectra(
                sq_strainrate_spectra,
                title="Spectrum of Sq. Strainrate",
                ylabel="Power spectrum",
            )

        # 速度構造関数の計算
        sf = calc_velocity_structure_function_uv(
            velocity_smoothed[:, :, 1],
            velocity_smoothed[:, :, 0],
            dx=DELTA_X,
            dy=DELTA_Y,
            max_lag_x=N_X,
            max_lag_y=N_Y,
            order=2,
            component="u",  # x方向の縦速度
        )

        # 速度構造関数の確認
        if True:
            plot_velocity_structure_function(
                sf,
                title="2nd-order velocity structure function",
                ylabel=r"$D_2(r)$",
            )


if __name__ == "__main__":
    main()
