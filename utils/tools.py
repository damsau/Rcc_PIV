import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch


# カラーマップの定義
def generate_cmap(colors, cmap_name="custom_cmap"):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for vi, ci in zip(values, colors):
        color_list.append((vi / vmax, ci))

    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)


def show_img_from_tensor(tensor, title="Image"):
    """
    ２次元テンソルを可視化する関数
    tensor: ２次元のテンソル
    title: タイトル
    """
    # numpy配列に変換
    img_np = tensor.detach().cpu().numpy()

    if img_np.ndim == 4:
        img_np = img_np[0, 0]
    elif img_np.ndim == 3:
        img_np == img_np[0]

    plt.imshow(img_np, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.show()


CMAP_THERMAL = generate_cmap(
    ["#1c3f75", "#068fb9", "#f1e235", "#d64e8b", "#730e22"], "cmthermal"
)


def show_3d_img_from_tensor(tensor, title="None"):
    # テンソルをCPUに移動して，Numpy配列に変換
    data = tensor.detach().cpu().numpy()
    h, w = data.shape

    # x, y座標の作成
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)

    # プロットの作成
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # 面の描画
    surf = ax.plot_surface(X, Y, data, cmap=CMAP_THERMAL, edgecolor="none", alpha=0.8)

    # ラベルとタイトルの設定
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta X$")
    ax.set_ylabel(r"$\Delta Y$")

    # カラーバーの追加
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 視点の調整（俯瞰で）
    ax.view_init(elev=30, azim=45)

    plt.show()


def plot_vector(x_mesh, y_mesh, vector, output_filepath=None):

    # 引数をnumpy配列に変換
    x_mesh = x_mesh.detach().cpu().numpy()
    y_mesh = y_mesh.detach().cpu().numpy()
    vector = vector.detach().cpu().numpy()

    plt.style.use("PIV_results")

    fig, ax = plt.subplots(1, 1)

    # 軸ラベル
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # 描画範囲
    ax.set_xlim(np.min(x_mesh) - 50, np.max(x_mesh) + 50)
    ax.set_ylim(np.min(y_mesh) - 50, np.max(y_mesh) + 50)

    # 間引き間隔
    s = slice(None, None, 1)

    # vectorの規格化
    vector_abs = np.sqrt(vector[:, :, 0] ** 2 + vector[:, :, 1] ** 2)
    vector_abs[vector_abs == 0] = 1.0
    vector_x = vector[:, :, 1] / vector_abs
    vector_y = vector[:, :, 0] / vector_abs

    # 描画
    q = ax.quiver(
        x_mesh[s, s],
        y_mesh[s, s],
        vector_x[s, s],
        vector_y[s, s],
        vector_abs[s, s],
        angles="xy",
        scale_units="xy",
        scale=0.075,
        pivot="mid",
        cmap=CMAP_THERMAL,
    )

    # アスペクト比を同じに
    ax.set_aspect("equal")

    # カラーバーの設定
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(q, cax=cax, extend="max")
    cbar.set_label("Magnitude")

    fig.tight_layout()
    # fig.subplots_adjust()

    if output_filepath is None:
        plt.show()
    else:
        fig.savefig(fname=output_filepath, bbox_inches="tight")
