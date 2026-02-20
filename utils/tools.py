import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    fig = plt.figure(figsize=(10, 7))
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
