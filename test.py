# モジュール
import yaml
import argparse
import pprint as pp
import os.path as osp
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 自作モジュール
from utils import io_data as id
from utils import tools
from modules import piv_modules as pm

# デバイス設定
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA Available: No")
    print(f"Warning: Training URAFT on CPU will be very slow.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# パラメータの読み込み
parser = argparse.ArgumentParser()
parser.add_argument("--params-file", type=str, default="-")
args = parser.parse_args()
with open(args.params_file, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)
    pp.pprint(params, sort_dicts=False)

# パラメータ取得
io_params = params["file_io"]
piv_params = params["rcc_piv"]

# Datasetの作成
particle_images = id.ParticleImageDataset(
    root=osp.join(io_params["workdir_path"], "images"),
    filename=piv_params["img_name"] + piv_params["img_ext"],
    ref_skip_low=piv_params["ref_skip_low"],
    ref_skip_high=piv_params["ref_skip_high"],
    pivstep_skip=piv_params["pivstep_skip"],
)

# 解析範囲の設定
if piv_params["start"] == "None" and piv_params["end"] == "None":
    start = particle_images.start
    end = particle_images.end
    skip = particle_images.pivstep_skip
else:
    start = piv_params["start"]
    end = piv_params["end"]
    skip = piv_params["pivstep_skip"]

# 解析するペア画像を取得
img_pair = particle_images[0].to(device)
print("img_pair.shape: ", img_pair.shape)

# 再帰処理ループ
rcc_num = piv_params["rcc_num"]
offset = 0
for rcc_step in range(1):
    windows_setting = pm.WindowSetup(
        piv_params,
        rcc_step,
        particle_images.img_width,
        particle_images.img_height,
        img_pair,
        offset,
    )

    # 検査画像の取得
    interrogation_img = windows_setting.get_interrogation_img()
    print("interrogation_img.shape: ", interrogation_img.shape)

    # 速度点の数を取得
    nx, ny = windows_setting.get_dimention()
    print("nx, ny: ", nx, ny)

    # 元画像との相関計算
    reference = img_pair[2:3, 0:1, :, :]
    tools.show_img_from_tensor(reference, title="reference")
    reference = reference - torch.mean(reference, dim=(2, 3), keepdim=True)

    template = interrogation_img[0, 0, 0:1, 0:1, :, :]
    tools.show_img_from_tensor(template, title="template")
    template = template - torch.mean(template, dim=(2, 3), keepdim=True)
    correlation_map = F.conv2d(reference, template) / interrogation_img.shape[4] ** 2
    print("correlation_map.shape: ", correlation_map.shape)
    print("correlation_map: ", correlation_map)
    tools.show_img_from_tensor(correlation_map, title="correlation")
