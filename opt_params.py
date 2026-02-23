# モジュール
import optuna
import yaml
import argparse
import pprint as pp
import os.path as osp
import os
from contextlib import redirect_stdout
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader

# 自作モジュール
from utils import io_data as id
from utils import tools
from modules import piv_modules as pm
from rcc_piv import rcc_piv

# デバイス設定
print("\nデバイスと環境の設定を確認します...")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA Available: No")
    print(f"Warning: Training URAFT on CPU will be very slow.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def objective(trial, base_params, rcc_step=0):
    """
    Oputunaが探索を行うための目的関数
    """

    # 探索パラメータの候補
    list_iw_size = [64, 48, 24, 12]
    list_sw_ratio = [2.5, 2, 1.5]

    # 2. 探索するパラメータの定義
    ref_skip_high = trial.suggest_int("skip_high", 0, 4)
    ref_skip_low = trial.suggest_int("skip_low", ref_skip_high, 8)  # 低速域の参照間隔
    iw_size = trial.suggest_int(
        "iw_size", list_iw_size[rcc_step + 1], list_iw_size[rcc_step], step=4
    )  # 検査領域サイズ
    sw_ratio = trial.suggest_float(
        "sw_ratio", 1.2, list_sw_ratio[rcc_step], step=0.1
    )  # 探査領域サイズ
    sw_size = (int(iw_size * sw_ratio) // 2) * 2
    mask_threshold = trial.suggest_int("mask_threshold", 1, 10, step=1)

    # パラメータの上書き
    params["rcc_piv"]["rcc_num"] = rcc_step + 1  # 再帰的処理のステップをrcc_stepに対応
    params["rcc_piv"]["ref_skip_low"] = ref_skip_low
    params["rcc_piv"]["ref_skip_high"] = ref_skip_high
    params["rcc_piv"]["iw_size"][rcc_step] = iw_size
    params["rcc_piv"]["sw_size"][rcc_step] = sw_size
    params["rcc_piv"]["margin_size"][rcc_step] = int(sw_size // 2) + 5
    params["rcc_piv"]["mask_threshold"] = mask_threshold

    try:
        # 3. PIV実行と評価
        with open(os.devnull, "w") as f, redirect_stdout(f):
            error_ratio_all, sn_ratio_all = rcc_piv(params, mode="optimize")

        # 総合スコアの算出
        error_ratio_mean = np.median(error_ratio_all)
        sn_ratio_mean = np.median(sn_ratio_all)

        score = error_ratio_mean / sn_ratio_mean

        print(
            f"Trial {trial.number}: iw_size = {iw_size}, sw_size = {sw_size} -> ER = {error_ratio_mean*100:.2f}%, SNR = {sn_ratio_mean:.2f}, Score = {score:.4f}"
        )

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    return score


# pivの解析パラメータをベイズ推定で最適化する関数
def optimize_params(params):
    print("\nPIVの解析パラメータを最適化します...")

    # 再帰的処理のステップごとに最適化
    rcc_num = params["rcc_piv"]["rcc_num"]
    for rcc_step in range(rcc_num):
        print(f"\nrcc_step: {rcc_step + 1} / {rcc_num}を最適化中...")

        # Optunaのスタディを作成
        study = optuna.create_study(direction="minimize")

        # 最適化の実行
        study.optimize(
            lambda trial: objective(trial, params, rcc_step=rcc_step), timeout=3600
        )

        # 最適化結果の表示
        print("\n=========================================")
        print(f"Optimization {rcc_step + 1} / {rcc_num} Finished!")
        print(f"Best Score: {study.best_value:.5f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=========================================")

    # 最適化したパラメータを反映
    optimized_params = copy.deepcopy(params)
    best = study.best_params
    optimized_params["rcc_piv"]["pivstep_skip"] = 0
    optimized_params["rcc_piv"]["ref_skip_low"] = best["skip_low"]
    optimized_params["rcc_piv"]["ref_skip_high"] = best["skip_high"]
    optimized_params["rcc_piv"]["iw_size"][rcc_step] = best["iw_size"]
    optimized_params["rcc_piv"]["sw_size"][rcc_step] = (
        int(best["iw_size"] * best["sw_ratio"]) // 2
    ) * 2

    pp.pprint(optimized_params, sort_dicts=False)

    return optimized_params


if __name__ == "__main__":
    # パラメータの読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", type=str, default="-")
    args = parser.parse_args()
    with open(args.params_file, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        pp.pprint(params, sort_dicts=False)

    # piv解析実行
    optimized_params = optimize_params(params)

    optimized_params_file = "./parameters/optimized_params.yaml"
    with open(optimized_params_file, "w", encoding="utf-8") as f:
        yaml.dump(optimized_params, f, default_flow_style=False, sort_keys=False)
