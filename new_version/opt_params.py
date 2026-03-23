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
import datetime

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


def objective(trial, base_params, n_trials, timeout, rcc_step=0):
    """
    Oputunaが探索を行うための目的関数
    """

    # 探索パラメータの候補
    list_iw_size = [104, 80, 56, 32, 8]

    # 2. 探索するパラメータの定義
    ref_skip_low = trial.suggest_int(
        "skip_low",
        base_params["rcc_piv"]["ref_skip_high"] + 1,
        base_params["rcc_piv"]["ref_skip_high"] + 6,
    )  # 低速域の参照間隔
    iw_size = trial.suggest_int(
        "iw_size",
        list_iw_size[rcc_step],
        list_iw_size[rcc_step] + 20,
        step=2,
    )  # 検査領域サイズ
    sw_ratio = trial.suggest_float("sw_ratio", 1.25, 2.5, step=0.25)  # 探査領域サイズ
    sw_size = (int(iw_size * sw_ratio) // 2) * 2

    # パラメータの上書き
    base_params["rcc_piv"]["rcc_num"] = (
        rcc_step + 1
    )  # 再帰的処理のステップをrcc_stepに対応
    base_params["rcc_piv"]["ref_skip_low"] = ref_skip_low
    # params["rcc_piv"]["ref_skip_high"] = ref_skip_high
    base_params["rcc_piv"]["iw_size"][rcc_step] = iw_size
    base_params["rcc_piv"]["sw_size"][rcc_step] = sw_size
    base_params["rcc_piv"]["margin_size"][rcc_step] = int(iw_size // 2) + 5
    pp.pprint(base_params, sort_dicts=False)

    try:
        # 3. PIV実行と評価
        with open(os.devnull, "w") as f, redirect_stdout(f):
            error_ratio_all, sn_ratio_all, dynamic_range_all, spatial_variations_all = (
                rcc_piv(base_params, mode="optimize")
            )
        # error_ratio_all, sn_ratio_all = rcc_piv(base_params, mode="optimize")

        # 総合スコアの算出
        error_ratio = np.mean(error_ratio_all)
        sn_ratio = np.mean(sn_ratio_all)
        dynamic_range = np.mean(dynamic_range_all)
        spatial_variations = np.mean(spatial_variations_all)

        score = (
            error_ratio * 100
            + sn_ratio ** (-1) * 100
            + dynamic_range ** (-1) * 100
            + spatial_variations / 10
        )

        # score = (
        #     np.log(error_ratio + 1e-6)
        #     - np.log(sn_ratio)
        #     - np.log(dynamic_range)
        #     + np.log(spatial_variations + 1e-6)
        # )

        # 最適化（Study）全体の開始時刻を取得
        start_time = trial.study.trials[0].datetime_start
        # 現在時刻との差分を計算
        now = datetime.datetime.now()
        elapsed_time = (now - start_time).total_seconds()

        print(f"- 試行回数 : {trial.number + 1} / {n_trials}")
        print(f"- 経過時間 : {elapsed_time:.4f} / {timeout} [sec]")
        print(f"- 誤ベクトル率 : {error_ratio}")
        print(f"- SN比 : {sn_ratio}")
        print(f"- 速度のダイナミックレンジ（max / min） : {dynamic_range}")
        print(f"- 速度場の滑らかさ : {spatial_variations}")
        print(f"- 総スコア : {score:.4f}")

    except Exception as e:
        print(f"Trial {trial.number} / {n_trials} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    return score


# pivの解析パラメータをベイズ推定で最適化する関数
def optimize_params(params, n_trials=5, timeout=99999):
    print("\nPIVの解析パラメータを最適化します...")

    # 最適化後のパラメータを保存する変数
    optimized_params = copy.deepcopy(params)

    # 再帰的処理のステップごとに最適化
    rcc_num = params["rcc_piv"]["rcc_num"]
    for rcc_step in range(rcc_num):
        print(f"\nrcc_step: {rcc_step + 1} / {rcc_num}を最適化中...")

        # Optunaのスタディを作成
        study = optuna.create_study(direction="minimize")

        # 最適化の実行
        study.optimize(
            lambda trial: objective(
                trial,
                optimized_params,
                rcc_step=rcc_step,
                n_trials=n_trials,
                timeout=timeout,
            ),
            n_trials=n_trials,
            timeout=timeout,
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
        best = study.best_params
        optimized_params["rcc_piv"]["ref_skip_low"] = best["skip_low"]
        optimized_params["rcc_piv"]["iw_size"][rcc_step] = best["iw_size"]
        optimized_params["rcc_piv"]["sw_size"][rcc_step] = (
            int(best["iw_size"] * best["sw_ratio"]) // 2
        ) * 2
        optimized_params["rcc_piv"]["margin_size"][rcc_step] = (
            int(optimized_params["rcc_piv"]["iw_size"][rcc_step] // 2) + 5
        )

    pp.pprint(optimized_params, sort_dicts=False)

    return optimized_params


if __name__ == "__main__":
    # パラメータの読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", type=str, default="-")
    args = parser.parse_args()
    with open(args.params_file, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        print("\n[デフォルトのパラメータ一覧]")
        pp.pprint(params, sort_dicts=False)

    # piv解析実行
    n_trials = 100
    timeout = 999
    optimized_params = optimize_params(params, n_trials=n_trials, timeout=timeout)

    optimized_params_file = "./parameters/optimized_params.yaml"
    with open(optimized_params_file, "w", encoding="utf-8") as f:
        yaml.dump(optimized_params, f, default_flow_style=False, sort_keys=False)
