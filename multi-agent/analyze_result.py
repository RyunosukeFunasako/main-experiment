import os
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import Counter

# ディレクトリ設定
result_dir = "./result"
os.makedirs(result_dir, exist_ok=True)

mode_scores_file = "./mode_score/mode_scores.csv"
cc_immediate_dir = "./cc_immediate"
rapport_scale_dir = "./rapport_scale"
dialogue_quality_dir = "./dialogue_quality"
# ctrs_eval_dir = "./ctrs_eval"

# Hiraginoフォントを指定
plt.rcParams['font.family'] = 'Hiragino Sans'

# 患者データ選択
tester_patient_id = [2, 7, 10, 13, 15, 19, 20, 22, 27, 28, 33, 37, 38, 40, 41, 48, 49]

# Mode Changeの計算
mode_changes = []
with open(mode_scores_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        first_score = float(row["1回目"])
        second_score = float(row["2回目"])
        if first_score != 0:  # 0除算を防ぐ
            mode_change = -((second_score - first_score) / first_score)
        else:
            mode_change = 0  # 1回目のスコアが0の場合は0とする
        mode_changes.append(mode_change)

# Mode Changeの統計量を計算
mode_mean = np.mean(mode_changes)
mode_var = np.var(mode_changes)
mode_std = np.std(mode_changes)

# Mode ChangeのCSV保存
csv_path = os.path.join(result_dir, "mode_change.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["テスト番号", "気分強度の変化"])
    for patient_id, change in zip(tester_patient_id, mode_changes):
        writer.writerow([f"test_{patient_id:03}", change])
    writer.writerow([])  # 空行を追加
    writer.writerow(["統計量", "値"])
    writer.writerow(["平均", mode_mean])
    writer.writerow(["分散", mode_var])
    writer.writerow(["標準偏差", mode_std])
print(f"✅ Mode ChangeのCSVファイルを保存しました: {csv_path}")

# Mode Changeの分布
plt.figure(figsize=(10, 6))
plt.hist(mode_changes, bins=50, alpha=0.7)
plt.title("気分強度の変化の分布")
plt.xlabel("気分強度の変化")
plt.ylabel("人数")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_dir, "mode_change_histogram.png"))
plt.close()

# cc-immediateの分析
cc_immediate_data = { f"Q{i}": [] for i in range(1, 6) }
cc_immediate_total_scores = []

for patient_id in tester_patient_id:
    cc_immediate_file = os.path.join(cc_immediate_dir, f"test_{patient_id:03}.json")
    with open(cc_immediate_file, "r", encoding="utf-8") as f:
        cc_immediate = json.load(f)
    
    # 各項目のスコアを格納
    for key in cc_immediate:
        if key != "patient_id":
            score = cc_immediate[key]
            cc_immediate_data[key].append(score)
    
    # 合計スコアを計算
    total_score = sum(cc_immediate.get(f"Q{i}", 0) for i in range(1, 6))
    cc_immediate_total_scores.append(total_score)

# cc-immediateの統計量を計算
cc_immediate_mean = np.mean(cc_immediate_total_scores)
cc_immediate_var = np.var(cc_immediate_total_scores)
cc_immediate_std = np.std(cc_immediate_total_scores)

# cc-immediateの平均値CSV保存
csv_path = os.path.join(result_dir, "cc_immediate.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for key, values in cc_immediate_data.items():
        writer.writerow([key, sum(values)/len(values)])
    writer.writerow(["合計スコア（平均）", cc_immediate_mean])
    writer.writerow(["合計スコア（分散）", cc_immediate_var])
    writer.writerow(["合計スコア（標準偏差）", cc_immediate_std])
print(f"✅ cc-immediateのCSVファイルを保存しました: {csv_path}")

# cc-immediateの箱ひげ図
plt.figure(figsize=(10, 5))
plt.boxplot(list(cc_immediate_data.values()), labels=list(cc_immediate_data.keys()))
plt.title("cc-immediate 各項目ごとのスコアの分布")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "cc_immediate_boxplot.png"))
plt.close()

# cc-immediateの合計スコアの分布
plt.figure(figsize=(10, 6))
plt.hist(cc_immediate_total_scores, bins=range(15, 30, 1), alpha=0.7)
plt.title("cc-immediate 合計スコアの分布")
plt.xlabel("合計スコア")
plt.ylabel("人数")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_dir, "cc_immediate_histogram.png"))
plt.close()

# cc-immediateの合計スコアの詳細CSV保存
csv_path = os.path.join(result_dir, "cc_immediate_total_scores.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["テスト番号", "合計スコア"])
    for patient_id, score in zip(tester_patient_id, cc_immediate_total_scores):
        writer.writerow([f"test_{patient_id:03}", score])
print(f"✅ cc-immediateの合計スコアの詳細CSVファイルを保存しました: {csv_path}")

# Rapport Scaleの分析
rapport_scale_data = { f"Q{i}": [] for i in range(1, 12) }
rapport_total_scores = []

# 否定的な項目のリスト
negative_items = ["Q2", "Q4", "Q6", "Q8", "Q11"]

for patient_id in tester_patient_id:
    rapport_scale_file = os.path.join(rapport_scale_dir, f"test_{patient_id:03}.json")
    with open(rapport_scale_file, "r", encoding="utf-8") as f:
        rapport_scale = json.load(f)
    
    # 各項目のスコアを格納
    for key in rapport_scale:
        if key != "patient_id":
            score = rapport_scale[key]
            rapport_scale_data[key].append(score)
    
    # 合計スコアを計算（否定的な項目は反転）
    total_score = 0
    for key in rapport_scale:
        if key != "patient_id":
            score = rapport_scale[key]
            if key in negative_items:
                score = 6 - score  # スコアを反転（1→5, 2→4, 3→3, 4→2, 5→1）
            total_score += score
    rapport_total_scores.append(total_score)

# Rapport Scaleの統計量を計算
rapport_mean = np.mean(rapport_total_scores)
rapport_var = np.var(rapport_total_scores)
rapport_std = np.std(rapport_total_scores)

# Rapport Scaleの平均値CSV保存
csv_path = os.path.join(result_dir, "rapport_scale.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for key, values in rapport_scale_data.items():
        writer.writerow([key, sum(values)/len(values)])
    writer.writerow(["合計スコア（平均）", rapport_mean])
    writer.writerow(["合計スコア（分散）", rapport_var])
    writer.writerow(["合計スコア（標準偏差）", rapport_std])
print(f"✅ Rapport ScaleのCSVファイルを保存しました: {csv_path}")

# Rapport Scaleの箱ひげ図
plt.figure(figsize=(10, 5))
plt.boxplot([rapport_scale_data[key] for key in rapport_scale_data], 
            labels=rapport_scale_data.keys(), vert=True)
plt.title("Rapport Scale 各項目ごとのスコアの分布")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "rapport_scale_boxplot.png"))
plt.close()

# Rapport Scale合計スコアの分布
plt.figure(figsize=(10, 6))
plt.hist(rapport_total_scores, bins=range(40, 55, 1), alpha=0.7)  # 最小値11（全項目1点）から最大値55（全項目5点）まで 
plt.title("Rapport Scale 合計スコアの分布")
plt.xlabel("合計スコア")
plt.ylabel("人数")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_dir, "rapport_scale_histogram.png"))
plt.close()

# Rapport Scale合計スコアの詳細CSV保存
csv_path = os.path.join(result_dir, "rapport_scale_total_scores.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["テスト番号", "合計スコア"])
    for patient_id, score in zip(tester_patient_id, rapport_total_scores):
        writer.writerow([f"test_{patient_id:03}", score])
print(f"✅ Rapport Scale合計スコアの詳細CSVファイルを保存しました: {csv_path}")

# Dialogue Qualityの分析
dialogue_quality_data = { f"Q{i}": [] for i in range(1, 16) }
dialogue_quality_total_scores = []

for patient_id in tester_patient_id:
    dialogue_quality_file = os.path.join(dialogue_quality_dir, f"test_{patient_id:03}.json")
    with open(dialogue_quality_file, "r", encoding="utf-8") as f:
        dialogue_quality = json.load(f)
    
    # 各項目のスコアを格納
    for key in dialogue_quality:
        if key != "patient_id":
            score = dialogue_quality[key]
            dialogue_quality_data[key].append(score)
    
    # 合計スコアを計算
    total_score = sum(dialogue_quality.get(f"Q{i}", 0) for i in range(1, 16))
    dialogue_quality_total_scores.append(total_score)

# Dialogue Qualityの統計量を計算
dialogue_quality_mean = np.mean(dialogue_quality_total_scores)
dialogue_quality_var = np.var(dialogue_quality_total_scores)
dialogue_quality_std = np.std(dialogue_quality_total_scores)

# Dialogue Qualityの平均値CSV保存
csv_path = os.path.join(result_dir, "dialogue_quality.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for key, values in dialogue_quality_data.items():
        writer.writerow([key, sum(values)/len(values)])
    writer.writerow(["合計スコア（平均）", dialogue_quality_mean])
    writer.writerow(["合計スコア（分散）", dialogue_quality_var])
    writer.writerow(["合計スコア（標準偏差）", dialogue_quality_std])
print(f"✅ Dialogue QualityのCSVファイルを保存しました: {csv_path}")

# Dialogue Qualityの箱ひげ図
plt.figure(figsize=(10, 5))
plt.boxplot([dialogue_quality_data[key] for key in dialogue_quality_data], 
            labels=dialogue_quality_data.keys(), vert=True)
plt.title("Dialogue Quality 各項目ごとのスコアの分布")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "dialogue_quality_boxplot.png"))
plt.close()

# Dialogue Quality合計スコアの分布
plt.figure(figsize=(10, 6))
plt.hist(dialogue_quality_total_scores, bins=range(100, 150, 1), alpha=0.7)  # 最小値0から最大値79（全項目7点）まで
plt.title("Dialogue Quality 合計スコアの分布")
plt.xlabel("合計スコア")
plt.ylabel("人数")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_dir, "dialogue_quality_histogram.png"))
plt.close()

# Dialogue Quality合計スコアの詳細CSV保存
csv_path = os.path.join(result_dir, "dialogue_quality_total_scores.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["テスト番号", "合計スコア"])
    for patient_id, score in zip(tester_patient_id, dialogue_quality_total_scores):
        writer.writerow([f"test_{patient_id:03}", score])
print(f"✅ Dialogue Quality合計スコアの詳細CSVファイルを保存しました: {csv_path}")

# CTRSの分析
# ctrs_data = { f"Q{i}": [] for i in range(1, 12) }  # CTRSは11項目
# ctrs_total_scores = []  # 合計スコアを格納するリスト

# for patient_id in tester_patient_id:
#     ctrs_file = os.path.join(ctrs_eval_dir, f"test_{patient_id:03}.json")
#     with open(ctrs_file, "r", encoding="utf-8") as f:
#         ctrs = json.load(f)
    
#     # 各項目のスコアを格納
#     ratings = ctrs.get("ratings", {})
#     for key in range(1, 12):  # 1から11までの項目
#         score = ratings.get(str(key), 0)  # キーは文字列として保存されている
#         ctrs_data[f"Q{key}"].append(score)
    
#     # 合計スコアを計算
#     total_score = sum(ratings.get(str(key), 0) for key in range(1, 12))
#     ctrs_total_scores.append(total_score)

# # CTRSの統計量を計算
# ctrs_mean = np.mean(ctrs_total_scores)
# ctrs_var = np.var(ctrs_total_scores)
# ctrs_std = np.std(ctrs_total_scores)

# # CTRSの平均値CSV保存
# csv_path = os.path.join(result_dir, "ctrs.csv")
# with open(csv_path, "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Metric", "Value"])
#     for key, values in ctrs_data.items():
#         writer.writerow([key, sum(values)/len(values)])
#     writer.writerow(["合計スコア（平均）", ctrs_mean])
#     writer.writerow(["合計スコア（分散）", ctrs_var])
#     writer.writerow(["合計スコア（標準偏差）", ctrs_std])
# print(f"✅ CTRSのCSVファイルを保存しました: {csv_path}")

# # CTRSの箱ひげ図
# plt.figure(figsize=(10, 5))
# plt.boxplot([ctrs_data[key] for key in ctrs_data], 
#             labels=ctrs_data.keys(), vert=True)
# plt.title("CTRS 各項目ごとのスコアの分布")
# plt.tight_layout()
# plt.savefig(os.path.join(result_dir, "ctrs_boxplot.png"))
# plt.close()

# # CTRS合計スコアの分布
# plt.figure(figsize=(10, 6))
# plt.hist(ctrs_total_scores, bins=range(0, 67, 1), alpha=0.7)  # 最小値0から最大値66（全項目6点）まで
# plt.title("CTRS 合計スコアの分布")
# plt.xlabel("合計スコア")
# plt.ylabel("人数")
# plt.grid(True, alpha=0.3)
# plt.savefig(os.path.join(result_dir, "ctrs_histogram.png"))
# plt.close()

# # CTRS合計スコアの詳細CSV保存
# csv_path = os.path.join(result_dir, "ctrs_total_scores.csv")
# with open(csv_path, "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["テスト番号", "合計スコア"])
#     for patient_id, score in zip(tester_patient_id, ctrs_total_scores):
#         writer.writerow([f"test_{patient_id:03}", score])
# print(f"✅ CTRS合計スコアのCSVファイルを保存しました: {csv_path}")