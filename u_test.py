# scenario-basedテストとLLM+scenarioテストの結果におけるマン・ホイットニーのU検定
from scipy.stats import mannwhitneyu
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

# 患者データ選択
tester_patient_id = [2, 7, 10, 13, 15, 19, 20, 22, 27, 28, 33, 37, 38, 40, 41, 48, 49]

scenario_based_mode_scores_file = "scenario-based/mode_score/mode_scores.csv"
scenario_based_cc_immediate_dir = "scenario-based/cc_immediate"
scenario_based_rapport_scale_dir = "scenario-based/rapport_scale"
scenario_based_dialogue_quality_dir = "scenario-based/dialogue_quality"
scenario_based_ctrs_eval_dir = "scenario-based/ctrs_eval"

LLM_scenario_mode_scores_file = "LLM+scenario/mode_score/mode_scores.csv"
LLM_scenario_cc_immediate_dir = "LLM+scenario/cc_immediate"
LLM_scenario_rapport_scale_dir = "LLM+scenario/rapport_scale"
LLM_scenario_dialogue_quality_dir = "LLM+scenario/dialogue_quality"
LLM_scenario_ctrs_eval_dir = "LLM+scenario/ctrs_eval"

# 気分の変化
scenario_based_mode_changes = []
LLM_scenario_mode_changes = []

with open(scenario_based_mode_scores_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        first_score = float(row["1回目"])
        second_score = float(row["2回目"])
        if first_score != 0:  # 0除算を防ぐ
            mode_change = -((second_score - first_score) / first_score)
        else:
            mode_change = 0  # 1回目のスコアが0の場合は0とする
        scenario_based_mode_changes.append(mode_change)

with open(LLM_scenario_mode_scores_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        first_score = float(row["1回目"])
        second_score = float(row["2回目"])
        if first_score != 0:  # 0除算を防ぐ
            mode_change = -((second_score - first_score) / first_score)
        else:
            mode_change = 0  # 1回目のスコアが0の場合は0とする
        LLM_scenario_mode_changes.append(mode_change)

# 検定の実行（デフォルトは 'two-sided' のp値）
mode_change_statistic, mode_change_p_value = mannwhitneyu(scenario_based_mode_changes, LLM_scenario_mode_changes, alternative='two-sided')

# cc-immediate
scenario_based_cc_immediate_data = { f"Q{i}": [] for i in range(1, 6) }
scenario_based_cc_immediate_total_scores = []

LLM_scenario_cc_immediate_data = { f"Q{i}": [] for i in range(1, 6) }
LLM_scenario_cc_immediate_total_scores = []

for patient_id in tester_patient_id:
    scenario_based_cc_immediate_file = os.path.join(scenario_based_cc_immediate_dir, f"test_{patient_id:03}.json")
    with open(scenario_based_cc_immediate_file, "r", encoding="utf-8") as f:
        scenario_based_cc_immediate = json.load(f)
    
    # 各項目のスコアを格納
    for key in scenario_based_cc_immediate:
        if key != "patient_id":
            score = scenario_based_cc_immediate[key]
            scenario_based_cc_immediate_data[key].append(score)
    
    # 合計スコアを計算
    total_score = sum(scenario_based_cc_immediate.get(f"Q{i}", 0) for i in range(1, 6))
    scenario_based_cc_immediate_total_scores.append(total_score)

for patient_id in tester_patient_id:
    LLM_scenario_cc_immediate_file = os.path.join(LLM_scenario_cc_immediate_dir, f"test_{patient_id:03}.json")
    with open(LLM_scenario_cc_immediate_file, "r", encoding="utf-8") as f:
        LLM_scenario_cc_immediate = json.load(f)
    
    # 各項目のスコアを格納
    for key in LLM_scenario_cc_immediate:
        if key != "patient_id":
            score = LLM_scenario_cc_immediate[key]
            LLM_scenario_cc_immediate_data[key].append(score)
    
    # 合計スコアを計算
    total_score = sum(LLM_scenario_cc_immediate.get(f"Q{i}", 0) for i in range(1, 6))
    LLM_scenario_cc_immediate_total_scores.append(total_score)
    
# 項目ごとにMann-Whitney U検定を実行
cc_immediate_Q1_statistic, cc_immediate_Q1_p_value = mannwhitneyu(scenario_based_cc_immediate_data["Q1"], LLM_scenario_cc_immediate_data["Q1"], alternative='two-sided')    
cc_immediate_Q2_statistic, cc_immediate_Q2_p_value = mannwhitneyu(scenario_based_cc_immediate_data["Q2"], LLM_scenario_cc_immediate_data["Q2"], alternative='two-sided')    
cc_immediate_Q3_statistic, cc_immediate_Q3_p_value = mannwhitneyu(scenario_based_cc_immediate_data["Q3"], LLM_scenario_cc_immediate_data["Q3"], alternative='two-sided')
cc_immediate_Q4_statistic, cc_immediate_Q4_p_value = mannwhitneyu(scenario_based_cc_immediate_data["Q4"], LLM_scenario_cc_immediate_data["Q4"], alternative='two-sided')
cc_immediate_Q5_statistic, cc_immediate_Q5_p_value = mannwhitneyu(scenario_based_cc_immediate_data["Q5"], LLM_scenario_cc_immediate_data["Q5"], alternative='two-sided')
cc_immediate_statistic, cc_immediate_p_value = mannwhitneyu(scenario_based_cc_immediate_total_scores, LLM_scenario_cc_immediate_total_scores, alternative='two-sided')

# Rapport Scale
scenario_based_rapport_scale_data = { f"Q{i}": [] for i in range(1, 12) }
scenario_based_rapport_scale_total_scores = []

LLM_scenario_rapport_scale_data = { f"Q{i}": [] for i in range(1, 12) }
LLM_scenario_rapport_scale_total_scores = []

# 否定的な項目のリスト
negative_items = ["Q2", "Q4", "Q6", "Q8", "Q11"]

for patient_id in tester_patient_id:
    scenario_based_rapport_scale_file = os.path.join(scenario_based_rapport_scale_dir, f"test_{patient_id:03}.json")
    with open(scenario_based_rapport_scale_file, "r", encoding="utf-8") as f:
        scenario_based_rapport_scale = json.load(f)
    
    # 各項目のスコアを格納
    for key in scenario_based_rapport_scale:
        if key != "patient_id":
            score = scenario_based_rapport_scale[key]
            scenario_based_rapport_scale_data[key].append(score)
    
    # 合計スコアを計算（否定的な項目は反転）
    total_score = 0
    for key in scenario_based_rapport_scale:
        if key != "patient_id":
            score = scenario_based_rapport_scale[key]
            if key in negative_items:
                score = 6 - score  # スコアを反転（1→5, 2→4, 3→3, 4→2, 5→1）
            total_score += score
    scenario_based_rapport_scale_total_scores.append(total_score)

for patient_id in tester_patient_id:
    LLM_scenario_rapport_scale_file = os.path.join(LLM_scenario_rapport_scale_dir, f"test_{patient_id:03}.json")
    with open(LLM_scenario_rapport_scale_file, "r", encoding="utf-8") as f:
        LLM_scenario_rapport_scale = json.load(f)
    
    # 各項目のスコアを格納
    for key in LLM_scenario_rapport_scale:
        if key != "patient_id":
            score = LLM_scenario_rapport_scale[key]
            LLM_scenario_rapport_scale_data[key].append(score)
    
    # 合計スコアを計算（否定的な項目は反転）
    total_score = 0
    for key in LLM_scenario_rapport_scale:
        if key != "patient_id":
            score = LLM_scenario_rapport_scale[key]
            if key in negative_items:
                score = 6 - score  # スコアを反転（1→5, 2→4, 3→3, 4→2, 5→1）
            total_score += score
    LLM_scenario_rapport_scale_total_scores.append(total_score)
    
# 項目ごとにMann-Whitney U検定を実行
rapport_scale_Q1_statistic, rapport_scale_Q1_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q1"], LLM_scenario_rapport_scale_data["Q1"], alternative='two-sided')
rapport_scale_Q2_statistic, rapport_scale_Q2_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q2"], LLM_scenario_rapport_scale_data["Q2"], alternative='two-sided')
rapport_scale_Q3_statistic, rapport_scale_Q3_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q3"], LLM_scenario_rapport_scale_data["Q3"], alternative='two-sided')
rapport_scale_Q4_statistic, rapport_scale_Q4_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q4"], LLM_scenario_rapport_scale_data["Q4"], alternative='two-sided')
rapport_scale_Q5_statistic, rapport_scale_Q5_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q5"], LLM_scenario_rapport_scale_data["Q5"], alternative='two-sided')
rapport_scale_Q6_statistic, rapport_scale_Q6_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q6"], LLM_scenario_rapport_scale_data["Q6"], alternative='two-sided')
rapport_scale_Q7_statistic, rapport_scale_Q7_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q7"], LLM_scenario_rapport_scale_data["Q7"], alternative='two-sided')
rapport_scale_Q8_statistic, rapport_scale_Q8_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q8"], LLM_scenario_rapport_scale_data["Q8"], alternative='two-sided')
rapport_scale_Q9_statistic, rapport_scale_Q9_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q9"], LLM_scenario_rapport_scale_data["Q9"], alternative='two-sided')
rapport_scale_Q10_statistic, rapport_scale_Q10_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q10"], LLM_scenario_rapport_scale_data["Q10"], alternative='two-sided')
rapport_scale_Q11_statistic, rapport_scale_Q11_p_value = mannwhitneyu(scenario_based_rapport_scale_data["Q11"], LLM_scenario_rapport_scale_data["Q11"], alternative='two-sided')
rapport_scale_statistic, rapport_scale_p_value = mannwhitneyu(scenario_based_rapport_scale_total_scores, LLM_scenario_rapport_scale_total_scores, alternative='two-sided')

# Dialogue Quality
scenario_based_dialogue_quality_data = { f"Q{i}": [] for i in range(1, 16) }
scenario_based_dialogue_quality_total_scores = []

LLM_scenario_dialogue_quality_data = { f"Q{i}": [] for i in range(1, 16) }
LLM_scenario_dialogue_quality_total_scores = []

for patient_id in tester_patient_id:
    scenario_based_dialogue_quality_file = os.path.join(scenario_based_dialogue_quality_dir, f"test_{patient_id:03}.json")
    with open(scenario_based_dialogue_quality_file, "r", encoding="utf-8") as f:
        scenario_based_dialogue_quality = json.load(f)
    
    # 各項目のスコアを格納
    for key in scenario_based_dialogue_quality:
        if key != "patient_id":
            score = scenario_based_dialogue_quality[key]
            scenario_based_dialogue_quality_data[key].append(score)
    
    # 合計スコアを計算
    total_score = sum(scenario_based_dialogue_quality.get(f"Q{i}", 0) for i in range(1, 16))
    scenario_based_dialogue_quality_total_scores.append(total_score)

for patient_id in tester_patient_id:
    LLM_scenario_dialogue_quality_file = os.path.join(LLM_scenario_dialogue_quality_dir, f"test_{patient_id:03}.json")
    with open(LLM_scenario_dialogue_quality_file, "r", encoding="utf-8") as f:
        LLM_scenario_dialogue_quality = json.load(f)
    
    # 各項目のスコアを格納
    for key in LLM_scenario_dialogue_quality:
        if key != "patient_id":
            score = LLM_scenario_dialogue_quality[key]
            LLM_scenario_dialogue_quality_data[key].append(score)
    
    # 合計スコアを計算
    total_score = sum(LLM_scenario_dialogue_quality.get(f"Q{i}", 0) for i in range(1, 16))
    LLM_scenario_dialogue_quality_total_scores.append(total_score)
    
# 項目ごとにMann-Whitney U検定を実行
dialogue_quality_Q1_statistic, dialogue_quality_Q1_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q1"], LLM_scenario_dialogue_quality_data["Q1"], alternative='two-sided')
dialogue_quality_Q2_statistic, dialogue_quality_Q2_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q2"], LLM_scenario_dialogue_quality_data["Q2"], alternative='two-sided')
dialogue_quality_Q3_statistic, dialogue_quality_Q3_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q3"], LLM_scenario_dialogue_quality_data["Q3"], alternative='two-sided')
dialogue_quality_Q4_statistic, dialogue_quality_Q4_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q4"], LLM_scenario_dialogue_quality_data["Q4"], alternative='two-sided')
dialogue_quality_Q5_statistic, dialogue_quality_Q5_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q5"], LLM_scenario_dialogue_quality_data["Q5"], alternative='two-sided')
dialogue_quality_Q6_statistic, dialogue_quality_Q6_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q6"], LLM_scenario_dialogue_quality_data["Q6"], alternative='two-sided')
dialogue_quality_Q7_statistic, dialogue_quality_Q7_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q7"], LLM_scenario_dialogue_quality_data["Q7"], alternative='two-sided')
dialogue_quality_Q8_statistic, dialogue_quality_Q8_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q8"], LLM_scenario_dialogue_quality_data["Q8"], alternative='two-sided')
dialogue_quality_Q9_statistic, dialogue_quality_Q9_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q9"], LLM_scenario_dialogue_quality_data["Q9"], alternative='two-sided')
dialogue_quality_Q10_statistic, dialogue_quality_Q10_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q10"], LLM_scenario_dialogue_quality_data["Q10"], alternative='two-sided')
dialogue_quality_Q11_statistic, dialogue_quality_Q11_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q11"], LLM_scenario_dialogue_quality_data["Q11"], alternative='two-sided')
dialogue_quality_Q12_statistic, dialogue_quality_Q12_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q12"], LLM_scenario_dialogue_quality_data["Q12"], alternative='two-sided')
dialogue_quality_Q13_statistic, dialogue_quality_Q13_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q13"], LLM_scenario_dialogue_quality_data["Q13"], alternative='two-sided')
dialogue_quality_Q14_statistic, dialogue_quality_Q14_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q14"], LLM_scenario_dialogue_quality_data["Q14"], alternative='two-sided')
dialogue_quality_Q15_statistic, dialogue_quality_Q15_p_value = mannwhitneyu(scenario_based_dialogue_quality_data["Q15"], LLM_scenario_dialogue_quality_data["Q15"], alternative='two-sided')
dialogue_quality_statistic, dialogue_quality_p_value = mannwhitneyu(scenario_based_dialogue_quality_total_scores, LLM_scenario_dialogue_quality_total_scores, alternative='two-sided')
    
# CTRS
scenario_based_ctrs_eval_data = { f"Q{i}": [] for i in range(1, 12) }
scenario_based_ctrs_eval_total_scores = []

LLM_scenario_ctrs_eval_data = { f"Q{i}": [] for i in range(1, 12) }
LLM_scenario_ctrs_eval_total_scores = []

for patient_id in tester_patient_id:
    scenario_based_ctrs_eval_file = os.path.join(scenario_based_ctrs_eval_dir, f"test_{patient_id:03}.json")
    with open(scenario_based_ctrs_eval_file, "r", encoding="utf-8") as f:
        scenario_based_ctrs_eval = json.load(f)

    # 各項目のスコアを格納
    ratings = scenario_based_ctrs_eval.get("ratings", {})
    for key in range(1, 12):  # 1から11までの項目
        score = ratings.get(str(key), 0)  # キーは文字列として保存されている
        scenario_based_ctrs_eval_data[f"Q{key}"].append(score)
    
    # 合計スコアを計算
    total_score = sum(ratings.get(str(key), 0) for key in range(1, 12))
    scenario_based_ctrs_eval_total_scores.append(total_score)

for patient_id in tester_patient_id:
    LLM_scenario_ctrs_eval_file = os.path.join(LLM_scenario_ctrs_eval_dir, f"test_{patient_id:03}.json")
    with open(LLM_scenario_ctrs_eval_file, "r", encoding="utf-8") as f:
        LLM_scenario_ctrs_eval = json.load(f)
    
    # 各項目のスコアを格納
    ratings = LLM_scenario_ctrs_eval.get("ratings", {})
    for key in range(1, 12):  # 1から11までの項目
        score = ratings.get(str(key), 0)  # キーは文字列として保存されている
        LLM_scenario_ctrs_eval_data[f"Q{key}"].append(score)
    
    # 合計スコアを計算
    total_score = sum(ratings.get(str(key), 0) for key in range(1, 12))
    LLM_scenario_ctrs_eval_total_scores.append(total_score)
    
# 項目ごとにMann-Whitney U検定を実行
ctrs_eval_Q1_statistic, ctrs_eval_Q1_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q1"], LLM_scenario_ctrs_eval_data["Q1"], alternative='two-sided')
ctrs_eval_Q2_statistic, ctrs_eval_Q2_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q2"], LLM_scenario_ctrs_eval_data["Q2"], alternative='two-sided')
ctrs_eval_Q3_statistic, ctrs_eval_Q3_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q3"], LLM_scenario_ctrs_eval_data["Q3"], alternative='two-sided')
ctrs_eval_Q4_statistic, ctrs_eval_Q4_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q4"], LLM_scenario_ctrs_eval_data["Q4"], alternative='two-sided')
ctrs_eval_Q5_statistic, ctrs_eval_Q5_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q5"], LLM_scenario_ctrs_eval_data["Q5"], alternative='two-sided')
ctrs_eval_Q6_statistic, ctrs_eval_Q6_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q6"], LLM_scenario_ctrs_eval_data["Q6"], alternative='two-sided')
ctrs_eval_Q7_statistic, ctrs_eval_Q7_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q7"], LLM_scenario_ctrs_eval_data["Q7"], alternative='two-sided')
ctrs_eval_Q8_statistic, ctrs_eval_Q8_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q8"], LLM_scenario_ctrs_eval_data["Q8"], alternative='two-sided')
ctrs_eval_Q9_statistic, ctrs_eval_Q9_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q9"], LLM_scenario_ctrs_eval_data["Q9"], alternative='two-sided')
ctrs_eval_Q10_statistic, ctrs_eval_Q10_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q10"], LLM_scenario_ctrs_eval_data["Q10"], alternative='two-sided')
ctrs_eval_Q11_statistic, ctrs_eval_Q11_p_value = mannwhitneyu(scenario_based_ctrs_eval_data["Q11"], LLM_scenario_ctrs_eval_data["Q11"], alternative='two-sided')
ctrs_eval_statistic, ctrs_eval_p_value = mannwhitneyu(scenario_based_ctrs_eval_total_scores, LLM_scenario_ctrs_eval_total_scores, alternative='two-sided')

# 結果保存
csv_path = os.path.join(result_dir, "mannwhitneyu_result.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # 気分の変化
    writer.writerow(["### 気分の変化"])
    writer.writerow(["U統計量", "p値"])
    writer.writerow([mode_change_statistic, mode_change_p_value])
    writer.writerow([])

    # cc-immediate
    writer.writerow(["### cc-immediate"])
    writer.writerow(["項目", "U統計量", "p値"])
    writer.writerow(["Q1", cc_immediate_Q1_statistic, cc_immediate_Q1_p_value])
    writer.writerow(["Q2", cc_immediate_Q2_statistic, cc_immediate_Q2_p_value])
    writer.writerow(["Q3", cc_immediate_Q3_statistic, cc_immediate_Q3_p_value])
    writer.writerow(["Q4", cc_immediate_Q4_statistic, cc_immediate_Q4_p_value])
    writer.writerow(["Q5", cc_immediate_Q5_statistic, cc_immediate_Q5_p_value])
    writer.writerow(["合計", cc_immediate_statistic, cc_immediate_p_value])
    writer.writerow([])

    # Rapport Scale
    writer.writerow(["### Rapport Scale"])
    writer.writerow(["項目", "U統計量", "p値"])
    writer.writerow(["Q1", rapport_scale_Q1_statistic, rapport_scale_Q1_p_value])
    writer.writerow(["Q2", rapport_scale_Q2_statistic, rapport_scale_Q2_p_value])
    writer.writerow(["Q3", rapport_scale_Q3_statistic, rapport_scale_Q3_p_value])
    writer.writerow(["Q4", rapport_scale_Q4_statistic, rapport_scale_Q4_p_value])
    writer.writerow(["Q5", rapport_scale_Q5_statistic, rapport_scale_Q5_p_value])
    writer.writerow(["Q6", rapport_scale_Q6_statistic, rapport_scale_Q6_p_value])
    writer.writerow(["Q7", rapport_scale_Q7_statistic, rapport_scale_Q7_p_value])
    writer.writerow(["Q8", rapport_scale_Q8_statistic, rapport_scale_Q8_p_value])
    writer.writerow(["Q9", rapport_scale_Q9_statistic, rapport_scale_Q9_p_value])
    writer.writerow(["Q10", rapport_scale_Q10_statistic, rapport_scale_Q10_p_value])
    writer.writerow(["Q11", rapport_scale_Q11_statistic, rapport_scale_Q11_p_value])
    writer.writerow(["合計", rapport_scale_statistic, rapport_scale_p_value])
    writer.writerow([])

    # Dialogue Quality
    writer.writerow(["### Dialogue Quality"])
    writer.writerow(["項目", "U統計量", "p値"])
    writer.writerow(["Q1", dialogue_quality_Q1_statistic, dialogue_quality_Q1_p_value])
    writer.writerow(["Q2", dialogue_quality_Q2_statistic, dialogue_quality_Q2_p_value])
    writer.writerow(["Q3", dialogue_quality_Q3_statistic, dialogue_quality_Q3_p_value])
    writer.writerow(["Q4", dialogue_quality_Q4_statistic, dialogue_quality_Q4_p_value])
    writer.writerow(["Q5", dialogue_quality_Q5_statistic, dialogue_quality_Q5_p_value])
    writer.writerow(["Q6", dialogue_quality_Q6_statistic, dialogue_quality_Q6_p_value])
    writer.writerow(["Q7", dialogue_quality_Q7_statistic, dialogue_quality_Q7_p_value])
    writer.writerow(["Q8", dialogue_quality_Q8_statistic, dialogue_quality_Q8_p_value])
    writer.writerow(["Q9", dialogue_quality_Q9_statistic, dialogue_quality_Q9_p_value])
    writer.writerow(["Q10", dialogue_quality_Q10_statistic, dialogue_quality_Q10_p_value])
    writer.writerow(["Q11", dialogue_quality_Q11_statistic, dialogue_quality_Q11_p_value])
    writer.writerow(["Q12", dialogue_quality_Q12_statistic, dialogue_quality_Q12_p_value])
    writer.writerow(["Q13", dialogue_quality_Q13_statistic, dialogue_quality_Q13_p_value])
    writer.writerow(["Q14", dialogue_quality_Q14_statistic, dialogue_quality_Q14_p_value])
    writer.writerow(["Q15", dialogue_quality_Q15_statistic, dialogue_quality_Q15_p_value])
    writer.writerow(["合計", dialogue_quality_statistic, dialogue_quality_p_value])
    writer.writerow([])

    # CTRS
    writer.writerow(["### CTRS"])
    writer.writerow(["項目", "U統計量", "p値"])
    writer.writerow(["Q1", ctrs_eval_Q1_statistic, ctrs_eval_Q1_p_value])
    writer.writerow(["Q2", ctrs_eval_Q2_statistic, ctrs_eval_Q2_p_value])
    writer.writerow(["Q3", ctrs_eval_Q3_statistic, ctrs_eval_Q3_p_value])
    writer.writerow(["Q4", ctrs_eval_Q4_statistic, ctrs_eval_Q4_p_value])
    writer.writerow(["Q5", ctrs_eval_Q5_statistic, ctrs_eval_Q5_p_value])
    writer.writerow(["Q6", ctrs_eval_Q6_statistic, ctrs_eval_Q6_p_value])
    writer.writerow(["Q7", ctrs_eval_Q7_statistic, ctrs_eval_Q7_p_value])
    writer.writerow(["Q8", ctrs_eval_Q8_statistic, ctrs_eval_Q8_p_value])
    writer.writerow(["Q9", ctrs_eval_Q9_statistic, ctrs_eval_Q9_p_value])
    writer.writerow(["Q10", ctrs_eval_Q10_statistic, ctrs_eval_Q10_p_value])
    writer.writerow(["Q11", ctrs_eval_Q11_statistic, ctrs_eval_Q11_p_value])
    writer.writerow(["合計", ctrs_eval_statistic, ctrs_eval_p_value])
print(f"✅ Mann-Whitney U検定の結果を保存しました: {csv_path}")