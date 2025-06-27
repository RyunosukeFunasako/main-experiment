import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

# 環境変数読み込み
load_dotenv()
api_key = os.getenv("NAL_LAB_KEY")

# OpenAI 初期化
openai = OpenAI(api_key=api_key)
model = "gpt-4o-mini"

# フォルダとファイル設定
patient_dir = "../../simulated_patient_infomation/patient_data"
scenario_file = "./counselor_scenario.json"
output_dir = "./dialogue_history"
os.makedirs(output_dir, exist_ok=True)

# 対話シナリオ
with open(scenario_file, "r", encoding="utf-8") as f:
    scenario_data = json.load(f)["counselor_scenario"]

# 患者情報プロンプト生成関数
def build_patient_data_prompt(patient_data):
    profile = patient_data["プロフィール"]
    chief = patient_data["主訴"]
    illness = patient_data["現病歴"]
    suicide = patient_data["自殺企図の有無"]
    substances = patient_data["物質使用歴"]
    history = patient_data["既住歴"]
    family = patient_data["家族歴"]
    upbringing = patient_data["成育歴"]
    strengths = "、".join(patient_data["強み/長所"])
    problems = patient_data["問題リスト"]

    return f"""

【プロフィール】
年齢: {profile['年齢']}歳、性別: {profile['性別']}

【主訴】
症状: {chief['症状']}、診断: {chief['診断']}

【現病歴】
発症: {illness['発症']}、きっかけ: {illness['きっかけ']}、経過: {illness['経過']}、前回の試み: {illness['前回の試み']}

【自殺企図の有無】
有無: {suicide['有無']}、詳細: {suicide['詳細']}

【物質使用歴】
アルコール: {substances['アルコール']}、薬物: {substances['薬物']}、ニコチン: {substances['ニコチン']}

【既住歴】
精神疾患: {history['精神疾患']}、身体疾患: {history['身体疾患']}

【家族歴】
精神疾患: {family['精神疾患']}、身体疾患: {family['身体疾患']}

【成育歴】
学歴: {upbringing['学歴']}、職歴: {upbringing['職歴']}、社会活動: {upbringing['社会での活動']}、婚姻歴: {upbringing['婚姻歴']}

【強み】
{strengths}

【困りごと】
生活: {problems['生活']}、人間関係: {problems['人間関係']}、家族: {problems['家族']}、健康: {problems['健康']}、仕事・学校: {problems['仕事・学校']}、その他: {problems['その他']}
"""

# カウンセラーの発話生成関数
def generate_counselor_message(counselor_message, messages_for_counselor, messages_for_patient, openai, model, turn, scenario_data):
    counselor_message_prompt = f"""
あなたは優秀なカウンセラーエージェントです。
あなたは認知行動療法を行う初回セッションを行なっています。
カウンセラーエージェントには発話シナリオが用意されています。

発話シナリオ一覧：
{json.dumps(scenario_data, ensure_ascii=False, indent=2)}

今回のターン{turn}の発話シナリオは以下の通りです。
発話シナリオに沿って、自然な発話を行なってください。
ただし、発話シナリオに含まれない質問や提案はしないでください。

発話シナリオ：
{counselor_message}
"""
    # カウンセラーのメッセージリストを更新
    messages_for_counselor = [{"role": "system", "content": counselor_message_prompt}] + messages_for_counselor[1:]

    counselor_response = openai.chat.completions.create(
        model=model,
        messages=messages_for_counselor
    )
    counselor_reply = counselor_response.choices[0].message.content.strip()
    messages_for_counselor.append({"role": "assistant", "content": counselor_reply})
    messages_for_patient.append({"role": "user", "content": counselor_reply})

    return counselor_reply, messages_for_counselor, messages_for_patient

# 患者の応答生成関数
def generate_patient_response(messages_for_patient, messages_for_counselor, openai, model):
    patient_response = openai.chat.completions.create(
        model=model,
        messages=messages_for_patient
    )
    patient_reply = patient_response.choices[0].message.content.strip()
    messages_for_patient.append({"role": "assistant", "content": patient_reply})
    messages_for_counselor.append({"role": "user", "content": patient_reply})
    return patient_reply, messages_for_patient, messages_for_counselor

# 患者データ選択
# tester_patient_id = [2, 7, 10, 13, 15, 19, 20, 22, 27, 28, 33, 37, 38, 40, 41, 48, 49]
# tester_patient_id = [2]
tester_patient_id = [7, 10, 13, 15, 19, 20, 22, 27, 28, 33, 37, 38, 40, 41, 48, 49]

# 対話開始
for patient_id in tester_patient_id:
    patient_file = os.path.join(patient_dir, f"patient_{patient_id:03}.json")
    with open(patient_file, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    patient_prompt = """
あなたは以下のような特徴を持つ患者です。
あなたは認知行動療法を行う初回セッションに来ました。
これからあなたに対してカウンセラーエージェントが質問をします。
それに対し、あなた自身の背景をもとに会話調で自然に答えてください。
毎回の発言は1~2文に留め、役割を保ったまま話してください。
    """ + build_patient_data_prompt(patient_data)

    counselor_prompt = """
あなたは優秀なカウンセラーエージェントです。
あなたは認知行動療法を行う初回セッションを行なっています。
    """

    # 擬似ユーザー用対話履歴
    messages_for_patient = [{"role": "system", "content": patient_prompt}]

    # カウンセラー用対話履歴
    messages_for_counselor = [{"role": "system", "content": counselor_prompt}]

    # 対話履歴
    dialogue_history = {
        "patient_id": patient_id,
        "turns": []
    }

    # 全ての発話にLLMを介入
    for turn in range(len(scenario_data)):
        counselor_scenario_message = scenario_data[turn]["counselor_message"]

        counselor_reply, messages_for_counselor, messages_for_patient = generate_counselor_message(
            counselor_scenario_message, messages_for_counselor, messages_for_patient, openai, model, turn, scenario_data
        )

        patient_reply, messages_for_patient, messages_for_counselor = generate_patient_response(
            messages_for_patient, messages_for_counselor, openai, model
        )

        dialogue_history["turns"].append({
            "id": scenario_data[turn]["id"],
            "counselor_message": counselor_reply,
            "patient_reply": patient_reply
        })

    # JSONファイルとして保存
    output_dialogue_dir = "./dialogue_history"
    output_dialogue_path = os.path.join(output_dialogue_dir, f"test_{patient_id:03}.json")
    with open(output_dialogue_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_history, f, ensure_ascii=False, indent=2)
    print(f"✅ 対話履歴を保存しました: {output_dialogue_path}")

    # セッション後のCC-immediateの回答
    cc_immediate_prompt = """
あなたは以下のような特徴を持つ患者です。
あなたは認知行動療法を行う初回セッションを行いました。
これからあなたにはCC-immediateに回答してもらいます。
あなた自身の背景と対話履歴をもとに回答してください。
""" + build_patient_data_prompt(patient_data) + "\n【対話履歴】\n" + json.dumps(dialogue_history, ensure_ascii=False, indent=2)
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": cc_immediate_prompt
            },
            {
                "role": "user",
                "content": """
以下はCC-immediateの質問です。
各項目について、今終わったばかりのセッションに対してどのように感じたかを、次の尺度を使ってお答えください。

0（全くそう思わない）, 1, 2（ややそう思う）, 3, 4（かなりそう思う）, 5, 6（全くそう思う）

【Q1】
このセッションの中で、否定的に考えることが少なくなっていることに気づいた。

【Q2】
このセッションの中で、否定的な考えに気づき、それが偏っていることを認識し、状況を再評価した。

【Q3】
このセッションの中で、長い間抱えていた否定的な信念が正しくないかもしれないと気づいた。

【Q4】
このセッションの中で、自分が考えていることに注目し、よりバランスの取れた見方をするよう努めた。

【Q5】
このセッションの中で、否定的な考えが正確ではないかもしれないと考えた。

"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "cc_immediate_responses",
                    "description": "セッション後のCC-immediateの回答を出力する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Q1": {"type": "integer", "minimum": 0, "maximum": 6},
                            "Q2": {"type": "integer", "minimum": 0, "maximum": 6},
                            "Q3": {"type": "integer", "minimum": 0, "maximum": 6},
                            "Q4": {"type": "integer", "minimum": 0, "maximum": 6},
                            "Q5": {"type": "integer", "minimum": 0, "maximum": 6},
                        },
                        "required": ["Q1", "Q2", "Q3", "Q4", "Q5"]
                    }
                }
            }
        ],
        tool_choice="required"
    )
    # tool_calls による構造化データの抽出
    tool_response = response.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(tool_response)
    data["patient_id"] = patient_id

    # JSONファイルとして保存
    output_cc_immediate_dir = "./cc_immediate"
    os.makedirs(output_cc_immediate_dir, exist_ok=True)
    output_cc_immediate_path = os.path.join(output_cc_immediate_dir, f"test_{patient_id:03}.json")
    with open(output_cc_immediate_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ セッション後のCC-immediateの回答を書き出しました: {output_cc_immediate_path}")

    # セッション後のラポール尺度の回答
    rapport_scale_prompt = """
あなたは以下のような特徴を持つ患者です。
あなたは認知行動療法を行う初回セッションを行いました。
これからあなたにはラポール尺度に回答してもらいます。
あなた自身の背景と対話履歴をもとに回答してください。
""" + build_patient_data_prompt(patient_data) + "\n【対話履歴】\n" + json.dumps(dialogue_history, ensure_ascii=False, indent=2)
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": rapport_scale_prompt
            },
            {
                "role": "user",
                "content": """
以下はラポール尺度の質問です。
各項目について、1~5点で評価してください。

【Q1】
カウンセラーエージェントは私を理解した

【Q2】
カウンセラーエージェントは意欲的に見えなかった

【Q3】
カウンセラーエージェントはワクワクしていた

【Q4】
カウンセラーエージェントの動きは自然ではなかった

【Q5】
カウンセラーエージェントはフレンドリーだった

【Q6】
カウンセラーエージェントは私に注意を払っていなかった

【Q7】
カウンセラーエージェントと私は共通の目標に向かって行動した

【Q8】
カウンセラーエージェントと私の心は通っていないようだった

【Q9】
カウンセラーエージェントとの身体的なつながりを感じた

【Q10】
カウンセラーエージェントが私を信頼していると感じる

【Q11】
カウンセラーエージェントのことが理解できなかった

"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rapport_scale_responses",
                    "description": "セッション後のラポール尺度の回答を出力する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Q1": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q2": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q3": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q4": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q5": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q6": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q7": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q8": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q9": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q10": {"type": "integer", "minimum": 1, "maximum": 5},
                            "Q11": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11"]
                    }
                }
            }
        ],
        tool_choice="required"
    )
    # tool_calls による構造化データの抽出
    tool_response = response.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(tool_response)
    data["patient_id"] = patient_id

    # JSONファイルとして保存
    output_rapport_scale_dir = "./rapport_scale"
    os.makedirs(output_rapport_scale_dir, exist_ok=True)
    output_rapport_scale_path = os.path.join(output_rapport_scale_dir, f"test_{patient_id:03}.json")
    with open(output_rapport_scale_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ セッション後のラポール尺度の回答を書き出しました: {output_rapport_scale_path}")

    # セッション後の対話の質評価の回答
    dialogue_quality_prompt = """
あなたは以下のような特徴を持つ患者です。
あなたは認知行動療法を行う初回セッションを行いました。
これからあなたには対話の質を評価をしてもらいます。
あなた自身の背景と対話履歴をもとに回答してください。
""" + build_patient_data_prompt(patient_data) + "\n【対話履歴】\n" + json.dumps(dialogue_history, ensure_ascii=False, indent=2)
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": dialogue_quality_prompt
            },
            {
                "role": "user",
                "content": """
以下は対話の質評価の質問です。
各項目について、0を「全くそう思わない」、10を「全くそう思う」として、0~10点で評価してください。

【Q1】
カウンセラーエージェントの発話は人間らしく自然だった。

【Q2】
簡単に対話を続けることができた。

【Q3】
カウンセラーエージェントとの対話は楽しかった。

【Q4】
カウンセラーエージェントの発話に共感できた。

【Q5】
カウンセラーエージェントはあなたに興味を持って積極的に話そうとしていた。

【Q6】
カウンセラーエージェントの話したことは信頼できると感じた。

【Q7】
カウンセラーエージェントの個性・人となりが感じられた。

【Q8】
カウンセラーエージェントは自身の考えをもって話していると感じた。

【Q9】
カウンセラーエージェントには話したい話題があると感じた。

【Q10】
カウンセラーエージェントは感情を持っていると感じた。

【Q11】
カウンセラーエージェントの発話は矛盾せず一貫していた。

【Q12】
この対話にのめりこめた。

【Q13】
またこのカウンセラーエージェントと話したい。

【Q14】
カウンセラーエージェントは共感を示した。

【Q15】
カウンセラーエージェントは対話の主導権を握ることができた。

"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "dialogue_quality_responses",
                    "description": "セッション後の対話の質評価の回答を出力する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Q1": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q2": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q3": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q4": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q5": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q6": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q7": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q8": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q9": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q10": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q11": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q12": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q13": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q14": {"type": "integer", "minimum": 0, "maximum": 10},
                            "Q15": {"type": "integer", "minimum": 0, "maximum": 10},
                        },
                        "required": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"]
                    }
                }
            }
        ],
        tool_choice="required"
    )
    # tool_calls による構造化データの抽出
    tool_response = response.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(tool_response)
    data["patient_id"] = patient_id

    # JSONファイルとして保存
    output_dialogue_quality_dir = "./dialogue_quality"
    os.makedirs(output_dialogue_quality_dir, exist_ok=True)
    output_dialogue_quality_path = os.path.join(output_dialogue_quality_dir, f"test_{patient_id:03}.json")
    with open(output_dialogue_quality_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ セッション後の対話の質評価の回答を書き出しました: {output_dialogue_quality_path}")
    
    
    