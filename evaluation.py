import os
import torch
import json
import re
import time
import numpy as np
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
load_in_4bit = True

# --------------------------------------------------------------------------------
# 1. 설정 및 파일 경로
# --------------------------------------------------------------------------------
# MODEL_ID = "lora_model"  # 학습된 모델 경로
MODEL_ID = "/home/smslab/PycharmProjects/smart_factory/Phi-4"
TEST_FILE_PATH = "factory_dataset_evaluation_104.jsonl"  # 테스트 데이터 파일명


# --------------------------------------------------------------------------------
# 2. 유틸리티 함수 (JSON 추출, Flatten, Metrics)
# --------------------------------------------------------------------------------

def extract_json_from_text(text):
    """
    텍스트(모델 출력 or 정답 데이터)에서 JSON 블록만 추출합니다.
    (thought 부분은 버리고 ```json ... ``` 내부만 가져옵니다)
    """
    try:
        # 1. ```json ... ``` 패턴 시도
        match = re.search(r'```json(.*?)```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        # 2. ``` ... ``` 패턴 시도
        match = re.search(r'```(.*?)```', text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(json_str[start:end])

        # 3. 그냥 { ... } 패턴 시도
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return None
    except Exception as e:
        return None


def flatten_json(y, prefix=''):
    """중첩된 JSON을 비교 가능한 평면(Key-Value Pair) 형태로 변환"""
    out = {}
    if isinstance(y, dict):
        for key, val in y.items():
            new_key = f"{prefix}.{key}" if prefix else key
            out.update(flatten_json(val, new_key))
    elif isinstance(y, list):
        for i, val in enumerate(y):
            new_key = f"{prefix}[{i}]"
            out.update(flatten_json(val, new_key))
    else:
        val_str = str(y).strip().lower()
        if val_str in ['none', 'null']: val_str = "null"
        out[prefix] = val_str
    return out


def calculate_metrics(pred_json, gt_json):
    """Semantic Accuracy & Reliability 계산"""
    pred_flat = flatten_json(pred_json)
    gt_flat = flatten_json(gt_json)

    pred_items = set(pred_flat.items())
    gt_items = set(gt_flat.items())
    pred_keys = set(pred_flat.keys())
    gt_keys = set(gt_flat.keys())

    # Slot F1
    correct_matches = len(pred_items.intersection(gt_items))
    precision = correct_matches / len(pred_items) if len(pred_items) > 0 else 0.0
    recall = correct_matches / len(gt_items) if len(gt_items) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # JGA (Joint Goal Accuracy)
    jga = 1.0 if pred_items == gt_items else 0.0

    # Hallucination Rate (정답에 없는 키 생성 비율)
    hallucinated_keys = len(pred_keys - gt_keys)
    hallucination_rate = hallucinated_keys / len(pred_keys) if len(pred_keys) > 0 else 0.0

    # Hallucination Rate 계산 직전
    # hallucinated_keys_set = pred_keys - gt_keys
    # if len(hallucinated_keys_set) > 0:
    #     print(f"\n[Debug] Hallucinated Keys Sample: {len(pred_keys)}")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "jga": jga, "hallucination_rate": hallucination_rate
    }


def check_schema_compliance(json_data):
    """Syntactic Correctness 체크"""
    try:
        if "general_slots" not in json_data or "step_slots" not in json_data: return 0.0
        if not isinstance(json_data["general_slots"], dict): return 0.0
        if not isinstance(json_data["step_slots"], list): return 0.0
        return 1.0
    except:
        return 0.0


def load_test_dataset(file_path):
    """
    JSONL 파일에서 User Input과 Ground Truth JSON을 추출합니다.
    """
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                entry = json.loads(line)
                messages = entry.get("messages", [])

                user_content = ""
                gt_text = ""

                # 메시지 파싱
                for msg in messages:
                    if msg['role'] == 'user':
                        user_content = msg['content']
                    elif msg['role'] == 'assistant':
                        gt_text = msg['content']

                # Ground Truth 텍스트에서 JSON 부분만 추출
                gt_json = extract_json_from_text(gt_text)

                if user_content and gt_json:
                    dataset.append({
                        "input": user_content,
                        "ground_truth": gt_json
                    })
    except FileNotFoundError:
        print(f"[Error] 파일({file_path})을 찾을 수 없습니다.")
        exit()
    except Exception as e:
        print(f"[Error] 데이터 로드 중 오류: {e}")
        exit()

    return dataset


# --------------------------------------------------------------------------------
# 3. 모델 로드
# --------------------------------------------------------------------------------
print(f"Loading model from {MODEL_ID}...")

# Pre-trained Phi-4
model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.bfloat16,
        device_map = "auto"
)

# Fine-tuned Phi-4
# model = AutoPeftModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     dtype=torch.bfloat16,
#     load_in_4bit=True,
#     device_map="auto"
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.")

# --------------------------------------------------------------------------------
# 4. 데이터 로드 및 평가 루프
# --------------------------------------------------------------------------------
TEST_DATASET = load_test_dataset(TEST_FILE_PATH)
print(f"Loaded {len(TEST_DATASET)} samples from {TEST_FILE_PATH}")

results = {
    "json_validity": [], "schema_compliance": [],
    "precision": [], "recall": [], "f1": [], "jga": [],
    "hallucination_rate": [], "inference_time": []
}

# Training 시 사용했던 시스템 프롬프트 (데이터셋에 있는 것과 동일하게 맞춤)
SYSTEM_PROMPT = "당신은 스마트 팩토리 시뮬레이션을 위한 데이터 추출 도우미입니다. 사용자의 자연어 입력을 분석하여 'general_slots', 'step_slots' (리소스 및 네트워크 포함) 정보를 추출하고, 정해진 JSON 스키마로 변환하세요. 출력 시 먼저 추론 과정(thought)을 서술하고, 그 뒤에 JSON 코드를 작성하세요."

print(f"\n=== Evaluation Start ===")

for idx, data in enumerate(TEST_DATASET):
    prompt = data['input']
    gt_json = data['ground_truth']

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    # 1. Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True,
            do_sample=False,  # 정량 평가를 위해 Greedy Decoding 사용 (일관성)
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    inference_time = time.time() - start_time
    results["inference_time"].append(inference_time)

    # 2. Output Handling
    generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    # 3. JSON Parsing & Metrics
    pred_json = extract_json_from_text(generated_text)

    validity = 0.0
    schema_score = 0.0
    prec = 0.0;
    rec = 0.0;
    f1 = 0.0;
    jga = 0.0;
    hallu = 1.0  # Default values (Failure)

    if pred_json is not None:
        validity = 1.0
        schema_score = check_schema_compliance(pred_json)

        metrics = calculate_metrics(pred_json, gt_json)
        prec = metrics["precision"]
        rec = metrics["recall"]
        f1 = metrics["f1"]
        jga = metrics["jga"]
        hallu = metrics["hallucination_rate"]

    # 결과 저장
    results["json_validity"].append(validity)
    results["schema_compliance"].append(schema_score)
    results["precision"].append(prec)
    results["recall"].append(rec)
    results["f1"].append(f1)
    results["jga"].append(jga)
    results["hallucination_rate"].append(hallu)

    print(
        f"Sample {idx + 1}/{len(TEST_DATASET)} | Time: {inference_time:.2f}s | Valid: {int(validity)} | JGA: {int(jga)} | F1: {f1:.2f}")

# --------------------------------------------------------------------------------
# 5. 최종 리포트 출력
# --------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             FINAL EVALUATION REPORT             ")
print("=" * 50)

avg_validity = np.mean(results["json_validity"]) * 100
avg_schema = np.mean(results["schema_compliance"]) * 100
avg_f1 = np.mean(results["f1"]) * 100
avg_jga = np.mean(results["jga"]) * 100
avg_hallucination = np.mean(results["hallucination_rate"]) * 100
avg_time = np.mean(results["inference_time"])

print(f"[Syntactic Correctness]")
print(f" - JSON Validity Rate    : {avg_validity:.2f}%")
print(f" - Schema Compliance Rate: {avg_schema:.2f}%")
print(f"\n[Semantic Accuracy]")
print(f" - Slot F1 Score         : {avg_f1:.2f}%")
print(f" - Joint Goal Acc (JGA)  : {avg_jga:.2f}%")
print(f"\n[Reliability & Efficiency]")
print(f" - Hallucination Rate    : {avg_hallucination:.2f}%")
print(f" - Avg. Inference Time   : {avg_time:.4f} sec/sample")
print("=" * 50)