import os
import torch
import json
import re
import sys
import io
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from KB import SIMULATION_KB
from system_parameters import mapping_system_parameters

# 터미널 인코딩 설정
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ---------------------------------------------------------
# 1. 모델 및 토크나이저 로드
# ---------------------------------------------------------
print("Loading model...")
load_in_4bit = True
model_id = "lora_model"  # 학습된 모델 경로

model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
print("Model loaded successfully.")


# ---------------------------------------------------------
# 2. 유틸리티 함수
# ---------------------------------------------------------

def extract_json_from_output(text):
    try:
        match = re.search(r'```json(.*?)```', text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)

        match = re.search(r'```(.*?)```', text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                return json.loads(json_str[start_idx:end_idx])

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return None
    except Exception as e:
        print(f"JSON Decode Error Detail: {e}")
        return None


def find_missing_slot(json_data):
    # 1. General Slots 검사
    general = json_data.get("general_slots", {})
    if not general.get("target_product"): return "어떤 제품을 생산하시나요?"
    if not general.get("goal_quantity"): return "목표 생산량은 몇 개 인가요?"
    if not general.get("work_time"): return "예상되는 총 작업 시간은 얼마인가요?"
    if not general.get("default_network"): return "공장 내에서 기본적으로 사용하는 네트워크 연결 방식은 무엇인가요? (예: 유선, 무선)"
    if not general.get("required_parts"): return "생산을 위해 필요한 핵심 부품의 종류와 수량은 어떻게 되나요?"

    # 2. Step Slots 검사
    steps = json_data.get("step_slots", [])
    if not steps:
        return "생산 공정 단계가 비어있습니다. 어떤 공정이 있나요?"

    for idx, step in enumerate(steps):
        s_name = step.get("step_name")
        step_name = f"'{s_name}'" if s_name else f"{idx + 1}번째 공정"

        # [Basic Info] 라인 이름 및 품질 기준
        if not step.get("line_name"): return f"{step_name}은(는) 어떤 생산 라인에서 진행되나요?"
        if not step.get("min_quality"): return f"{step_name}의 최소 품질 기준(%)은 얼마인가요?"

        # [Resource Info] 리소스 타입에 따른 조건부 검사
        r_type = step.get("resource_type")
        if not r_type:
            return f"{step_name}에 투입될 리소스 타입(작업자/기계)을 알 수 없습니다."

        r_slots = step.get("resource_slots", {})

        # Case A: 작업자(Worker)인 경우
        if r_type == "worker":
            worker_data = r_slots.get("worker")
            if not worker_data:
                return f"{step_name}의 작업자 상세 정보가 비어있습니다."

            if not worker_data.get("role"):
                return f"{step_name}에 투입되는 작업자의 역할(Role)은 무엇인가요?"
            if not worker_data.get("worker_count"):
                return f"{step_name}에는 몇 명의 작업자가 투입되나요?"

        # Case B: 기계(Machine)인 경우
        elif r_type == "machine":
            machine_data = r_slots.get("machine")
            if not machine_data:
                return f"{step_name}의 기계 상세 정보가 비어있습니다."

            if not machine_data.get("machine_name"):
                return f"{step_name}에 사용되는 기계의 이름(종류)은 무엇인가요?"
            if not machine_data.get("machine_count"):
                return f"{step_name}에는 기계가 몇 대 사용되나요?"

        # [Network Info] 네트워크 리스트 및 필수 필드 검사
        networks = step.get("network_slots", [])
        if not networks: return f"{step_name} 단계의 네트워크 정보가 비어있습니다."

        for n_idx, net in enumerate(networks):
            net_prefix = f"{step_name}의 네트워크({n_idx + 1}번)"

            if not net.get("target_node"): return f"{net_prefix}가 연결될 대상 노드(서버 등)는 어디인가요?"
            if not net.get("connection_type"): return f"{net_prefix}의 연결 방식(Wireless/Wired)은 무엇인가요?"
            if not net.get("content_type"): return f"{net_prefix}를 통해 전송되는 데이터의 종류(log, video 등)는 무엇인가요?"
            if not net.get("reliability"): return f"{net_prefix}의 통신 신뢰성 등급(High/Low)은 어떻게 되나요?"

    return None


# ---------------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------------

messages = [
    {
        "role": "system",
        "content": "당신은 스마트 팩토리 시뮬레이션을 위한 데이터 추출 도우미입니다. 사용자의 자연어 입력을 분석하여 'general_slots', 'step_slots' (리소스 및 네트워크 포함) 정보를 추출하고, 정해진 JSON 스키마로 변환하세요. 출력 시 먼저 추론 과정(thought)을 서술하고, 그 뒤에 JSON 코드를 작성하세요."
    }
]

print("\n=== 스마트 팩토리 설정 도우미 (종료: q) ===")

last_bot_question = None
current_json = {}

while True:
    user_input = input("\nUser: ")
    if user_input.lower() == 'q':
        break

    if last_bot_question:
        combined_content = f"이전 질문: \"{last_bot_question}\"\n사용자 답변: \"{user_input}\""
        messages.append({"role": "user", "content": combined_content})
    else:
        messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    # JSON 추출 시도
    extracted_data = extract_json_from_output(generated_text)

    if extracted_data:
        current_json = extracted_data  # 상태 업데이트
        print(f"\nChatBot: 데이터 업데이트 성공")

        messages.append({"role": "assistant", "content": generated_text})
        next_question = find_missing_slot(current_json)

        if next_question:
            print(f"ChatBot: {next_question}")
            last_bot_question = next_question
        else:
            print("ChatBot: 모든 설정 완료. 시뮬레이션 준비.")
            # 최종 결과 미리보기
            print(json.dumps(current_json, indent=2, ensure_ascii=False))
            break

    else:
        print("ChatBot: 이해하지 못했습니다. 다시 말씀해 주세요.")

# ---------------------------------------------------------
# 4. KB 매핑 및 결과 출력
# ---------------------------------------------------------
if current_json:
    final_abstract_json = current_json

    print("\n=== Parameter mapping... ===")
    try:
        final_system_json = mapping_system_parameters(final_abstract_json, SIMULATION_KB)

        print("=== Final System Parameters ===")
        print(json.dumps(final_system_json, indent=2, ensure_ascii=False))

        with open('system_parameters.json', 'w', encoding='utf-8') as f:
            json.dump(final_system_json, f, indent=2, ensure_ascii=False)

        print("\n[Done] 'system_parameters.json' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"\n[Error] KB 매핑 중 오류 발생: {e}")
        print("Abstract Parameter 상태를 유지합니다.")
        print(json.dumps(final_abstract_json, indent=2, ensure_ascii=False))
else:
    print("\n[System] 수집된 데이터가 없어 종료합니다.")

