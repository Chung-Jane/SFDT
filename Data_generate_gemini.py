import os
import json
import argparse
import time
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# -----------------------------
# 설정 구역
# -----------------------------

# 사용할 모델 이름
MODEL_NAME = "gemini-2.5-flash"

# Fine-tuning용 system prompt
TARGET_SYSTEM_PROMPT = (
    "당신은 스마트 팩토리 시뮬레이션을 위한 데이터 추출 도우미입니다. "
    "사용자의 자연어 입력을 분석하여 'general_slots', 'step_slots' (리소스 및 네트워크 포함) 정보를 추출하고, "
    "정해진 JSON 스키마로 변환하세요. "
    "출력 시 먼저 추론 과정(thought)을 서술하고, 그 뒤에 JSON 코드를 작성하세요."
)

# 데이터 생성용 meta prompt
DATA_GEN_INSTRUCTION = """
너는 스마트 팩토리 시뮬레이션 모델 학습을 위한 '합성 데이터 생성기(Synthetic Data Generator)'이다.
다양한 제조 현장의 시나리오를 가정하여, 학습 데이터 1세트를 생성하라.

# 수행 작업
1. User Input 창작: 공장 작업자, 엔지니어, 또는 관리자가 시뮬레이션을 요청하는 다양한 상황의 자연어 발화를 창작하라.
   - 비전문가처럼 모호한 표현("적당히", "빨리", "중요해")을 섞거나, 구체적인 수치를 섞어서 다양성을 주어라.
   - 상황: 신규 라인 셋업, 기존 라인 변경, 특정 공정 추가 등.
2. Assistant Output 생성: 위 Input에 대한 이상적인 AI 응답을 생성하라.
   - Thought: 왜 그런 파라미터를 추출했는지 논리적 추론 과정을 먼저 서술하라.
   - JSON: 아래 스키마에 맞는 JSON 데이터를 생성하라. (JSON은 반드시 마크다운 코드 블록으로 감쌀 것)

# Target JSON Schema
{
    "general_slots": {
        "target_product": string or null,
        "goal_quantity": string or null,
        "work_time": string or null,
        "default_network": "Wireless" | "Wired" | null,
        "required_parts": map<string, int> | null,
    },
    "step_slots": [
        {
            "line_name": string or null,
            "step_name": string or null,
            "next_step": string or null,
            "min_quality": string or null,
            "resource_type": "worker" | "machine" | null,
            "resource_slots": {
                "worker": { "role": string or null, "worker_count": string or null } | null,
                "machine": { "machine_name": string or null, "machine_condition": string or null, "machine_count": string or null } | null
            },
            "network_slots": [
                {
                    "target_node": string or null,
                    "connection_type": string or null,
                    "content_type": string or null,
                    "reliability": string or null
                }
            ]
        }
    ]
}

# Few-shot Examples (참고하여 유사한 스타일로 생성할 것)
---
[Example 1]
**User Input:** "이번에는 컵라면 키트 600세트를 포장해야 해. 8시간 동안 근무하고, 네트워크는 안 쓸거야.
한 세트에는 용기 1개, 면 블록 1개, 스프 봉지 1개가 들어가.
면 블록 생산 라인에는 자동 계량 투입 공정이 먼저 있고, 그 다음에 실링 포장 공정이 있어.
자동 계량 투입 공정은 자동 투입기 2대가 면과 스프를 컵에 떨어뜨리는 공정이고, 설비는 꽤 오래돼서 상태가 썩 좋지는 않아. 계량 데이터는 장비에만 남기고 서버로는 안 보내.
실링 포장 공정은 작업자 4명이 실링기를 돌리면서 불량을 눈으로 체크하는 사람 중심 공정이고, 네트워크는 사용하지 않아."

**Assistant Output:**
thought: 컵라면 키트 600세트, 8시간, 네트워크 미사용이므로 default_network=null로 설정한다. 한 세트 구성(용기 1, 면 1, 스프 1)을 required_parts에 매핑한다. 면 생산 라인이므로 line_name을 "Line_noodle"로 설정한다. 자동 계량 투입은 자동 투입기 2대가 수행하는 머신 공정(resource_type="machine")이고, 설비가 오래됐다고 했으므로 machine_condition="Old"로 둔다. 네트워크 사용이 없으므로 network_slots는 빈 배열이다. 실링 포장은 작업자 4명의 사람 공정(resource_type="worker")이므로 worker에 role과 worker_count를 넣고 machine=null로 둔다. 역시 네트워크는 사용하지 않는다. control_condition 언급이 없으므로 null 그대로 유지한다.

```json
{"general_slots":{"target_product":"컵라면 키트","goal_quantity":"600세트","work_time":"8시간","default_network":null,"required_parts":{"용기":1,"면 블록":1,"스프 봉지":1}},"step_slots":[{"line_name":"Line_Noodle","step_name":"자동 계량 투입","next_step":"실링 포장","min_quality":null,"resource_type":"machine","resource_slots":{"worker":null,"machine":{"machine_name":"자동 투입기","machine_condition":"Old","machine_count":"2"}},"network_slots":[]},{"line_name":"Line_Noodle","step_name":"실링 포장","next_step":null,"min_quality":null,"resource_type":"worker","resource_slots":{"worker":{"role":"작업자","worker_count":"4"},"machine":null},"network_slots":[]}]}
```

[Example 2]
**User Input:** "소형 드론 완제품 50대를 만들어야 해. 9시간 동안 돌릴 거고, 공장 전체 네트워크는 유선 이더넷이지만, 최종 테스트 구역은 와이파이로만 연결돼 있어. 
드론 한 대에는 프레임 1개, 모터 4개, 프로펠러 4개, 배터리 1개, 메인보드 1개가 필요해. 
'Line_Drone'에는 프레임 조립 → 전자부품 장착 → 펌웨어 다운로드 → 비행 테스트 공정이 있어. 프레임 조립은 작업자 2명이 프레임과 모터, 프로펠러를 조립하는 공정이야. 여기서는 조립 품질 목표를 92점 이상으로 잡고, 네트워크는 사용하지 않아. 
전자부품 장착 공정은 장착 로봇 2대가 보드와 커넥터를 자동으로 끼우는 공정이고, 설비는 신형이라 거의 새거로 보면 돼. 
이 공정에서는 장착 로그를 장비 내부에만 저장하고 서버로는 안 보내. 펌웨어 다운로드 공정은 작업자 1명이 PC에서 펌웨어를 올리는 공정인데, 드론 각각의 시리얼과 버전 정보는 유선으로 생산 이력 서버에 저장해. 이건 추적용이라 데이터가 절대 틀어지면 안돼. 
마지막 비행 테스트는 시험 장비 1대가 자동으로 호버링 테스트를 돌릴 거고, 테스트 결과와 로그는 와이파이로 드론 테스트 서버에 전송돼. 
이 결과가 깨지면 불량 판정이 엉킬 수 있어서 절대 유실되면 안돼."

**Assistant Output:**
thought: 소형 드론 완제품 50대, 9시간, 기본 네트워크는 유선(Wired)로 설정하고, 부품 구성을 general_slots에 넣은 뒤, 프레임 조립(작업자), 전자부품 장착(신형 장착 로봇), 펌웨어 다운로드(작업자+유선 이력 서버), 비행 테스트(시험 장비+와이파이 테스트 서버) 네 공정을 순서대로 step_slots에 worker/machine 및 네트워크 조건과 함께 매핑한다. 네트워크로 전송되는 정보는 추적용 시리얼·버전 정보와 테스트 결과 로그이므로, 둘 다 content_type을 log로 통일한다.

```json
{"general_slots":{"target_product":"소형 드론 완제품","goal_quantity":"50대","work_time":"9시간","default_network":"Wired","required_parts":{"프레임":1,"모터":4,"프로펠러":4,"배터리":1,"메인보드":1}},"step_slots":[{"line_name":"Line_Drone","step_name":"프레임 조립","next_step":"전자부품 장착","min_quality":"92","resource_type":"worker","resource_slots":{"worker":{"role":"작업자","worker_count":"2"},"machine":null},"network_slots":[]},{"line_name":"Line_Drone","step_name":"전자부품 장착","next_step":"펌웨어 다운로드","min_quality":null,"resource_type":"machine","resource_slots":{"worker":null,"machine":{"machine_name":"장착 로봇","machine_condition":"New","machine_count":"2"}},"network_slots":[]},{"line_name":"Line_Drone","step_name":"펌웨어 다운로드","next_step":"비행 테스트","min_quality":null,"resource_type":"worker","resource_slots":{"worker":{"role":"작업자","worker_count":"1"},"machine":null},"network_slots":[{"target_node":"생산 이력 서버","connection_type":"Wired","content_type":"log","reliability":"High"}]},{"line_name":"Line_Drone","step_name":"비행 테스트","next_step":null,"min_quality":null,"resource_type":"machine","resource_slots":{"worker":null,"machine":{"machine_name":"시험 장비","machine_condition":null,"machine_count":"1"}},"network_slots":[{"target_node":"드론 테스트 서버","connection_type":"Wireless","content_type":"log","reliability":"High"}]}]}
```
---

# 출력 형식 (JSON 포맷 준수)
{
    "user_input": "창작한 발화 내용...",
    "assistant_output": "thought: ... \\n\\n```json\\n{ ... }\\n```"
}
"""


def clean_and_parse_json(text):
    """LLM 응답에서 JSON 객체만 추출"""
    try:
        # 1. ```json ... ``` 패턴 추출
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # 2. { ... } 패턴 추출
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                text = text[start:end]
        return json.loads(text, strict=False)
    except Exception:
        return None


def generate_one_pair(model) -> dict:
    """
    LLM을 호출하여 (User Input, Assistant Output) 쌍을 생성합니다.
    """
    try:
        response = model.generate_content(
            f"{DATA_GEN_INSTRUCTION}\n\n새로운 공장 시뮬레이션 시나리오 데이터 1개를 생성해줘. JSON 형식으로만 답해줘.",
            generation_config={"response_mime_type": "application/json"}
        )

        return clean_and_parse_json(response.text)

    except Exception as e:
        print(f"Error generating data: {e}")
        return None


def save_to_jsonl(data_pair, file_handle):
    """
    파일 저장
    """
    if not data_pair:
        return

    jsonl_entry = {
        "messages": [
            {"role": "system", "content": TARGET_SYSTEM_PROMPT},
            {"role": "user", "content": data_pair.get("user_input", "")},
            {"role": "assistant", "content": data_pair.get("assistant_output", "")}
        ]
    }

    file_handle.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")


# -----------------------------
# Main 실행
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM 기반 스마트 팩토리 데이터셋 생성기")
    parser.add_argument("-n", "--num", type=int, default=10, help="생성할 데이터 개수")
    parser.add_argument("-o", "--out", type=str, default="factory_dataset_v2.jsonl", help="저장할 파일명")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("환경 변수 API_KEY를 먼저 설정해주세요.")
        return

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        MODEL_NAME,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    print(f"데이터 {args.num}개 생성 시작...", end="\r")

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        while count < args.num:
            print(f"  - Generating sample {count + 1}/{args.num}...", end="\r")

            # 1. 데이터 생성 (Input + Output 창작)
            data_pair = generate_one_pair(model)

            if data_pair and "user_input" in data_pair and "assistant_output" in data_pair:
                # 2. 파일 저장
                save_to_jsonl(data_pair, f)
                count += 1
                time.sleep(1)  # 무료 티어 속도 제한 방지 (1초 대기)
                if count % 10 == 0:
                    print(f"\n{count}개 생성됨...")
            else:
                print("생성 실패 또는 포맷 불일치. 재시도합니다.")

    print(f"완료! 총 {count}개의 SFT 샘플이 {args.out} 파일에 저장되었습니다.")


if __name__ == "__main__":
    main()
