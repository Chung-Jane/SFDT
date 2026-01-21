import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# print(torch.cuda.is_available())  # 如果返回 True，说明 GPU 可用
# print(torch.cuda.devicepu_count())  # 查看 GPU 数量
# print(torch.cuda.get_device_name(0))  #

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer,AutoModelForCausalLM
load_in_4bit = True

# model = AutoModelForCausalLM.from_pretrained(
#         "/home/smslab/PycharmProjects/smart_factory/Phi-4",
#         torch_dtype = torch.bfloat16,
#         device_map = "auto"
# )

model = AutoPeftModelForCausalLM.from_pretrained(
    "lora_model",  # YOUR MODEL YOU USED FOR TRAINING
    dtype = torch.bfloat16,
    load_in_4bit=load_in_4bit,
)

tokenizer = AutoTokenizer.from_pretrained("lora_model")
messages = [
    {"role": "system", "content": "당신은 스마트 팩토리 시뮬레이션을 위한 데이터 추출 도우미입니다. 사용자의 자연어 입력을 분석하여 'general_slots', 'step_slots' (리소스 및 네트워크 포함) 정보를 추출하고, 정해진 JSON 스키마로 변환하세요. 출력 시 먼저 추론 과정(thought)을 서술하고, 그 뒤에 JSON 코드를 작성하세요."},
    {"role": "user", "content": "우리 이번에 새로 나오는 '프리미엄 수분 크림' 한 2000개 정도 10시간 내에 만들어야 하는데, 기존 라인 좀 손봐야 할 것 같아.\n기존 'Line_Cream' 쓰면 될 것 같고, 크림 한 통에는 본품 용기 1개, 캡 1개, 그리고 물론 크림 원액이 들어가.\n공장 전체 네트워크는 그냥 평범한 와이파이야.\n우선 '원액 혼합' 공정이 있는데, 이건 자동 혼합기 1대가 해. 이건 최신 장비라 상태는 아주 좋아. 혼합 레시피 정보는 혼합기에서 바로 서버로 보내야 하는데, 이게 좀 중요해서 절대 유실되면 안 돼.\n다음은 '용기 충진'인데, 자동 충진기 2대가 크림을 용기에 채워 넣어. 이것도 신형이라 문제없어. 여기서는 충진량 데이터만 장비에 남겨두고 서버로는 안 보내도 돼.\n그 다음 '캡 체결' 공정은 작업자 3명이 캡을 닫고 육안으로 불량 여부를 1차 확인하는 수동 공정이야. 여기서는 제품 불량률 5% 미만으로 유지해야 해.\n마지막으로 '라벨링 및 검수' 공정인데, 자동 라벨링기 1대가 라벨을 붙이고, 비전 검사기 1대가 라벨 부착 위치랑 외관 불량을 꼼꼼히 확인해. 이 라벨링기는 좀 오래돼서 가끔 오류가 나. 비전 검사 결과 데이터는 무조건 서버로 전송돼야 해. 이게 제일 중요해. 라벨 위치 데이터는 절대 틀어지면 안 되거든. 빨리 확인해야 하니까."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

# outputs = model.generate(
#     input_ids = inputs, max_new_tokens = 64, use_cache = True, temperature = 1.5, min_p = 0.1
# )
# tokenizer.batch_decode(outputs)
# print(response)

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(
    input_ids = inputs, streamer = text_streamer, max_new_tokens = 2048,
    use_cache = True, temperature = 1.5, min_p = 0.1
)
