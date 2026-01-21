import json

def find_missing_slot(data):
    missing_slots = []

    # 1. general_slots
    general = data.get("general_slots", {})
    for key, value in general.items():
        if value is None:
            missing_slots.append({
                "path": f"general_slots.{key}",
                "description": f"팩토리의 {key} 정보"
            })

    # 2. step_slots
    steps = data.get("step_slots", [])
    for idx, step in enumerate(steps):
        prefix = f"step_slots[{idx}]"

        # 2-1. step_slots의 basic info
        basic_keys = ["line_name", "step_name", "next_step", "min_quality", "resource_type"]
        for key in basic_keys:
            if step.get(key) is None:
                missing_slots.append({
                    "path": f"{prefix}.{key}",
                    "description": f"{idx+1} 공정의 {key}"
                })

        # 2-2. resource_slots
        r_type = step.get("resource_type")
        r_slots = step.get("resource_slots", {})

        if r_type == "worker":
            worker_data = r_slots.get("worker")
            if worker_data is None:
                missing_slots.append({"path": f"{prefix}.resource_slots.worker", "description": "작업자 세부 정보"})
            else:
                for w_key, w_value in worker_data.items():
                    if w_value is None:
                        missing_slots.append({
                            "path": f"{prefix}.resource_slots.worker.{w_key}",
                            "description": f"{idx+1} 공정 작업자의 {w_key}"
                        })

        elif r_type == "machine":
            machine_data = r_slots.get("machine")
            if machine_data is None:
                missing_slots.append({"path": f"{prefix}.resource_slots.machine", "description": "설비 세부 정보"})
            else:
                for m_key, m_value in machine_data.items():
                    if m_value is None:
                        missing_slots.append({
                            "path": f"{prefix}.resource_slots.worker.{m_key}",
                            "description": f"{idx+1} 공정 기계의 {m_key}"
                        })

        elif r_type is None:
            pass

        # 2-3. network_slots
        networks = step.get("network_slots", [])
        for n_idx, net in enumerate(networks):
            for n_key, n_value in net.items():
                if n_value is None:
                    missing_slots.append({
                        "path": f"{prefix}.network_slots[{n_idx}].{n_key}",
                        "description": f"{idx+1}번 공정 {n_idx+1}번 네트워크의 {n_key}"
                    })

    return missing_slots

# ===== test =====
current_json = {"general_slots":{"target_product":"컵라면 키트","goal_quantity":"600세트","work_time":"8시간","default_network":None,"required_parts":{"용기":1,"면 블록":1,"스프 봉지":1}},"step_slots":[{"line_name":"Line_Noodle","step_name":"자동 계량 투입","next_step":"실링 포장","min_quality":None,"resource_type":"machine","resource_slots":{"worker":None,"machine":{"machine_name":"자동 투입기","machine_condition":"Old","machine_count":"2"}},"network_slots":[]},{"line_name":"Line_Noodle","step_name":"실링 포장","next_step":None,"min_quality":None,"resource_type":"worker","resource_slots":{"worker":{"role":"작업자","worker_count":"4"},"machine":None},"network_slots":[]}]}

missing = find_missing_slot(current_json)
print(json.dumps(missing, indent=2, ensure_ascii=False))