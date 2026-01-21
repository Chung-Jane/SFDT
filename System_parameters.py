import json
import copy
from KB import SIMULATION_KB


def mapping_system_parameters(abstract_json, kb):
    # 원본 데이터 보존을 위해 깊은 복사
    system_json = copy.deepcopy(abstract_json)

    print("\n[Process] KB Mapping 시작...")

    # Step Slots 순회
    for idx, step in enumerate(system_json.get("step_slots", [])):
        step_name = step.get("step_name", f"{idx + 1}번 공정")

        # -------------------------------------------------
        # 1. Resource Mapping (Worker / Machine)
        # -------------------------------------------------
        r_type = step.get("resource_type")
        r_slots = step.get("resource_slots", {})

        # 1-1. 작업자(Worker)인 경우
        if r_type == "worker" and r_slots.get("worker"):
            worker_info = r_slots["worker"]
            role_name = worker_info.get("role")

            if role_name in kb["worker_profiles"]:
                detailed_params = kb["worker_profiles"][role_name]
            else:
                detailed_params = kb["default_worker"]

            worker_info.update(detailed_params)

            if worker_info.get("worker_count"):
                worker_info["worker_count"] = int(worker_info["worker_count"])

        # 1-2. 기계(Machine)인 경우
        elif r_type == "machine" and r_slots.get("machine"):
            machine_info = r_slots["machine"]
            m_name = machine_info.get("machine_name")

            if m_name in kb["machine_profiles"]:
                detailed_params = kb["machine_profiles"][m_name]
                machine_info.update(detailed_params)

            if machine_info.get("machine_count"):
                machine_info["machine_count"] = int(machine_info["machine_count"])

        # -------------------------------------------------
        # 2. Network Mapping (Physical / App / Transport)
        # -------------------------------------------------
        network_slots = step.get("network_slots", [])

        for n_idx, net in enumerate(network_slots):
            # 2-1. Physical Layer Mapping (connection_type: Wired, Wireless)
            conn_type = net.get("connection_type")
            if conn_type and conn_type in kb["network_profiles"]["physical_layer"]:
                phy_params = kb["network_profiles"]["physical_layer"][conn_type]
                net.update(phy_params)
            else:
                print(f"=== Warning: Connection Type '{conn_type}'이 KB에 없습니다.")

            # 2-2. Application Layer Mapping (content_type: log, video, etc.)
            cont_type = net.get("content_type")
            if cont_type and cont_type in kb["network_profiles"]["app_layer"]:
                app_params = kb["network_profiles"]["app_layer"][cont_type]
                net.update(app_params)

            # 2-3. Transport Layer Mapping (reliability: High, Low)
            rel_type = net.get("reliability")
            if rel_type and rel_type in kb["network_profiles"]["transport_layer"]:
                trans_params = kb["network_profiles"]["transport_layer"][rel_type]
                net.update(trans_params)

    print("[Process] System Parameter 생성 완료.\n")
    return system_json