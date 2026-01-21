SIMULATION_KB = {
    "worker_profiles": {
        "작업자": {
            "processing_time_mean": 10.0,  # 분 단위
            "processing_time_std": 1.5,
            "working_time": 50,
            "break_time": 10,              # 분
            "defect_probability": 0.05,    # 5%
            "defect_mean": 0.01,
            "defect_std": 0.002
        },
        "숙련 작업자": {
            "processing_time_mean": 8.0,   # 더 빠름
            "processing_time_std": 0.5,    # 편차 적음
            "working_time": 70,
            "break_time": 10,
            "defect_probability": 0.01,    # 불량률 낮음
            "defect_mean": 0.005,
            "defect_std": 0.001
        }
    },
    "machine_profiles": {
        "PC": {
            "modelica_class": "null",
            "cycle_time": 300,
            "mtbf": 500,                    # 평균 고장 간격
            "mttr": 30,                     # 평균 수리 시간
            "efficiency": 0.8
        },
        "Robot_Arm": {
            "modelica_class": "null",
            "cycle_time": 400,
            "mtbf": 1000,
            "mttr": 60,
            "efficiency": 0.9
        }
    },
    # 만약 KB에 없는 역할이 들어올 경우를 대비한 기본값
    "default_worker": {
        "processing_time_mean": 12.0,
        "processing_time_std": 2.0,
        "working_time": 50,
        "break_time": 10,
        "defect_probability": 0.1,
        "defect_mean": 0.02,
        "defect_std": 0.005,
    },
    "network_profiles": {
        "physical_layer": {
            "Wired": {
                "net_device": "Csma",
                "link_capacity": "1G",       # bps
                "delay": 2,              # ms
                "error_rate": 0.00001,
            },
            "Wireless": {
                "net_device": "Wifi",
                "link_capacity": "100M",     # bps
                "delay": 0,              # ms
                "error_rate": 0.01,
                "wifi_standard": "Wifi 6"
            }
        },
        "app_layer": {
            "log": {
                "packet_size": 64,
                "data_rate": "1K",       # bps
                "application_type": "UdpClient",
                "interval": 1,           # s
            },
            "monitoring": {
                "packet_size": 64,
                "data_rate": "1K",       # bps
                "application_type": "UdpClient",
                "interval": 1,           # s
            },
            "video": {
                "packet_size": 1460,
                "data_rate": "50M",     # bps
                "application_type": "OnOffApp",
                "on_time": 100,
                "off_time": 0
            }
        },
        "transport_layer": {
            "High": {
                "protocol": "TCP"
            },
            "Low": {
                "protocol": "UDP"
            }
        }
    }
}