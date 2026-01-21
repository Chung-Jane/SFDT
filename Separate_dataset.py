#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从一个 JSONL 文件中提取前 N 行，生成新的 JSONL 文件。
用法示例：

    python extract_first_1000.py \
        --input factory_sft.jsonl \
        --output factory_sft_1000.jsonl \
        --num 1000
"""

import argparse


def extract_first_n(input_path: str, output_path: str, n: int):
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if count >= n:
                break
            fout.write(line)
            count += 1

    print(f"提取完成！共提取 {count} 条数据到 '{output_path}'")


def extract_slice(input_path: str, output_path: str, start: int, n: int):
    read_count = 0
    saved_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if read_count < start:
                read_count += 1
                continue
            if n != -1 and saved_count >= n:
                break

            fout.write(line)
            saved_count += 1
            read_count += 1

    print(f"작업 완료! 전체 {read_count}줄 중, 앞의 {start}줄을 건너뛰고 {saved_count}줄을 '{output_path}'에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(description="提取 JSONL 文件的前 N 条记录")
    parser.add_argument("--input", "-i", type=str, required=True, help="原始 JSONL 文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--start", "-s", type=int, default=0, help="시작 위치 (앞에서 몇 개를 건너뛸지)")
    parser.add_argument("--num", "-n", type=int, default=1000, help="提取条数，默认 1000 条")

    args = parser.parse_args()
    # extract_first_n(args.input, args.output, args.num)
    extract_slice(args.input, args.output, args.start, args.num)


if __name__ == "__main__":
    main()
