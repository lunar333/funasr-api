import requests
import time
import multiprocessing as mp
import random
from typing import List

# ===================== 配置项 =====================
API_URL = "http://127.0.0.1:8000/asr/batch"
TXT_PATH = "/home/huihang/pythoncode/asr/test2/1.txt"
# TXT_PATH = "/data/huihang/code/asr/test2/temp/wav.list"
PROCESS_NUM = 8                   # 进程数
BATCH_SIZE_PER_REQUEST = 16     # 每次调用API的音频数
TOTAL_TEST_COUNT = 16       # ✅ 直接指定：总共要测试多少条音频
API_TIMEOUT = 500                # 超时
# ==================================================

def load_audio_list(txt_path: str) -> List[str]:
    """从txt加载音频列表"""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def generate_test_audio_list(base_audios: List[str], total_count: int) -> List[str]:
    """
    生成指定总数的测试音频列表
    - 自动随机采样
    - 不足时循环补齐
    """
    random.seed(42)  # 固定随机种子，可复现
    result = []
    
    # 循环随机抽取，直到凑够 total_count 条
    while len(result) < total_count:
        # 随机打乱
        shuffled = random.sample(base_audios, len(base_audios))
        result.extend(shuffled)
    
    # 截断到指定数量
    result = result[:total_count]
    return result

def worker_process(audio_list: List[str], process_id: int):
    """
    单个进程：
    严格串行执行：
    发送一批 → 等待结果 → 再发送下一批
    """
    total = len(audio_list)
    idx = 0
    success = 0
    fail = 0

    print(f"[进程 {process_id}] 启动，分配音频数：{total}")

    # 核心：必须等上一个请求结束，才发送下一个
    while idx < total:
        # 取当前批次
        batch = audio_list[idx : idx + BATCH_SIZE_PER_REQUEST]
        current_batch_size = len(batch)
        
        try:
            # ===================== 发送请求 =====================
            resp = requests.post(
                API_URL,
                json={"oss_urls": batch},
                timeout=API_TIMEOUT
            )

            # ===================== 等待结果返回 =====================
            if resp.status_code == 200:
                success += 1
                data = resp.json()
                print(f"[进程 {process_id}] ✅ 成功 | 批次:{len(batch)}条 | GPU:{data['gpu_id']}")
            else:
                fail += 1
                print(f"[进程 {process_id}] ❌ 失败 | 批次:{len(batch)}条 | code:{resp.status_code} | {resp.text}")

        except Exception as e:
            fail += 1
            print(f"[进程 {process_id}] ❌ 异常 | 批次:{len(batch)}条 | 错误:{str(e)}")

        # 只有当前批次完全结束，才移动指针
        idx += current_batch_size

    print(f"\n[进程 {process_id}] 🎉 全部完成 | 总成功:{success} | 总失败:{fail}")

def main():
    # 1. 读取原始音频列表
    base_audios = load_audio_list(TXT_PATH)
    print(f"读取到原始音频：{len(base_audios)} 条")

    # 2. 生成指定总数的测试列表（自动随机补齐）
    full_audios = generate_test_audio_list(base_audios, TOTAL_TEST_COUNT)
    print(f"✅ 自动生成测试集 → 总条数：{len(full_audios)} 条")

    # 3. 均分给 N 个进程
    num_per_process = len(full_audios) // PROCESS_NUM
    process_tasks = []
    for i in range(PROCESS_NUM):
        start = i * num_per_process
        end = start + num_per_process if i != PROCESS_NUM-1 else len(full_audios)
        process_tasks.append(full_audios[start:end])

    # 4. 启动多进程
    processes = []
    start_time = time.time()

    print(f"\n🚀 开始启动 {PROCESS_NUM} 个进程压测...\n")
    for i in range(PROCESS_NUM):
        p = mp.Process(target=worker_process, args=(process_tasks[i], i+1))
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 5. 最终统计
    total_cost = round(time.time() - start_time, 2)
    total_batches = len(full_audios) / BATCH_SIZE_PER_REQUEST

    print("\n" + "="*80)
    print(f"🔥 压测完成")
    print(f"📊 总测试音频条数：{TOTAL_TEST_COUNT} 条")
    print(f"📊 总进程数：{PROCESS_NUM}")
    print(f"📊 每批次大小：{BATCH_SIZE_PER_REQUEST} 条")
    print(f"⏱️ 总耗时：{total_cost} s")
    print(f"⚡ 平均每秒处理：{round(TOTAL_TEST_COUNT / total_cost, 2)} 条")
    print("="*80)

if __name__ == "__main__":
    main()