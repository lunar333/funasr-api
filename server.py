import time
import uuid
import asyncio
import multiprocessing as mp
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from funasr import AutoModel
import torch

# ===================== 全局配置 =====================
GPU_NUM = 8
MAX_QUEUE_SIZE = 1000
BATCH_SIZE = 64
PUNC_MODEL_REVISION = "v2.0.4"
ROTATE_SECONDS = 86400

TASK_TIMEOUT = 600
BATCH_WAIT_TIMEOUT = 0.01  # 动态batch等待
DOWNLOAD_WORKERS_PER_PROCESS = 16
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_ROOT_DIR = os.path.join(tempfile.gettempdir(), "funasr_batch_api")
DOWNLOAD_RETRY_TIMES = 3
DOWNLOAD_RETRY_BACKOFF_SECONDS = 0.5
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_POOL_CONNECTIONS = 500
DOWNLOAD_POOL_MAXSIZE = 500
PUNC_BATCH_SIZE = 16

# ===================== OSS加速配置 =====================
OSS_DOMAIN_RULES = [
    {
        "old_domain": "ai-data-solutions.oss-cn-beijing.aliyuncs.com",
        "new_domain": "ai-data-solutions.oss-accelerate.aliyuncs.com"
    }
]

# ===================== 日志 =====================
class GPULogger:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.base_log_dir = "logs"
        self.base_error_dir = "error"
        self.current_log_dir = None
        self.logger = None
        self.error_logger = None
        self.last_rotate_time = time.time()
        self._make_dirs()
        self._init_logger()

    def _make_dirs(self):
        now = datetime.now().strftime("%Y%m%d_%H%M")
        self.current_log_dir = os.path.join(self.base_log_dir, now)
        os.makedirs(self.current_log_dir, exist_ok=True)
        os.makedirs(self.base_error_dir, exist_ok=True)

    def _init_logger(self):
        self.logger = self._get_logger(
            os.path.join(self.current_log_dir, f"gpu_{self.gpu_id}.log"),
            f"gpu_log_{self.gpu_id}_{uuid.uuid4()}"
        )
        self.error_logger = self._get_logger(
            os.path.join(self.base_error_dir, f"gpu_{self.gpu_id}_error.log"),
            f"gpu_error_{self.gpu_id}_{uuid.uuid4()}"
        )

    def _get_logger(self, path: str, name: str):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def check_rotate(self):
        if time.time() - self.last_rotate_time >= ROTATE_SECONDS:
            self._make_dirs()
            self._init_logger()
            self.last_rotate_time = time.time()

    def info(self, msg: str):
        self.check_rotate()
        self.logger.info(msg)

    def error(self, msg: str):
        self.check_rotate()
        self.error_logger.error(msg)

# ===================== 请求模型 =====================
class ASRRequest(BaseModel):
    oss_urls: List[str]

app = FastAPI(title="Multi-GPU FunASR Service")

# 🚀 全局队列
global_task_queue = mp.Queue(MAX_QUEUE_SIZE)
gpu_processes = []

# ===================== URL处理 =====================
def preprocess_oss_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    for rule in OSS_DOMAIN_RULES:
        if rule["old_domain"] in url:
            url = url.replace(rule["old_domain"], rule["new_domain"])
    return url


def _format_exception(e: Exception) -> str:
    msg = str(e).strip()
    return msg if msg else repr(e)


def _safe_strip_text(v) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    return v.strip()


def _create_download_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=DOWNLOAD_POOL_CONNECTIONS,
        pool_maxsize=DOWNLOAD_POOL_MAXSIZE,
        max_retries=0
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _download_one_audio(url: str, dst_dir: str, idx: int, session: requests.Session) -> str:
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1] or ".wav"
    last_err = None

    for attempt in range(1, DOWNLOAD_RETRY_TIMES + 2):
        local_name = f"{int(time.time() * 1000)}_{idx}_{uuid.uuid4().hex}{suffix}"
        local_path = os.path.join(dst_dir, local_name)

        try:
            with session.get(
                url,
                headers={"User-Agent": "funasr-batch-api/1.0"},
                timeout=(5, DOWNLOAD_TIMEOUT),
                stream=True
            ) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as out_f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            out_f.write(chunk)

            if os.path.getsize(local_path) <= 0:
                raise RuntimeError("下载后的音频文件为空")
            return local_path
        except Exception as e:
            last_err = e
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass

            if attempt <= DOWNLOAD_RETRY_TIMES:
                time.sleep(DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt)
                continue
            break

    raise RuntimeError(f"下载失败，已重试{DOWNLOAD_RETRY_TIMES}次: {str(last_err)}")


def download_audios_concurrently(
    urls: List[str], dst_dir: str, logger: GPULogger, session: requests.Session
):
    local_paths = [None] * len(urls)
    errors = [""] * len(urls)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS_PER_PROCESS) as executor:
        futures = {
            executor.submit(_download_one_audio, url, dst_dir, idx, session): idx
            for idx, url in enumerate(urls)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                local_paths[idx] = future.result()
            except Exception as e:
                errors[idx] = str(e)
                logger.error(f"下载失败 idx={idx} url={urls[idx]} err={errors[idx]}")

    return local_paths, errors

# ===================== Worker =====================
def asr_worker(gpu_id: int, task_queue: mp.Queue):
    mp.current_process().name = f"GPU-{gpu_id}"
    logger = GPULogger(gpu_id)
    device = f"cuda:{gpu_id}"
    worker_download_dir = os.path.join(DOWNLOAD_ROOT_DIR, f"gpu_{gpu_id}")
    os.makedirs(worker_download_dir, exist_ok=True)
    download_session = _create_download_session()

    try:
        asr_model = AutoModel(model="paraformer-zh", disable_update=True, device=device)
        punc_model = AutoModel(model="ct-punc", model_revision=PUNC_MODEL_REVISION, disable_update=True, device=device)
        logger.info(f"[GPU{gpu_id}] 模型加载完成，进程已启动")
    except Exception as e:
        logger.error(f"[GPU{gpu_id}] 模型加载失败: {str(e)}")
        return

    logger.info(f"[GPU{gpu_id}] 准备就绪，开始接收任务")

    while True:
        batch_tasks = []

        # ===== 动态batch =====
        while len(batch_tasks) < BATCH_SIZE:
            try:
                task = task_queue.get(timeout=BATCH_WAIT_TIMEOUT)
                if task is None:
                    logger.info(f"[GPU{gpu_id}] 进程退出")
                    return
                batch_tasks.append(task)
            except:
                break

        if not batch_tasks:
            continue

        all_urls = []
        meta = []

        # ===== 收集任务 =====
        for task in batch_tasks:
            task_id = task["task_id"]
            oss_urls = task["oss_urls"]
            conn = task["conn"]

            try:
                processed_urls = [preprocess_oss_url(u) for u in oss_urls]
            except:
                processed_urls = oss_urls

            logger.info(f"[GPU{gpu_id}] TASK={task_id} | 入参: {oss_urls}")

            all_urls.extend(processed_urls)
            meta.append((task_id, conn, len(oss_urls), oss_urls))

        # ===== 下载 + 推理 =====
        start_time = time.time()
        download_cost_time = 0.0
        infer_cost_time = 0.0
        punc_cost_time = 0.0
        local_paths = []

        try:
            download_start = time.time()
            local_paths, download_errors = download_audios_concurrently(
                all_urls, worker_download_dir, logger, download_session
            )
            download_cost_time = round(time.time() - download_start, 2)

            infer_inputs = []
            infer_input_indexes = []
            for idx, path in enumerate(local_paths):
                if path:
                    infer_inputs.append(path)
                    infer_input_indexes.append(idx)

            asr_res = [{"text": "", "timestamp": [], "error": ""} for _ in range(len(all_urls))]
            for idx, err in enumerate(download_errors):
                if err:
                    asr_res[idx]["error"] = f"下载失败: {err}"

            if infer_inputs:
                infer_start = time.time()
                with torch.no_grad():
                    infer_res = asr_model.generate(input=infer_inputs, batch_size=BATCH_SIZE)
                infer_cost_time = round(time.time() - infer_start, 2)

                for j, idx in enumerate(infer_input_indexes):
                    if j < len(infer_res):
                        asr_res[idx] = infer_res[j]
                    else:
                        asr_res[idx] = {"text": "", "timestamp": [], "error": "推理结果数量不匹配"}

        except Exception as e:
            err = str(e)
            logger.error(f"[GPU{gpu_id}] batch下载/推理失败: {err}")
            asr_res = [{"text": "", "error": err}] * len(all_urls)
        finally:
            for p in local_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.error(f"[GPU{gpu_id}] 删除临时音频失败: {p} err={str(e)}")

        # ===== 标点批量推理 =====
        punc_errors = {}
        punc_texts = []
        punc_indexes = []
        for i, ar in enumerate(asr_res):
            if ar.get("error", ""):
                continue
            text = _safe_strip_text(ar.get("text", ""))
            if text:
                punc_texts.append(text)
                punc_indexes.append(i)

        if punc_texts:
            punc_start = time.time()
            try:
                for start in range(0, len(punc_texts), PUNC_BATCH_SIZE):
                    chunk_texts = punc_texts[start: start + PUNC_BATCH_SIZE]
                    chunk_indexes = punc_indexes[start: start + PUNC_BATCH_SIZE]

                    # 二次保护：空文本不进标点
                    non_empty_pairs = [
                        (t, idx) for t, idx in zip(chunk_texts, chunk_indexes) if _safe_strip_text(t)
                    ]
                    if not non_empty_pairs:
                        continue

                    chunk_texts = [x[0] for x in non_empty_pairs]
                    chunk_indexes = [x[1] for x in non_empty_pairs]

                    with torch.no_grad():
                        punc_res = punc_model.generate(
                            input=chunk_texts
                        )

                    if len(punc_res) != len(chunk_indexes):
                        raise RuntimeError(
                            f"标点结果数量不匹配: input={len(chunk_indexes)} output={len(punc_res)}"
                        )

                    for j, idx in enumerate(chunk_indexes):
                        punc_text = _safe_strip_text(punc_res[j].get("text", ""))
                        if punc_text:
                            asr_res[idx]["text"] = punc_text
            except Exception as e:
                err = _format_exception(e)
                logger.error(f"[GPU{gpu_id}] 标点批量推理失败 err={err}")
                for idx in punc_indexes:
                    punc_errors[idx] = f"标点批量推理失败: {err}"

            punc_cost_time = round(time.time() - punc_start, 2)

        # ===== 拆分回各task =====
        idx = 0

        for task_id, conn, count, oss_urls in meta:
            task_start_idx = idx
            sub_res = asr_res[idx: idx + count]
            idx += count

            final_result = []

            for i, ar in enumerate(sub_res):
                item = {
                    "key": oss_urls[i],
                    "text": "",
                    "error": "",
                    "start": 0.0,
                    "end": 0.0
                }

                try:
                    global_idx = task_start_idx + i
                    item["error"] = ar.get("error", "")
                    if global_idx in punc_errors:
                        if item["error"]:
                            item["error"] = f"{item['error']} | {punc_errors[global_idx]}"
                        else:
                            item["error"] = punc_errors[global_idx]

                    text = _safe_strip_text(ar.get("text", ""))
                    if text:
                        item["text"] = text

                    # ✅ timestamp恢复
                    timestamp = ar.get("timestamp", [])
                    if isinstance(timestamp, list) and len(timestamp) > 0:
                        item["start"] = round(timestamp[0][0] / 1000.0, 3)
                        item["end"] = round(timestamp[-1][1] / 1000.0, 3)

                except Exception as e:
                    item["error"] = str(e)

                final_result.append(item)

            cost_time = round(time.time() - start_time, 2)

            # ===== 保持你原始日志格式 =====
            logger.info(
                f"[GPU{gpu_id}] TASK={task_id} | BATCH_SIZE={count} | "
                f"下载耗时={download_cost_time}s | 推理耗时={infer_cost_time}s | 标点耗时={punc_cost_time}s | 总耗时={cost_time}s"
            )
            logger.info(f"[GPU{gpu_id}] TASK={task_id} | 出参: {final_result}")

            # ⚠️ 每个task只发送一次
            try:
                conn.send({
                    "success": True,
                    "task_id": task_id,
                    "result": final_result,
                    "gpu_id": gpu_id
                })
            except Exception as e:
                logger.error(f"[GPU{gpu_id}] TASK={task_id} | conn发送失败: {str(e)}")
# ===================== 启动 =====================
def init_workers():
    global gpu_processes

    for p in gpu_processes:
        if p.is_alive():
            p.terminate()
            p.join()

    gpu_processes.clear()

    for gpu_id in range(GPU_NUM):
        p = mp.Process(target=asr_worker, args=(gpu_id, global_task_queue), daemon=True)
        p.start()
        gpu_processes.append(p)
        print(f"✅ GPU {gpu_id} 进程启动成功 PID={p.pid}")

@app.on_event("startup")
async def startup():
    init_workers()

@app.on_event("shutdown")
async def shutdown():
    for _ in range(GPU_NUM):
        try:
            global_task_queue.put_nowait(None)
        except:
            pass

    await asyncio.sleep(1)

    for p in gpu_processes:
        if p.is_alive():
            p.terminate()

# ===================== API =====================
@app.post("/asr/batch")
async def batch_asr(request: ASRRequest):
    oss_urls = request.oss_urls

    if not oss_urls:
        raise HTTPException(400, "至少传入1个音频")
    if len(oss_urls) > 64:
        raise HTTPException(400, "最多支持64个音频")

    task_id = str(uuid.uuid4())
    parent_conn, child_conn = mp.Pipe()

    try:
        global_task_queue.put_nowait({
            "task_id": task_id,
            "oss_urls": oss_urls,
            "conn": child_conn
        })
    except:
        raise HTTPException(503, "队列已满")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(parent_conn.recv),
            timeout=TASK_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, f"任务超时 {TASK_TIMEOUT}s")

    if not result["success"]:
        raise HTTPException(500, result["error"])

    return {
        "task_id": task_id,
        "gpu_id": result["gpu_id"],
        "num_inputs": len(oss_urls),
        "results": result["result"]
    }

# ===================== main =====================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
