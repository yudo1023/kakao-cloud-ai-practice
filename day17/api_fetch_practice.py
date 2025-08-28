# p.316
# 1. 목표 : 5개의 API URL에 GET요청을 보내기
# 2. 요구사항
# 세가지 방식으로 구현하고 성능을 비교
# - 순차 처리
# - ThreadPoolExecutor 사용
# - asyncio와 aiohttp 사용

import time
import requests
import concurrent.futures
import asyncio
import aiohttp

API_URLS = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    "https://jsonplaceholder.typicode.com/posts/4",
    "https://jsonplaceholder.typicode.com/posts/5",
]

# ---- 순차 처리 ---
def sequential_requests(urls, timeout=10):
    print("\n[순차 처리] 시작")
    start_time = time.time()
    per_item = []
    for url in urls:
        start = time.time()
        request = requests.get(url, timeout=timeout)
        request.raise_for_status()
        per_item.append(time.time() - start)
        print(f"성공 : {url}  {len(request.text)}바이트  {per_item[-1]:.3f}초")
    total = time.time() - start_time
    avg = sum(per_item) / len(per_item)
    print(f"[순차 처리] 총 소요: {total:.3f}초, 평균: {avg:.3f}초")
    return {"name": "sequential", "total": total, "avg": avg, "per_item": per_item}

# --- ThreadPoolExecutor ---
def fetch_requests(url, timeout=10):
    start_time = time.time()
    request = requests.get(url, timeout=timeout)
    request.raise_for_status()
    return url, len(request.text), time.time() - start_time

def threaded_requests(urls, max_workers=5, timeout=10):
    print("\n[ThreadPoolExecutor] 시작")
    start_time = time.time()
    per_item = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_url = [ex.submit(fetch_requests, url, timeout) for url in urls]
        for future in concurrent.futures.as_completed(future_to_url):
            url, size, elapsed = future.result()
            per_item.append(elapsed)
            print(f"성공 : {url}  {size}바이트  {elapsed:.3f}초")
    total = time.time() - start_time
    avg = sum(per_item) / len(per_item)
    print(f"[ThreadPoolExecutor] 총 소요: {total:.3f}초, 평균: {avg:.3f}초 (워커={max_workers})")
    return {"name": "threadpool", "total": total, "avg": avg, "per_item": per_item}

# --- asyncio, aiohttp ---
async def afetch(session, url, timeout=10):
    start_time = time.time()
    async with session.get(url, timeout=timeout) as resp:
        text = await resp.text()
    return url, len(text), time.time() - start_time

async def aiohttp_sequential(urls, timeout=10):
    print("\n[asyncio+aiohttp 순차] 시작")
    start_time = time.time()
    per_item = []
    async with aiohttp.ClientSession() as session:
        for url in urls:
            url, size, elapsed = await afetch(session, url, timeout)
            per_item.append(elapsed)
            print(f"성공 : {url}  {size}바이트  {elapsed:.3f}초")
    total = time.time() - start_time
    avg = sum(per_item) / len(per_item)
    print(f"[aiohttp 순차] 총 소요: {total:.3f}초, 평균: {avg:.3f}초")
    return {"name": "aiohttp_sequential", "total": total, "avg": avg, "per_item": per_item}

async def aiohttp_parallel(urls, timeout=10):
    print("\n[asyncio+aiohttp 병렬] 시작")
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [afetch(session, url, timeout) for url in urls]
        results = await asyncio.gather(*tasks)
    per_item = []
    for url, size, elapsed in results:
        per_item.append(elapsed)
        print(f"성공 : {url}  {size}바이트  {elapsed:.3f}초")
    total = time.time() - start_time
    avg = sum(per_item) / len(per_item)
    print(f"[aiohttp 병렬] 총 소요: {total:.3f}초, 평균: {avg:.3f}초")
    return {"name": "aiohttp_parallel", "total": total, "avg": avg, "per_item": per_item}

def main():
    r1 = sequential_requests(API_URLS)
    r2 = threaded_requests(API_URLS, max_workers=5)
    r3 = asyncio.run(aiohttp_sequential(API_URLS))
    r4 = asyncio.run(aiohttp_parallel(API_URLS))

if __name__ == "__main__":
    main()


    
