import time
import random
import threading
import time
import concurrent.futures

# 센서 -> 서버로 데이터 전달
def sensor_data_stream():
    while True:
        temperature = 20 + random.uniform(-5, 5)
        yield f"온도 : {temperature:.2f}, 시간 : {time}"
        time.sleep(1)

stream = sensor_data_stream()
for _ in range(5):
    print(next(stream))


# p.267 Thread 클래스 주요 메서드
def background_task():
    while True:
        print("백그라운드 작업 실행 중")
        time.sleep(1)
daemon_thread = threading.Thread(target=background_task, daemon=True)
daemon_thread.start()
print("메인 스레드 시작")
time.sleep(3)
print("메인 쓰레드 종료")

# p.269 스레드 동기화 도구
event = threading.Event()
def waiter():
    print("대기자 : 이벤트를 기다리는 중")
    event.wait()
    print("대기자 : 이벤트 수신 및 작업 진행")

def setter():
    print("설정자 작업 중")
    time.sleep(3)
    print("설정자 설정 완료")
    event.set()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)
t1.start()
t2.start()

# p.271 Condition
data = None
condition = threading.Condition()

def wait_for_data():
    print("대기 : 데이터 준비중")
    with condition:
        condition.wait()
        print(f"대기 : 데이터 '{data}' 수신")

def prepare_data():
    global data
    print("준비 : 데이터 준비중")
    time.sleep(2)
    with condition :
        data = "준비된 데이터"
        print("준비 : 데이터 준비 완료")
        condition.notify()

t1 = threading.Thread(target=wait_for_data)
t2 = threading.Thread(target=prepare_data)

t1.start()
t2.start()

t1.join()
t2.join()

# p.273 스레드 간 데이터 공유와 동기화
counter = 0
counter_lock = threading.Lock()

# 여러 스레드가 동시에 같은 데이터에 접근하면 데이터 동기화 문제 발생
def increment(count):
    global counter
    for _ in range(count):
        current = count
        time.sleep(0.001)
        counter = current + 1

# lock으로 한 번에 하나의 스레드만 공유 데이터 접근하도록 제한(기본)
def increment_with_lock(count):
    global counter
    for _ in range(count):
        counter_lock.acquire()
        try:
            current = counter
            time.sleep(0.001)   
            counter = current + 1
        finally:
            counter_lock.release()

# lock으로 한 번에 하나의 스레드만 공유 데이터 접근하도록 제한(with문)
def increment_with_lock_content(count):
    global counter
    for _ in range(count):
        with counter_lock:
            current = counter
            time.sleep(0.001)
            counter = current + 1

t1 = threading.Thread(target=increment, args=(1000,))
t2 = threading.Thread(target=increment, args=(1000,))

t3 = threading.Thread(target=increment_with_lock, args=(1000,))
t4 = threading.Thread(target=increment_with_lock, args=(1000,))

t5 = threading.Thread(target=increment_with_lock_content, args=(1000,))
t6 = threading.Thread(target=increment_with_lock_content, args=(1000,))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()

# p.281 Queue
import queue

task_queue = queue.Queue()
result_queue = queue.Queue()

def create_tasks():
    print("작업 생성 시작")
    for i in range(10):
        task = f"작업-{i}"
        task_queue.put(task)
        print(f"작업 '{task}' 추가 됨")
        time.sleep(random.uniform(0.1, 0.3))
    for _ in range(3):
        task_queue.put(None)

    print("모든 작업 생성 완료")

def worker(worker_id):
    print(f"워커 : {worker_id} 시작")
    while True:
        task = task_queue.get()
        if task is None:
            print(f"워커 : {worker_id} 작업 종료")
            break
        print(f"워커 {worker_id}가 {task} 처리 중...")
        processing_time = random.uniform(0.5, 1.5)
        result = f"{task} 완료 (소요시간 : {processing_time:.2f})"
        result_queue.put((worker_id, result))
        task_queue.task_done()
        print(f"남은 작업 수 : {task_queue.qsize()}")

def result_collector():
    print("결과 수집기 시작")
    results = []
    for _ in range(10):
        worker_id, result = result_queue.get()
        print(f"결과 수집 워커 {worker_id} -> {result}")
        results.append(result)
        result_queue.task_done()
    print(f"{len(results)} 처리 완료")

creator = threading.Thread(target=create_tasks)
workers = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
collector = threading.Thread(target=result_collector)

creator.start()
for w in workers:
    w.start()
collector.start()

creator.join()
for w in workers:
    w.join()
collector.join()

# p.289 ThreadPoolExecutor
def task(params):
    name, duration = params
    print(f"작업 {name} 시작")
    time.sleep(duration)
    return f"{name} 완료, 소요시간 : {duration}초"

params = [
    ("A", 1),
    ("B", 3),
    ("C", 2),
    ("D", 1.5)
]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(task, params))
    for result in results:
        print(result)



# p.310 여러 웹사이트에서 동시에 정보 수집
import aiohttp
import asyncio

websites = [
    "https://www.google.com",
    "https://www.naver.com",
    "https://www.daum.net",
    "https://www.github.com",
    "https://www.python.org",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.reddit.com"
]

async def fetch(session, url):
    try:
        start_time = time.time()
        async with session.get(url, timeout=10) as response:
            content = await response.text()
            elapsed = time.time() - start_time
            print(f"{url} 응답 완료 : {len(content)}바이트 (소요시간 : {elapsed:.2f}초)")
            return url, len(content), elapsed
    except Exception as e:
        print(f"{url} 오류 발생 : {e}")
        return url, 0, 0
    
async def fetch_all_sequential(urls):
    start_time = time.time()
    results = []
    async with aiohttp.ClientSession() as session:
        for url in urls:
            result = await fetch(session, url)
            results.append(result)
    end_time = time.time()
    print(f"순차 처리 완료 : {end_time - start_time:.2f}초 소요")
    return results

async def fetch_all_parallel(urls):
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in websites]
        results = await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"병렬 처리 완료 : {end_time - start_time:.2f}초 소요")
    return results

async def main():
    sequential_results = await fetch_all_sequential(websites)

    await asyncio.sleep(1)
    parallel_results = await fetch_all_parallel(websites)

if __name__ == "__main__":
    asyncio.run(main())



        

