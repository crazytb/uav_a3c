"""
간단한 테스트: 공유 클라우드 자원이 제대로 작동하는지 확인
"""
import torch.multiprocessing as mp
from drl_framework.custom_env import make_env
from drl_framework.network_state import NetworkState
from drl_framework import params

def test_worker(worker_id, network_state, results_queue):
    """각 워커가 OFFLOAD를 시도하고 결과 기록"""
    ENV_PARAMS = params.ENV_PARAMS.copy()
    REWARD_PARAMS = params.REWARD_PARAMS

    env_fn = make_env(**ENV_PARAMS, reward_params=REWARD_PARAMS,
                     network_state=network_state, worker_id=worker_id)
    env = env_fn()

    obs, _ = env.reset()

    success_count = 0
    failure_count = 0

    # 50번 OFFLOAD 시도
    for step in range(50):
        # 항상 OFFLOAD (action=1) 시도
        obs, reward, done, truncated, info = env.step(1)  # OFFLOAD

        if env.offload_success == 1:
            success_count += 1
        else:
            failure_count += 1

        if done:
            obs, _ = env.reset()

    results_queue.put({
        'worker_id': worker_id,
        'success': success_count,
        'failure': failure_count
    })

    env.close()

def main():
    mp.set_start_method('spawn', force=True)

    n_workers = 5
    max_cloud_capacity = 1000  # 작은 값으로 설정하여 경쟁 유도

    # NetworkState 생성
    network_state = NetworkState(n_workers, max_cloud_capacity=max_cloud_capacity)

    print(f"\n{'='*60}")
    print(f"공유 클라우드 자원 테스트")
    print(f"{'='*60}")
    print(f"워커 수: {n_workers}")
    print(f"초기 클라우드 자원: {max_cloud_capacity}")
    print(f"각 워커가 50번 OFFLOAD 시도")
    print(f"{'='*60}\n")

    # 결과 수집용 큐
    results_queue = mp.Queue()

    # 워커 프로세스 시작
    processes = []
    for worker_id in range(n_workers):
        p = mp.Process(target=test_worker, args=(worker_id, network_state, results_queue))
        p.start()
        processes.append(p)

    # 모든 워커 완료 대기
    for p in processes:
        p.join()

    # 결과 수집
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    results.sort(key=lambda x: x['worker_id'])

    # 결과 출력
    print("\n결과:")
    print(f"{'Worker':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<15}")
    print("-" * 50)

    total_success = 0
    total_failure = 0

    for r in results:
        total = r['success'] + r['failure']
        rate = (r['success'] / total * 100) if total > 0 else 0
        print(f"{r['worker_id']:<10} {r['success']:<10} {r['failure']:<10} {rate:.1f}%")
        total_success += r['success']
        total_failure += r['failure']

    print("-" * 50)
    total = total_success + total_failure
    overall_rate = (total_success / total * 100) if total > 0 else 0
    print(f"{'Total':<10} {total_success:<10} {total_failure:<10} {overall_rate:.1f}%")

    # 네트워크 상태 확인
    stats = network_state.get_network_stats()
    print(f"\n최종 네트워크 상태:")
    print(f"  사용된 클라우드 자원: {max_cloud_capacity - network_state.available_cloud_capacity.value:.0f}")
    print(f"  남은 클라우드 자원: {network_state.available_cloud_capacity.value:.0f}")
    print(f"  클라우드 사용률: {network_state.get_cloud_utilization()*100:.1f}%")

    print(f"\n{'='*60}")
    if total_failure > 0:
        print("✅ SUCCESS: 일부 OFFLOAD가 실패했습니다 (자원 경쟁 발생)")
        print("   공유 클라우드 자원이 제대로 작동하고 있습니다!")
    else:
        print("⚠️  WARNING: 모든 OFFLOAD가 성공했습니다")
        print("   클라우드 자원이 너무 많거나 자원 공유가 작동하지 않을 수 있습니다")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()