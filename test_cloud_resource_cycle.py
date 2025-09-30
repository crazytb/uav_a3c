"""
클라우드 자원이 제대로 소모되고 회수되는지 확인
"""
import torch.multiprocessing as mp
from drl_framework.custom_env import make_env
from drl_framework.network_state import NetworkState
from drl_framework import params
import time

def single_worker_test():
    """단일 워커로 자원 순환 테스트"""
    print("\n" + "="*60)
    print("단일 워커 클라우드 자원 순환 테스트")
    print("="*60 + "\n")

    ENV_PARAMS = params.ENV_PARAMS.copy()
    REWARD_PARAMS = params.REWARD_PARAMS

    # NetworkState 생성 (작은 자원으로 테스트)
    network_state = NetworkState(1, max_cloud_capacity=500)

    env_fn = make_env(**ENV_PARAMS, reward_params=REWARD_PARAMS,
                     network_state=network_state, worker_id=0)
    env = env_fn()

    obs, _ = env.reset()

    print(f"초기 클라우드 자원: {network_state.available_cloud_capacity.value}")

    offload_attempts = 0
    offload_success = 0
    offload_failures = 0

    resource_log = []

    # 20 스텝 실행
    for step in range(20):
        # OFFLOAD 시도
        obs, reward, done, truncated, info = env.step(1)  # action=1: OFFLOAD

        offload_attempts += 1
        if env.offload_success == 1:
            offload_success += 1
        else:
            offload_failures += 1

        current_resource = network_state.available_cloud_capacity.value
        resource_log.append(current_resource)

        print(f"Step {step:2d}: Reward={reward:6.1f}, "
              f"Offload={'SUCCESS' if env.offload_success else 'FAIL   '}, "
              f"Cloud Resource={current_resource:6.1f}")

        if done:
            obs, _ = env.reset()
            print(f"  └─ Episode ended, reset")

    env.close()

    print(f"\n{'='*60}")
    print(f"결과:")
    print(f"  OFFLOAD 시도: {offload_attempts}")
    print(f"  성공: {offload_success}")
    print(f"  실패: {offload_failures}")
    print(f"  최종 클라우드 자원: {network_state.available_cloud_capacity.value}")
    print(f"  초기 대비 변화: {500 - network_state.available_cloud_capacity.value}")

    # 자원 변화 추이
    print(f"\n자원 변화 추이:")
    print(f"  최소값: {min(resource_log):.1f}")
    print(f"  최대값: {max(resource_log):.1f}")
    print(f"  평균값: {sum(resource_log)/len(resource_log):.1f}")

    # 문제 진단
    print(f"\n{'='*60}")
    if network_state.available_cloud_capacity.value < 100:
        print("⚠️  WARNING: 클라우드 자원이 거의 고갈됨!")
        print("   자원 회수가 제대로 안되고 있을 가능성이 높습니다.")
    elif min(resource_log) < 100:
        print("⚠️  WARNING: 중간에 자원이 크게 감소했습니다.")
        print("   자원 회수 속도가 소모 속도를 따라가지 못합니다.")
    else:
        print("✅ SUCCESS: 클라우드 자원이 적절히 순환되고 있습니다.")
    print(f"{'='*60}\n")


def multi_worker_test():
    """다중 워커로 자원 경쟁 테스트"""
    print("\n" + "="*60)
    print("다중 워커 클라우드 자원 경쟁 테스트")
    print("="*60 + "\n")

    # 짧은 테스트
    mp.set_start_method('spawn', force=True)

    n_workers = 3
    network_state = NetworkState(n_workers, max_cloud_capacity=300)

    print(f"워커 수: {n_workers}")
    print(f"초기 클라우드 자원: {network_state.available_cloud_capacity.value}")
    print(f"각 워커가 10스텝씩 OFFLOAD 시도\n")

    # 간단한 워커 함수
    def worker_func(worker_id, network_state):
        ENV_PARAMS = params.ENV_PARAMS.copy()
        REWARD_PARAMS = params.REWARD_PARAMS

        env_fn = make_env(**ENV_PARAMS, reward_params=REWARD_PARAMS,
                         network_state=network_state, worker_id=worker_id)
        env = env_fn()
        env.reset()

        for step in range(10):
            obs, reward, done, truncated, info = env.step(1)  # OFFLOAD
            time.sleep(0.05)  # 약간의 지연
            if done:
                env.reset()

        env.close()

    processes = []
    for wid in range(n_workers):
        p = mp.Process(target=worker_func, args=(wid, network_state))
        p.start()
        processes.append(p)

    # 주기적으로 자원 모니터링
    resources_over_time = []
    for _ in range(15):
        time.sleep(0.1)
        resources_over_time.append(network_state.available_cloud_capacity.value)
        print(f"  현재 클라우드 자원: {network_state.available_cloud_capacity.value:.1f}")

    for p in processes:
        p.join()

    print(f"\n최종 클라우드 자원: {network_state.available_cloud_capacity.value}")
    print(f"\n자원 변화:")
    print(f"  최소값: {min(resources_over_time):.1f}")
    print(f"  최대값: {max(resources_over_time):.1f}")

    print(f"\n{'='*60}")
    if network_state.available_cloud_capacity.value < 50:
        print("⚠️  CRITICAL: 다중 워커 환경에서 자원 고갈!")
    elif min(resources_over_time) < 50:
        print("⚠️  WARNING: 순간적으로 자원이 크게 감소했습니다.")
    else:
        print("✅ SUCCESS: 다중 워커에서도 자원이 적절히 관리됩니다.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 단일 워커 테스트
    single_worker_test()

    # 다중 워커 테스트
    # multi_worker_test()  # 시간이 걸리므로 주석 처리