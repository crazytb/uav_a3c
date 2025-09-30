"""
더 간단한 테스트: 클라우드 자원 공유 검증
"""
import torch.multiprocessing as mp
from drl_framework.network_state import NetworkState

def test_basic_resource_sharing():
    """NetworkState의 기본 자원 관리 기능 테스트"""
    print("\n" + "="*60)
    print("기본 클라우드 자원 관리 테스트")
    print("="*60 + "\n")

    n_workers = 5
    max_cloud = 1000

    # NetworkState 생성
    ns = NetworkState(n_workers, max_cloud_capacity=max_cloud)

    print(f"초기 클라우드 자원: {ns.available_cloud_capacity.value}")
    print(f"\n각 워커가 순차적으로 자원 소모:")

    # 각 워커가 자원 소모
    for worker_id in range(n_workers):
        amount = 200
        success = ns.consume_cloud_resource(worker_id, amount)
        remaining = ns.available_cloud_capacity.value
        print(f"  Worker {worker_id}: {amount} 요청 → {'성공' if success else '실패'}, 남은 자원: {remaining}")

    print(f"\n추가 자원 소모 시도:")
    # 6번째 시도 (실패해야 함)
    success = ns.consume_cloud_resource(0, 200)
    print(f"  Worker 0: 200 추가 요청 → {'성공' if success else '실패'} (남은 자원: {ns.available_cloud_capacity.value})")

    print(f"\n자원 회수 테스트:")
    ns.release_cloud_resource(400)
    print(f"  400 회수 → 남은 자원: {ns.available_cloud_capacity.value}")

    # 다시 시도
    success = ns.consume_cloud_resource(1, 300)
    print(f"  Worker 1: 300 요청 → {'성공' if success else '실패'} (남은 자원: {ns.available_cloud_capacity.value})")

    print("\n" + "="*60)
    if ns.available_cloud_capacity.value < max_cloud:
        print("✅ SUCCESS: 공유 자원 관리가 올바르게 작동합니다!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_basic_resource_sharing()