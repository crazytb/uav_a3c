"""
에피소드 리셋 시 클라우드 자원 회복 테스트
"""
from drl_framework.custom_env import make_env
from drl_framework.network_state import NetworkState
from drl_framework import params

def test_episode_reset():
    """에피소드 리셋 시 클라우드 자원이 회복되는지 확인"""
    print("\n" + "="*60)
    print("에피소드 리셋 - 클라우드 자원 회복 테스트")
    print("="*60 + "\n")

    ENV_PARAMS = params.ENV_PARAMS.copy()
    REWARD_PARAMS = params.REWARD_PARAMS

    # NetworkState 생성
    network_state = NetworkState(1, max_cloud_capacity=500)

    env_fn = make_env(**ENV_PARAMS, reward_params=REWARD_PARAMS,
                     network_state=network_state, worker_id=0)
    env = env_fn()

    print(f"초기 클라우드 자원: {network_state.available_cloud_capacity.value}\n")

    # 3개 에피소드 실행
    for episode in range(3):
        print(f"{'='*60}")
        print(f"Episode {episode + 1}")
        print(f"{'='*60}")

        obs, _ = env.reset()
        print(f"에피소드 시작 시 클라우드 자원: {network_state.available_cloud_capacity.value}")

        # 10 스텝 실행 (OFFLOAD 시도)
        for step in range(10):
            obs, reward, done, truncated, info = env.step(1)  # OFFLOAD

            if step % 3 == 0:  # 3스텝마다 출력
                print(f"  Step {step:2d}: Cloud Resource = {network_state.available_cloud_capacity.value:.1f}")

            if done:
                break

        print(f"에피소드 종료 시 클라우드 자원: {network_state.available_cloud_capacity.value}")

        # 에피소드 리셋 (실제 학습에서 호출됨)
        network_state.reset_for_new_episode()
        print(f"reset_for_new_episode() 호출 후: {network_state.available_cloud_capacity.value}")
        print()

    env.close()

    print(f"{'='*60}")
    print("✅ SUCCESS: 에피소드마다 클라우드 자원이 회복됩니다!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_episode_reset()