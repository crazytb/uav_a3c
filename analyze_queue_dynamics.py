"""
큐 동적 분석: 왜 자원이 충분한지 확인
"""

import numpy as np
from drl_framework.custom_env import CustomEnv
from drl_framework.params import ENV_PARAMS, REWARD_PARAMS

def analyze_queue_dynamics():
    env = CustomEnv(
        max_comp_units=ENV_PARAMS['max_comp_units'],
        max_epoch_size=ENV_PARAMS['max_epoch_size'],
        max_queue_size=ENV_PARAMS['max_queue_size'],
        max_comp_units_for_cloud=ENV_PARAMS['max_comp_units_for_cloud'],
        reward_weights=ENV_PARAMS['reward_weights'],
        agent_velocities=ENV_PARAMS['agent_velocities'],
        reward_params=REWARD_PARAMS
    )

    print("=" * 70)
    print("🔍 큐 동적 분석 - LOCAL 액션만 반복")
    print("=" * 70)
    print(f"\n환경 설정:")
    print(f"  max_comp_units: {ENV_PARAMS['max_comp_units']}")
    print(f"  max_queue_size: {ENV_PARAMS['max_queue_size']}")
    print(f"  max_proc_times: ~{int(np.ceil(ENV_PARAMS['max_epoch_size'] / 10))}")

    obs, _ = env.reset()

    print(f"\n초기 상태:")
    print(f"  available_computation_units: {env.available_computation_units}")

    # LOCAL만 계속 선택
    step = 0
    max_steps = 50

    print("\n" + "=" * 70)
    print("Step-by-Step 진행 (LOCAL만 시도)")
    print("=" * 70)
    print(f"{'Step':>4} | {'Queue':>6} | {'Avail':>6} | {'MEC_used':>8} | {'Action':>8} | {'Success':>7}")
    print("-" * 70)

    for step in range(max_steps):
        queue_size = env.queue_comp_units
        available = env.available_computation_units
        mec_used_slots = np.count_nonzero(env.mec_comp_units)

        # LOCAL 액션 가능 여부 확인
        local_possible = (
            (env.available_computation_units >= env.queue_comp_units) and
            (env.mec_comp_units[env.mec_comp_units == 0].size > 0) and
            (env.queue_comp_units > 0)
        )

        action = 0 if local_possible else 2  # LOCAL or DISCARD
        action_name = "LOCAL" if action == 0 else "DISCARD"

        obs, reward, done, truncated, info = env.step(action)

        success = "YES" if local_possible else "NO"

        print(f"{step:>4} | {queue_size:>6} | {available:>6} | {mec_used_slots:>8} | {action_name:>8} | {success:>7}")

        if step == 10 or step == 20 or step == 30:
            print(f"     MEC queue content: {env.mec_comp_units[:10]}")
            print(f"     MEC proc times:    {env.mec_proc_times[:10]}")

        if done:
            break

    print("\n" + "=" * 70)
    print("📊 최종 통계")
    print("=" * 70)
    print(f"MEC 큐 사용 현황:")
    print(f"  사용 중인 슬롯: {np.count_nonzero(env.mec_comp_units)} / {ENV_PARAMS['max_queue_size']}")
    print(f"  점유 중인 자원: {np.sum(env.mec_comp_units)}")
    print(f"  가용 자원: {env.available_computation_units}")
    print(f"\n  MEC comp_units: {env.mec_comp_units}")
    print(f"  MEC proc_times: {env.mec_proc_times}")

    print("\n" + "=" * 70)
    print("💡 분석")
    print("=" * 70)

    # 처리 속도 vs 유입 속도
    avg_queue_size = 100  # 평균
    avg_proc_time = 5     # 평균

    print(f"\n이론적 분석:")
    print(f"  평균 작업 크기: ~{avg_queue_size} units")
    print(f"  평균 처리 시간: ~{avg_proc_time} steps")
    print(f"  처리 속도: ~{avg_queue_size / avg_proc_time:.1f} units/step")
    print(f"  유입 속도: ~{avg_queue_size} units/step")
    print(f"\n  → 유입 속도가 처리 속도의 {avg_proc_time}배!")
    print(f"  → 자원이 빠르게 고갈되어야 정상")

    if np.count_nonzero(env.mec_comp_units) < ENV_PARAMS['max_queue_size'] * 0.5:
        print("\n⚠️  큐가 반도 안 참! 자원 부족이 충분히 발생하지 않음")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_queue_dynamics()
