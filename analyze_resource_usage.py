"""
자원 사용 패턴 분석 스크립트
- 각 worker의 로컬 자원이 실제로 부족해지는지 확인
- OFFLOAD가 필요한 상황이 발생하는지 검증
"""

import numpy as np
from drl_framework.custom_env import CustomEnv
from drl_framework.params import ENV_PARAMS, REWARD_PARAMS

def analyze_resource_patterns(num_episodes=100, max_steps=100):
    """환경에서 자원 사용 패턴 분석"""

    env = CustomEnv(
        max_comp_units=ENV_PARAMS['max_comp_units'],
        max_epoch_size=ENV_PARAMS['max_epoch_size'],
        max_queue_size=ENV_PARAMS['max_queue_size'],
        max_comp_units_for_cloud=ENV_PARAMS['max_comp_units_for_cloud'],
        reward_weights=ENV_PARAMS['reward_weights'],
        agent_velocities=ENV_PARAMS['agent_velocities'],
        reward_params=REWARD_PARAMS
    )

    # 통계 수집
    stats = {
        'local_possible': [],
        'offload_possible': [],
        'both_possible': [],
        'neither_possible': [],
        'available_resources': [],
        'queue_sizes': [],
        'channel_good': [],
        'mec_queue_full': [],
        'cloud_queue_full': []
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()

        for step in range(max_steps):
            # 현재 상태 기록
            stats['available_resources'].append(env.available_computation_units)
            stats['queue_sizes'].append(env.queue_comp_units)
            stats['channel_good'].append(1 if env.channel_quality == 1 else 0)

            # MEC/Cloud 큐 상태
            mec_full = (env.mec_comp_units[env.mec_comp_units == 0].size == 0)
            cloud_full = (env.cloud_comp_units[env.cloud_comp_units == 0].size == 0)
            stats['mec_queue_full'].append(1 if mec_full else 0)
            stats['cloud_queue_full'].append(1 if cloud_full else 0)

            # 각 액션의 가능 여부 확인
            local_possible = (
                (env.available_computation_units >= env.queue_comp_units) and
                (env.mec_comp_units[env.mec_comp_units == 0].size > 0) and
                (env.queue_comp_units > 0)
            )

            offload_possible = (
                (env.available_computation_units_for_cloud >= env.queue_comp_units) and
                (env.cloud_comp_units[env.cloud_comp_units == 0].size > 0) and
                (env.queue_comp_units > 0) and
                (env.channel_quality == 1)
            )

            stats['local_possible'].append(1 if local_possible else 0)
            stats['offload_possible'].append(1 if offload_possible else 0)
            stats['both_possible'].append(1 if (local_possible and offload_possible) else 0)
            stats['neither_possible'].append(1 if (not local_possible and not offload_possible) else 0)

            # 랜덤 액션으로 환경 진행
            valid_actions = env.get_valid_actions()
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)

            if done:
                break

    # 결과 출력
    print("=" * 70)
    print("🔍 자원 사용 패턴 분석 결과")
    print("=" * 70)
    print(f"\n📊 총 분석 스텝 수: {len(stats['local_possible'])}")

    print("\n" + "=" * 70)
    print("💾 자원 가용성 분석")
    print("=" * 70)
    print(f"평균 가용 자원: {np.mean(stats['available_resources']):.2f} / {ENV_PARAMS['max_comp_units']}")
    print(f"최소 가용 자원: {np.min(stats['available_resources'])}")
    print(f"최대 가용 자원: {np.max(stats['available_resources'])}")
    print(f"가용 자원 표준편차: {np.std(stats['available_resources']):.2f}")

    print("\n" + "=" * 70)
    print("📦 작업 크기 분석")
    print("=" * 70)
    print(f"평균 큐 크기 (comp_units): {np.mean(stats['queue_sizes']):.2f}")
    print(f"최소 큐 크기: {np.min(stats['queue_sizes'])}")
    print(f"최대 큐 크기: {np.max(stats['queue_sizes'])}")
    print(f"큐 크기 표준편차: {np.std(stats['queue_sizes']):.2f}")

    # 자원 부족 비율
    resource_shortage = sum(1 for i in range(len(stats['available_resources']))
                           if stats['available_resources'][i] < stats['queue_sizes'][i])
    print(f"\n⚠️  자원 부족 발생 비율: {resource_shortage / len(stats['available_resources']) * 100:.1f}%")
    print(f"   (가용 자원 < 큐 크기인 경우)")

    print("\n" + "=" * 70)
    print("🎬 액션 가능성 분석")
    print("=" * 70)
    local_pct = np.mean(stats['local_possible']) * 100
    offload_pct = np.mean(stats['offload_possible']) * 100
    both_pct = np.mean(stats['both_possible']) * 100
    neither_pct = np.mean(stats['neither_possible']) * 100

    print(f"LOCAL 가능:          {local_pct:5.1f}%")
    print(f"OFFLOAD 가능:        {offload_pct:5.1f}%")
    print(f"둘 다 가능:          {both_pct:5.1f}%")
    print(f"둘 다 불가능:        {neither_pct:5.1f}%")
    print(f"DISCARD만 가능:      {neither_pct:5.1f}%")

    print("\n" + "=" * 70)
    print("📡 채널 품질 분석")
    print("=" * 70)
    channel_good_pct = np.mean(stats['channel_good']) * 100
    print(f"채널 좋음 (quality=1): {channel_good_pct:.1f}%")
    print(f"채널 나쁨 (quality=0): {100-channel_good_pct:.1f}%")

    print("\n" + "=" * 70)
    print("🗄️  큐 포화 상태 분석")
    print("=" * 70)
    mec_full_pct = np.mean(stats['mec_queue_full']) * 100
    cloud_full_pct = np.mean(stats['cloud_queue_full']) * 100
    print(f"MEC 큐 가득 참:      {mec_full_pct:.1f}%")
    print(f"Cloud 큐 가득 참:    {cloud_full_pct:.1f}%")

    print("\n" + "=" * 70)
    print("🎯 OFFLOAD 필요성 분석")
    print("=" * 70)

    # OFFLOAD가 실제로 필요한 경우 = LOCAL 불가능하지만 OFFLOAD 가능
    offload_needed = sum(1 for i in range(len(stats['local_possible']))
                        if not stats['local_possible'][i] and stats['offload_possible'][i])
    offload_needed_pct = offload_needed / len(stats['local_possible']) * 100

    print(f"OFFLOAD 필수 상황:   {offload_needed_pct:.1f}%")
    print(f"   (LOCAL 불가능 & OFFLOAD 가능)")

    # LOCAL만 가능한 경우
    local_only = sum(1 for i in range(len(stats['local_possible']))
                    if stats['local_possible'][i] and not stats['offload_possible'][i])
    local_only_pct = local_only / len(stats['local_possible']) * 100
    print(f"LOCAL만 가능:        {local_only_pct:.1f}%")

    # OFFLOAD만 가능한 경우
    offload_only = sum(1 for i in range(len(stats['local_possible']))
                      if not stats['local_possible'][i] and stats['offload_possible'][i])
    offload_only_pct = offload_only / len(stats['local_possible']) * 100
    print(f"OFFLOAD만 가능:      {offload_only_pct:.1f}%")

    print("\n" + "=" * 70)
    print("💡 결론")
    print("=" * 70)

    if offload_needed_pct < 5:
        print("⚠️  OFFLOAD가 필요한 상황이 거의 발생하지 않습니다!")
        print("   → 환경 설계가 OFFLOAD 학습을 유도하지 못함")
    elif offload_needed_pct < 20:
        print("⚠️  OFFLOAD가 필요한 상황이 드뭅니다.")
        print("   → OFFLOAD 학습에 충분한 샘플 부족 가능성")
    else:
        print("✅ OFFLOAD가 필요한 상황이 충분히 발생합니다.")

    if both_pct > 50:
        print("✅ 대부분의 경우 선택지가 있어 학습 가능합니다.")

    if neither_pct > 30:
        print("⚠️  선택지가 없는 상황이 많습니다 (DISCARD 강제).")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_resource_patterns(num_episodes=100, max_steps=100)
