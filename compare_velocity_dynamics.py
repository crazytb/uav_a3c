"""
다양한 velocity에서 channel dynamics 비교
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drl_framework.custom_env import CustomEnv
from drl_framework.params import ENV_PARAMS, REWARD_PARAMS
import math

def analyze_velocity(velocity, num_steps=5000):
    """특정 velocity에서 channel dynamics 분석"""

    env_params = ENV_PARAMS.copy()
    env_params['agent_velocities'] = velocity

    env = CustomEnv(
        max_comp_units=env_params['max_comp_units'],
        max_epoch_size=env_params['max_epoch_size'],
        max_queue_size=env_params['max_queue_size'],
        max_comp_units_for_cloud=env_params['max_comp_units_for_cloud'],
        reward_weights=env_params['reward_weights'],
        agent_velocities=velocity,
        reward_params=REWARD_PARAMS
    )

    obs, _ = env.reset()

    # Transition 확률 계산
    f_0 = 5.9e9
    speedoflight = 300000
    f_d = velocity/(3600*speedoflight)*f_0
    snr_thr = 15
    snr_ave = snr_thr + 10
    packettime = 100*1000/ENV_PARAMS['max_epoch_size']
    fdtp = f_d*packettime/1e6
    TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
    TRAN_00 = 1 - TRAN_01
    TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
    TRAN_11 = 1 - TRAN_10

    # 데이터 수집
    channel_history = []
    good_durations = []
    bad_durations = []

    current_state = env.channel_quality
    duration = 1

    for step in range(num_steps):
        channel_history.append(env.channel_quality)

        if step > 0:
            if channel_history[step] == current_state:
                duration += 1
            else:
                if current_state == 1:
                    good_durations.append(duration)
                else:
                    bad_durations.append(duration)
                current_state = channel_history[step]
                duration = 1

        # Random action
        action = np.random.choice([0, 1, 2])
        obs, reward, done, truncated, info = env.step(action)

        if done:
            obs, _ = env.reset()

    # 통계 계산
    channel_array = np.array(channel_history)
    avg_good_dur = np.mean(good_durations) if len(good_durations) > 0 else 0
    avg_bad_dur = np.mean(bad_durations) if len(bad_durations) > 0 else 0
    good_ratio = np.mean(channel_array)

    # Autocorrelation
    lag1_corr = np.corrcoef(channel_array[:-1], channel_array[1:])[0, 1] if len(channel_array) > 1 else 0

    return {
        'velocity': velocity,
        'doppler_freq': f_d,
        'TRAN_00': TRAN_00,
        'TRAN_01': TRAN_01,
        'TRAN_10': TRAN_10,
        'TRAN_11': TRAN_11,
        'avg_good_duration': avg_good_dur,
        'avg_bad_duration': avg_bad_dur,
        'good_ratio': good_ratio,
        'lag1_autocorr': lag1_corr,
        'channel_history': channel_history[:1000],  # 처음 1000개만
        'good_durations': good_durations[:100],  # 처음 100개만
        'bad_durations': bad_durations[:100]
    }

def main():
    print("=" * 70)
    print("🚗 Velocity에 따른 Channel Dynamics 비교")
    print("=" * 70)

    # 다양한 velocity 테스트
    velocities = [5, 10, 20, 30, 50, 70, 100]

    results = []

    for vel in velocities:
        print(f"\n📍 Analyzing velocity = {vel} km/h...")
        result = analyze_velocity(vel)
        results.append(result)

    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 비교 결과")
    print("=" * 70)

    print(f"\n{'Velocity':<10} {'Doppler':<12} {'P(G→G)':<10} {'Avg Good':<12} {'Avg Bad':<12} {'Lag-1 Corr':<12} {'학습성'}")
    print("-" * 90)

    for r in results:
        learnability = "✅" if r['lag1_autocorr'] > 0.5 and r['avg_good_duration'] > 5 else \
                      "⚠️" if r['lag1_autocorr'] > 0.3 and r['avg_good_duration'] > 3 else "❌"

        print(f"{r['velocity']:<10} {r['doppler_freq']:<12.2f} {r['TRAN_11']:<10.4f} "
              f"{r['avg_good_duration']:<12.2f} {r['avg_bad_duration']:<12.2f} "
              f"{r['lag1_autocorr']:<12.3f} {learnability}")

    print("\n" + "=" * 70)
    print("💡 상세 분석")
    print("=" * 70)

    for r in results:
        print(f"\n📍 Velocity = {r['velocity']} km/h")
        print(f"  Doppler frequency: {r['doppler_freq']:.2f} Hz")
        print(f"  Transition probabilities:")
        print(f"    - P(Bad→Bad):   {r['TRAN_00']:.4f}")
        print(f"    - P(Bad→Good):  {r['TRAN_01']:.4f}")
        print(f"    - P(Good→Bad):  {r['TRAN_10']:.4f}")
        print(f"    - P(Good→Good): {r['TRAN_11']:.4f}")
        print(f"  Duration:")
        print(f"    - Good state: {r['avg_good_duration']:.2f} steps")
        print(f"    - Bad state:  {r['avg_bad_duration']:.2f} steps")
        print(f"  Autocorrelation (lag-1): {r['lag1_autocorr']:.3f}")
        print(f"  Good state ratio: {r['good_ratio']*100:.1f}%")

        # 평가
        if r['lag1_autocorr'] > 0.5 and r['avg_good_duration'] > 5:
            print(f"  ✅ RNN 학습 가능: State가 {r['avg_good_duration']:.1f} steps 지속, 높은 상관관계")
        elif r['lag1_autocorr'] > 0.3:
            print(f"  ⚠️  학습 어려움: State 지속 짧거나 상관관계 중간")
        else:
            print(f"  ❌ 학습 불가능: 거의 랜덤 변화 (corr={r['lag1_autocorr']:.3f})")

    # Visualization
    print("\n📊 시각화 생성 중...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Transition probability vs velocity
    ax1 = fig.add_subplot(gs[0, :])
    vels = [r['velocity'] for r in results]
    ax1.plot(vels, [r['TRAN_11'] for r in results], 'o-', label='P(Good→Good)', linewidth=2)
    ax1.plot(vels, [r['TRAN_00'] for r in results], 's-', label='P(Bad→Bad)', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.set_xlabel('Velocity (km/h)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('State Persistence Probability vs Velocity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Duration vs velocity
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(vels, [r['avg_good_duration'] for r in results], 'o-', label='Good state duration', linewidth=2, color='green')
    ax2.plot(vels, [r['avg_bad_duration'] for r in results], 's-', label='Bad state duration', linewidth=2, color='red')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5 steps (학습 가능 기준)')
    ax2.set_xlabel('Velocity (km/h)', fontsize=12)
    ax2.set_ylabel('Duration (steps)', fontsize=12)
    ax2.set_title('Average State Duration vs Velocity', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Autocorrelation vs velocity
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(vels, [r['lag1_autocorr'] for r in results], 'o-', linewidth=2, color='purple')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='0.5 (학습 가능 기준)')
    ax3.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='0.3 (학습 어려움 기준)')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Velocity (km/h)', fontsize=12)
    ax3.set_ylabel('Lag-1 Autocorrelation', fontsize=12)
    ax3.set_title('Channel Predictability vs Velocity', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-9. Channel history for different velocities
    selected_velocities = [5, 10, 30, 50, 70, 100]
    for idx, vel in enumerate(selected_velocities):
        ax = fig.add_subplot(gs[3, idx % 3])
        result = next((r for r in results if r['velocity'] == vel), None)
        if result:
            history = result['channel_history'][:200]  # 처음 200 steps
            ax.step(range(len(history)), history, where='post', linewidth=1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(f'{vel} km/h (dur={result["avg_good_duration"]:.1f})', fontsize=10)
            ax.set_xlabel('Step', fontsize=9)
            ax.set_ylabel('Quality', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Bad', 'Good'])

    plt.suptitle('Channel Dynamics Analysis: Velocity Comparison', fontsize=16, fontweight='bold', y=0.995)

    filename = 'velocity_comparison_dynamics.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   → {filename} 저장 완료")

    # 추천
    print("\n" + "=" * 70)
    print("🎯 추천 Velocity")
    print("=" * 70)

    best = None
    for r in results:
        if r['lag1_autocorr'] > 0.5 and r['avg_good_duration'] > 5:
            if best is None or r['velocity'] > best['velocity']:
                best = r

    if best:
        print(f"\n✅ 추천: {best['velocity']} km/h")
        print(f"   - Good state 평균 {best['avg_good_duration']:.1f} steps 지속")
        print(f"   - Autocorrelation: {best['lag1_autocorr']:.3f}")
        print(f"   - RNN이 패턴 학습 가능")
    else:
        marginal = None
        for r in results:
            if r['lag1_autocorr'] > 0.3 and r['avg_good_duration'] > 3:
                if marginal is None or r['velocity'] > marginal['velocity']:
                    marginal = r

        if marginal:
            print(f"\n⚠️  타협안: {marginal['velocity']} km/h")
            print(f"   - Good state 평균 {marginal['avg_good_duration']:.1f} steps 지속")
            print(f"   - Autocorrelation: {marginal['lag1_autocorr']:.3f}")
            print(f"   - 학습 가능하지만 어려움")
        else:
            print(f"\n❌ 모든 velocity에서 학습 어려움")
            print(f"   - Channel model 자체를 변경 권장")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
