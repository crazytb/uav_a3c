"""
Channel Quality Dynamics 분석
- Transition 확률
- 지속 시간 (Good/Bad state duration)
- 예측 가능성
- RNN이 학습할 수 있는지 평가
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drl_framework.custom_env import CustomEnv
from drl_framework.params import ENV_PARAMS, REWARD_PARAMS

def analyze_channel_dynamics(num_steps=10000):
    """Channel quality의 동적 특성 분석"""

    env = CustomEnv(
        max_comp_units=ENV_PARAMS['max_comp_units'],
        max_epoch_size=ENV_PARAMS['max_epoch_size'],
        max_queue_size=ENV_PARAMS['max_queue_size'],
        max_comp_units_for_cloud=ENV_PARAMS['max_comp_units_for_cloud'],
        reward_weights=ENV_PARAMS['reward_weights'],
        agent_velocities=ENV_PARAMS['agent_velocities'],
        reward_params=REWARD_PARAMS
    )

    obs, _ = env.reset()

    # 데이터 수집
    channel_history = []
    transitions = {'00': 0, '01': 0, '10': 0, '11': 0}
    good_durations = []
    bad_durations = []

    current_state = env.channel_quality
    duration = 1

    print("=" * 70)
    print("📡 Channel Quality Dynamics 분석")
    print("=" * 70)

    # 환경 파라미터 출력
    print(f"\n⚙️  환경 설정:")
    print(f"  Agent velocity: {env.agent_velocities} km/h")
    print(f"  Carrier freq: 5.9 GHz (IEEE 802.11bd)")
    print(f"  Packet time: {100*1000/ENV_PARAMS['max_epoch_size']:.1f} μs")

    # Doppler frequency 계산
    velocity = env.agent_velocities
    f_0 = 5.9e9
    speedoflight = 300000
    f_d = velocity/(3600*speedoflight)*f_0
    print(f"  Doppler frequency: {f_d:.2f} Hz")

    # Transition 확률 계산 (코드에서 복사)
    import math
    snr_thr = 15
    snr_ave = snr_thr + 10
    packettime = 100*1000/ENV_PARAMS['max_epoch_size']
    fdtp = f_d*packettime/1e6
    TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
    TRAN_00 = 1 - TRAN_01
    TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
    TRAN_11 = 1 - TRAN_10

    print(f"\n📊 이론적 Transition 확률:")
    print(f"  P(Bad→Bad):   {TRAN_00:.4f}")
    print(f"  P(Bad→Good):  {TRAN_01:.4f}")
    print(f"  P(Good→Bad):  {TRAN_10:.4f}")
    print(f"  P(Good→Good): {TRAN_11:.4f}")

    print(f"\n⏱️  데이터 수집 중 ({num_steps} steps)...")

    for step in range(num_steps):
        channel_history.append(env.channel_quality)

        # Transition 기록
        if step > 0:
            prev = channel_history[step-1]
            curr = channel_history[step]
            key = f"{prev}{curr}"
            transitions[key] += 1

            # Duration 추적
            if prev == curr:
                duration += 1
            else:
                if prev == 1:
                    good_durations.append(duration)
                else:
                    bad_durations.append(duration)
                duration = 1

        # Random action으로 환경 진행
        action = np.random.choice([0, 1, 2])
        obs, reward, done, truncated, info = env.step(action)

        if done:
            obs, _ = env.reset()

    # 결과 분석
    print("\n" + "=" * 70)
    print("📈 실험적 Transition 확률")
    print("=" * 70)

    total_00_01 = transitions['00'] + transitions['01']
    total_10_11 = transitions['10'] + transitions['11']

    emp_TRAN_00 = transitions['00'] / total_00_01 if total_00_01 > 0 else 0
    emp_TRAN_01 = transitions['01'] / total_00_01 if total_00_01 > 0 else 0
    emp_TRAN_10 = transitions['10'] / total_10_11 if total_10_11 > 0 else 0
    emp_TRAN_11 = transitions['11'] / total_10_11 if total_10_11 > 0 else 0

    print(f"  P(Bad→Bad):   {emp_TRAN_00:.4f} (이론: {TRAN_00:.4f})")
    print(f"  P(Bad→Good):  {emp_TRAN_01:.4f} (이론: {TRAN_01:.4f})")
    print(f"  P(Good→Bad):  {emp_TRAN_10:.4f} (이론: {TRAN_10:.4f})")
    print(f"  P(Good→Good): {emp_TRAN_11:.4f} (이론: {TRAN_11:.4f})")

    print("\n" + "=" * 70)
    print("⏳ State Duration 분석")
    print("=" * 70)

    if len(good_durations) > 0:
        print(f"\nGood State 지속 시간:")
        print(f"  평균: {np.mean(good_durations):.2f} steps")
        print(f"  중앙값: {np.median(good_durations):.2f} steps")
        print(f"  최소: {np.min(good_durations)} steps")
        print(f"  최대: {np.max(good_durations)} steps")
        print(f"  표준편차: {np.std(good_durations):.2f} steps")

    if len(bad_durations) > 0:
        print(f"\nBad State 지속 시간:")
        print(f"  평균: {np.mean(bad_durations):.2f} steps")
        print(f"  중앙값: {np.median(bad_durations):.2f} steps")
        print(f"  최소: {np.min(bad_durations)} steps")
        print(f"  최대: {np.max(bad_durations)} steps")
        print(f"  표준편차: {np.std(bad_durations):.2f} steps")

    print("\n" + "=" * 70)
    print("📊 전체 통계")
    print("=" * 70)

    good_ratio = np.mean(channel_history)
    print(f"\nGood state 비율: {good_ratio*100:.2f}%")
    print(f"Bad state 비율: {(1-good_ratio)*100:.2f}%")

    # Autocorrelation 분석
    print("\n" + "=" * 70)
    print("🔄 Autocorrelation 분석 (예측 가능성)")
    print("=" * 70)

    channel_array = np.array(channel_history)
    for lag in [1, 2, 3, 5, 10, 20]:
        if lag < len(channel_array):
            corr = np.corrcoef(channel_array[:-lag], channel_array[lag:])[0, 1]
            print(f"  Lag {lag:2d} steps: {corr:.4f}", end="")
            if corr > 0.7:
                print(" ← 강한 상관관계!")
            elif corr > 0.3:
                print(" ← 중간 상관관계")
            else:
                print(" ← 약한 상관관계")

    print("\n" + "=" * 70)
    print("🎯 RNN 학습 가능성 평가")
    print("=" * 70)

    # 평가 기준
    avg_good_dur = np.mean(good_durations) if len(good_durations) > 0 else 0
    avg_bad_dur = np.mean(bad_durations) if len(bad_durations) > 0 else 0

    print(f"\n1. State Persistence (지속성)")
    if avg_good_dur > 5:
        print(f"   ✅ Good state 평균 {avg_good_dur:.1f} steps 지속")
        print(f"      → RNN이 패턴 학습 가능")
    else:
        print(f"   ⚠️  Good state 평균 {avg_good_dur:.1f} steps만 지속")
        print(f"      → 너무 빠른 변화, 학습 어려움")

    print(f"\n2. Predictability (예측 가능성)")
    lag1_corr = np.corrcoef(channel_array[:-1], channel_array[1:])[0, 1]
    if lag1_corr > 0.7:
        print(f"   ✅ Lag-1 autocorr = {lag1_corr:.3f}")
        print(f"      → 현재 state로 다음 state 예측 가능")
    elif lag1_corr > 0.3:
        print(f"   ⚠️  Lag-1 autocorr = {lag1_corr:.3f}")
        print(f"      → 예측 가능하지만 불확실성 있음")
    else:
        print(f"   ❌ Lag-1 autocorr = {lag1_corr:.3f}")
        print(f"      → 거의 랜덤, RNN도 학습 어려움")

    print(f"\n3. Stationary Distribution (정상 분포)")
    if abs(good_ratio - 0.5) < 0.1:
        print(f"   ✅ Good/Bad 균형적 ({good_ratio*100:.1f}% / {(1-good_ratio)*100:.1f}%)")
        print(f"      → 둘 다 충분한 샘플")
    elif good_ratio > 0.7:
        print(f"   ⚠️  Good state 지배적 ({good_ratio*100:.1f}%)")
        print(f"      → OFFLOAD 위험 낮음, 학습 쉬움")
    else:
        print(f"   ⚠️  Bad state 지배적 ({(1-good_ratio)*100:.1f}%)")
        print(f"      → OFFLOAD 위험 높음")

    print("\n" + "=" * 70)
    print("💡 결론 및 제안")
    print("=" * 70)

    if avg_good_dur > 5 and lag1_corr > 0.5:
        print("\n✅ RNN이 channel pattern을 학습할 수 있습니다!")
        print(f"   - State가 충분히 오래 지속 ({avg_good_dur:.1f} steps)")
        print(f"   - 높은 자기상관 (corr={lag1_corr:.3f})")
        print("\n   제안: Statistical features로도 충분할 수 있음")
        print(f"   - 'avg_channel_last_5': 지난 5 step 평균")
        print(f"   - 'current_duration': 현 state 지속 시간")
    else:
        print("\n⚠️  Channel이 너무 빠르게 변합니다.")
        print(f"   - State 지속 시간: {avg_good_dur:.1f} steps")
        print(f"   - Autocorrelation: {lag1_corr:.3f}")
        print("\n   제안: Perfect CSI를 주거나, velocity를 낮추세요")

    # Visualization
    print("\n📊 시각화 생성 중...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 1. Channel state over time
    axes[0, 0].plot(channel_history[:500], linewidth=0.5)
    axes[0, 0].set_title('Channel Quality (first 500 steps)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Quality (0=Bad, 1=Good)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. State duration histogram
    if len(good_durations) > 0 and len(bad_durations) > 0:
        axes[0, 1].hist(good_durations, bins=30, alpha=0.6, label='Good', color='green')
        axes[0, 1].hist(bad_durations, bins=30, alpha=0.6, label='Bad', color='red')
        axes[0, 1].set_title('State Duration Distribution')
        axes[0, 1].set_xlabel('Duration (steps)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Transition matrix
    trans_matrix = np.array([
        [emp_TRAN_00, emp_TRAN_01],
        [emp_TRAN_10, emp_TRAN_11]
    ])
    im = axes[1, 0].imshow(trans_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1, 0].set_title('Transition Probability Matrix')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Bad', 'Good'])
    axes[1, 0].set_yticklabels(['Bad', 'Good'])
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{trans_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black")
    plt.colorbar(im, ax=axes[1, 0])

    # 4. Autocorrelation
    lags = range(1, 51)
    autocorrs = []
    for lag in lags:
        if lag < len(channel_array):
            corr = np.corrcoef(channel_array[:-lag], channel_array[lag:])[0, 1]
            autocorrs.append(corr)
    axes[1, 1].plot(lags[:len(autocorrs)], autocorrs)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 1].set_title('Autocorrelation Function')
    axes[1, 1].set_xlabel('Lag (steps)')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Moving average
    window = 10
    moving_avg = np.convolve(channel_history[:1000], np.ones(window)/window, mode='valid')
    axes[2, 0].plot(moving_avg)
    axes[2, 0].axhline(y=0.5, color='r', linestyle='--', label='50%')
    axes[2, 0].set_title(f'Moving Average (window={window}, first 1000 steps)')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Avg Channel Quality')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. State ratio pie chart
    axes[2, 1].pie([1-good_ratio, good_ratio],
                   labels=['Bad', 'Good'],
                   colors=['red', 'green'],
                   autopct='%1.1f%%',
                   startangle=90)
    axes[2, 1].set_title('Overall State Distribution')

    plt.tight_layout()
    filename = 'channel_dynamics_analysis.png'
    plt.savefig(filename, dpi=150)
    print(f"   → {filename} 저장 완료")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_channel_dynamics(num_steps=10000)
