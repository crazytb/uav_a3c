"""
Ablation Study Configuration File
각 ablation 실험의 파라미터 설정을 정의
"""

# Baseline configuration (현재 최적 설정)
BASELINE_CONFIG = {
    'name': 'baseline',
    'description': 'Baseline configuration with all optimal settings',
    'use_recurrent': True,
    'use_layer_norm': True,
    'hidden_dim': 128,
    'entropy_coef': 0.05,
    'value_loss_coef': 0.25,
    'lr': 1e-4,
    'max_grad_norm': 2.0,
    'n_workers': 5,
    'target_episode_count': 2000,
    'max_comp_units_for_cloud': 1000,
    'agent_velocities': 50,  # 기본값 (각 worker는 5~25로 분산됨)
    'reward_scale': 0.05,
    'use_action_masking': True,
}

# Phase 1: Network Architecture Ablations
ARCHITECTURE_ABLATIONS = {
    'ablation_1_no_rnn': {
        'name': 'ablation_1_no_rnn',
        'description': 'Remove RNN (use feedforward ActorCritic)',
        'use_recurrent': False,
        'phase': 1,
        'priority': 'high',
    },

    'ablation_2_no_layer_norm': {
        'name': 'ablation_2_no_layer_norm',
        'description': 'Remove Layer Normalization',
        'use_layer_norm': False,
        'phase': 1,
        'priority': 'high',
    },

    'ablation_3_small_hidden': {
        'name': 'ablation_3_small_hidden',
        'description': 'Reduce hidden dimension to 64',
        'hidden_dim': 64,
        'phase': 1,
        'priority': 'medium',
    },

    'ablation_4_large_hidden': {
        'name': 'ablation_4_large_hidden',
        'description': 'Increase hidden dimension to 256',
        'hidden_dim': 256,
        'phase': 1,
        'priority': 'medium',
    },
}

# Phase 2: Learning Hyperparameter Ablations
HYPERPARAMETER_ABLATIONS = {
    'ablation_5_low_entropy': {
        'name': 'ablation_5_low_entropy',
        'description': 'Low exploration (entropy_coef=0.01)',
        'entropy_coef': 0.01,
        'phase': 2,
        'priority': 'medium',
    },

    'ablation_6_high_entropy': {
        'name': 'ablation_6_high_entropy',
        'description': 'High exploration (entropy_coef=0.1)',
        'entropy_coef': 0.1,
        'phase': 2,
        'priority': 'medium',
    },

    'ablation_7_medium_value_loss': {
        'name': 'ablation_7_medium_value_loss',
        'description': 'Medium value loss coefficient (0.5)',
        'value_loss_coef': 0.5,
        'phase': 2,
        'priority': 'low',
    },

    'ablation_8_high_value_loss': {
        'name': 'ablation_8_high_value_loss',
        'description': 'High value loss coefficient (1.0)',
        'value_loss_coef': 1.0,
        'phase': 2,
        'priority': 'low',
    },

    'ablation_9_low_lr': {
        'name': 'ablation_9_low_lr',
        'description': 'Lower learning rate (5e-5)',
        'lr': 5e-5,
        'phase': 2,
        'priority': 'low',
    },

    'ablation_10_high_lr': {
        'name': 'ablation_10_high_lr',
        'description': 'Higher learning rate (5e-4)',
        'lr': 5e-4,
        'phase': 2,
        'priority': 'low',
    },
}

# Phase 3: Environment Configuration Ablations
ENVIRONMENT_ABLATIONS = {
    'ablation_11_limited_cloud': {
        'name': 'ablation_11_limited_cloud',
        'description': 'Limited cloud resources (500 units)',
        'max_comp_units_for_cloud': 500,
        'phase': 3,
        'priority': 'medium',
    },

    'ablation_12_abundant_cloud': {
        'name': 'ablation_12_abundant_cloud',
        'description': 'Abundant cloud resources (2000 units)',
        'max_comp_units_for_cloud': 2000,
        'phase': 3,
        'priority': 'medium',
    },

    'ablation_13_low_velocity': {
        'name': 'ablation_13_low_velocity',
        'description': 'Low UAV velocity (30 km/h baseline)',
        'agent_velocities': 30,
        'phase': 3,
        'priority': 'medium',
    },

    'ablation_14_high_velocity': {
        'name': 'ablation_14_high_velocity',
        'description': 'High UAV velocity (100 km/h baseline)',
        'agent_velocities': 100,
        'phase': 3,
        'priority': 'medium',
    },

    'ablation_15_few_workers': {
        'name': 'ablation_15_few_workers',
        'description': 'Fewer workers (n_workers=3)',
        'n_workers': 3,
        'phase': 3,
        'priority': 'high',
    },

    'ablation_16_many_workers': {
        'name': 'ablation_16_many_workers',
        'description': 'More workers (n_workers=10)',
        'n_workers': 10,
        'phase': 3,
        'priority': 'high',
    },
}

# Phase 4: Reward Design Ablations (Optional)
REWARD_ABLATIONS = {
    'ablation_17_low_reward_scale': {
        'name': 'ablation_17_low_reward_scale',
        'description': 'Lower reward scaling (0.01)',
        'reward_scale': 0.01,
        'phase': 4,
        'priority': 'low',
    },

    'ablation_18_high_reward_scale': {
        'name': 'ablation_18_high_reward_scale',
        'description': 'Higher reward scaling (0.1)',
        'reward_scale': 0.1,
        'phase': 4,
        'priority': 'low',
    },
}

# Combine all ablations
ALL_ABLATIONS = {}
ALL_ABLATIONS.update(ARCHITECTURE_ABLATIONS)
ALL_ABLATIONS.update(HYPERPARAMETER_ABLATIONS)
ALL_ABLATIONS.update(ENVIRONMENT_ABLATIONS)
ALL_ABLATIONS.update(REWARD_ABLATIONS)

def get_config(ablation_name):
    """
    특정 ablation의 전체 설정을 반환
    baseline + ablation의 변경사항을 merge
    """
    config = BASELINE_CONFIG.copy()

    if ablation_name == 'baseline':
        return config

    if ablation_name not in ALL_ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    ablation = ALL_ABLATIONS[ablation_name]

    # Merge ablation changes into baseline
    for key, value in ablation.items():
        if key not in ['name', 'description', 'phase', 'priority']:
            config[key] = value

    config['name'] = ablation_name
    config['description'] = ablation.get('description', '')

    return config

def get_ablations_by_phase(phase):
    """특정 phase의 ablation들을 반환"""
    result = {}
    for name, ablation in ALL_ABLATIONS.items():
        if ablation.get('phase') == phase:
            result[name] = ablation
    return result

def get_ablations_by_priority(priority):
    """특정 priority의 ablation들을 반환"""
    result = {}
    for name, ablation in ALL_ABLATIONS.items():
        if ablation.get('priority') == priority:
            result[name] = ablation
    return result

def list_all_ablations():
    """모든 ablation 목록 출력"""
    print("="*80)
    print("Available Ablation Studies")
    print("="*80)
    print()

    for phase in [1, 2, 3, 4]:
        ablations = get_ablations_by_phase(phase)
        if ablations:
            print(f"Phase {phase}:")
            for name, ablation in ablations.items():
                priority = ablation.get('priority', 'unknown')
                desc = ablation.get('description', '')
                print(f"  [{priority:>6}] {name:30s} - {desc}")
            print()

if __name__ == "__main__":
    # Test configuration retrieval
    list_all_ablations()

    print("="*80)
    print("Example Configuration: ablation_1_no_rnn")
    print("="*80)
    config = get_config('ablation_1_no_rnn')
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
