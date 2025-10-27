"""
Check if models have Layer Normalization by comparing parameter counts.
"""

import torch
from drl_framework.networks import RecurrentActorCritic
from drl_framework.params import hidden_dim
from drl_framework.custom_env import make_env
from drl_framework.params import ENV_PARAMS
from drl_framework.utils import flatten_dict_values

# Get state/action dims
temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs = temp_env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n
temp_env.close()

print("=" * 60)
print("Model Parameter Count Comparison")
print("=" * 60)

# Create models with and without LayerNorm
model_with_ln = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=True)
model_without_ln = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=False)

params_with_ln = sum(p.numel() for p in model_with_ln.parameters())
params_without_ln = sum(p.numel() for p in model_without_ln.parameters())

print(f"\nModel with LayerNorm: {params_with_ln:,} parameters")
print(f"Model without LayerNorm: {params_without_ln:,} parameters")
print(f"Difference: {params_with_ln - params_without_ln:,} parameters")

# Load saved model and check parameter count
timestamp = "20251027_141324"
a3c_model_path = f"runs/a3c_{timestamp}/models/global_final.pth"
ind_model_path = f"runs/individual_{timestamp}/models/individual_worker_0_final.pth"

print("\n" + "=" * 60)
print("Saved Model Analysis")
print("=" * 60)

# Check A3C model
checkpoint = torch.load(a3c_model_path, map_location="cpu", weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

saved_params = sum(p.numel() for p in state_dict.values())
print(f"\nA3C Global model ({timestamp}): {saved_params:,} parameters")

# Check for LayerNorm layers
has_ln_feature = any('ln_feature' in k for k in state_dict.keys())
has_ln_rnn = any('ln_rnn' in k for k in state_dict.keys())

print(f"  Has ln_feature layers: {has_ln_feature}")
print(f"  Has ln_rnn layers: {has_ln_rnn}")

# Check Individual model
checkpoint_ind = torch.load(ind_model_path, map_location="cpu", weights_only=False)
if isinstance(checkpoint_ind, dict) and 'model_state_dict' in checkpoint_ind:
    state_dict_ind = checkpoint_ind['model_state_dict']
else:
    state_dict_ind = checkpoint_ind

saved_params_ind = sum(p.numel() for p in state_dict_ind.values())
print(f"\nIndividual Worker 0 ({timestamp}): {saved_params_ind:,} parameters")

has_ln_feature_ind = any('ln_feature' in k for k in state_dict_ind.keys())
has_ln_rnn_ind = any('ln_rnn' in k for k in state_dict_ind.keys())

print(f"  Has ln_feature layers: {has_ln_feature_ind}")
print(f"  Has ln_rnn layers: {has_ln_rnn_ind}")

# Conclusion
print("\n" + "=" * 60)
print("Conclusion")
print("=" * 60)

if saved_params == params_with_ln:
    print("\n✓ Models were trained WITH Layer Normalization")
elif saved_params == params_without_ln:
    print("\n✗ Models were trained WITHOUT Layer Normalization")
else:
    print(f"\n? Unknown configuration (saved: {saved_params}, expected with LN: {params_with_ln}, without LN: {params_without_ln})")

print("=" * 60)
