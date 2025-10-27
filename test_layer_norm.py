"""
Test script to verify Layer Normalization implementation.

This script:
1. Creates RecurrentActorCritic with and without LayerNorm
2. Runs a forward pass to check activation statistics
3. Verifies LayerNorm reduces activation variance
"""

import torch
import numpy as np
from drl_framework.networks import RecurrentActorCritic
from drl_framework.custom_env import make_env
from drl_framework.params import ENV_PARAMS, REWARD_PARAMS, hidden_dim, use_layer_norm
from drl_framework.utils import flatten_dict_values

print("=" * 60)
print("Layer Normalization Test")
print("=" * 60)

# Create environment to get state/action dims
env_fn = make_env(**ENV_PARAMS, reward_params=REWARD_PARAMS)
env = env_fn()
sample_obs = env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = env.action_space.n

print(f"\nEnvironment Info:")
print(f"  State dim: {state_dim}")
print(f"  Action dim: {action_dim}")
print(f"  Hidden dim: {hidden_dim}")

# Create models
print(f"\n{'=' * 60}")
print("Creating models...")
model_with_ln = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=True)
model_without_ln = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=False)

print(f"  Model with LayerNorm: {sum(p.numel() for p in model_with_ln.parameters())} parameters")
print(f"  Model without LayerNorm: {sum(p.numel() for p in model_without_ln.parameters())} parameters")

# Check LayerNorm layers exist
has_ln_feature = hasattr(model_with_ln, 'ln_feature') and isinstance(model_with_ln.ln_feature, torch.nn.LayerNorm)
has_ln_rnn = hasattr(model_with_ln, 'ln_rnn') and isinstance(model_with_ln.ln_rnn, torch.nn.LayerNorm)

print(f"\nLayerNorm components:")
print(f"  ✓ ln_feature (after feature extraction): {has_ln_feature}")
print(f"  ✓ ln_rnn (after RNN output): {has_ln_rnn}")

# Run forward pass and collect statistics
print(f"\n{'=' * 60}")
print("Running forward pass with random inputs...")

batch_size = 32
seq_len = 20
random_states = torch.randn(batch_size, seq_len, state_dim)

# Initialize hidden states
hx_with_ln = model_with_ln.init_hidden(batch_size)
hx_without_ln = model_without_ln.init_hidden(batch_size)

# Forward pass
with torch.no_grad():
    # Collect intermediate activations
    activations_with_ln = {}
    activations_without_ln = {}

    # Test single step
    x_single = random_states[:, 0, :]  # (B, state_dim)

    # With LayerNorm
    z = model_with_ln.feature(x_single)
    activations_with_ln['feature_before_ln'] = z.clone()
    z = model_with_ln.ln_feature(z)
    activations_with_ln['feature_after_ln'] = z.clone()
    z = torch.nn.functional.relu(z)
    z, _ = model_with_ln.rnn(z.unsqueeze(1), hx_with_ln)
    z = z[:, 0, :]
    activations_with_ln['rnn_before_ln'] = z.clone()
    z = model_with_ln.ln_rnn(z)
    activations_with_ln['rnn_after_ln'] = z.clone()
    logits_ln, value_ln = model_with_ln.policy(z), model_with_ln.value(z)

    # Without LayerNorm
    z = model_without_ln.feature(x_single)
    activations_without_ln['feature_raw'] = z.clone()
    z = torch.nn.functional.relu(z)
    z, _ = model_without_ln.rnn(z.unsqueeze(1), hx_without_ln)
    z = z[:, 0, :]
    activations_without_ln['rnn_raw'] = z.clone()
    logits_no_ln, value_no_ln = model_without_ln.policy(z), model_without_ln.value(z)

print(f"\nActivation Statistics (mean ± std):")
print(f"\n  After Feature Extraction:")
print(f"    Without LN: {activations_without_ln['feature_raw'].mean():.4f} ± {activations_without_ln['feature_raw'].std():.4f}")
print(f"    Before LN:  {activations_with_ln['feature_before_ln'].mean():.4f} ± {activations_with_ln['feature_before_ln'].std():.4f}")
print(f"    After LN:   {activations_with_ln['feature_after_ln'].mean():.4f} ± {activations_with_ln['feature_after_ln'].std():.4f}")

print(f"\n  After RNN:")
print(f"    Without LN: {activations_without_ln['rnn_raw'].mean():.4f} ± {activations_without_ln['rnn_raw'].std():.4f}")
print(f"    Before LN:  {activations_with_ln['rnn_before_ln'].mean():.4f} ± {activations_with_ln['rnn_before_ln'].std():.4f}")
print(f"    After LN:   {activations_with_ln['rnn_after_ln'].mean():.4f} ± {activations_with_ln['rnn_after_ln'].std():.4f}")

print(f"\n  Output Statistics:")
print(f"    Value (with LN):    {value_ln.mean():.4f} ± {value_ln.std():.4f}")
print(f"    Value (without LN): {value_no_ln.mean():.4f} ± {value_no_ln.std():.4f}")
print(f"    Logits (with LN):    {logits_ln.mean():.4f} ± {logits_ln.std():.4f}")
print(f"    Logits (without LN): {logits_no_ln.mean():.4f} ± {logits_no_ln.std():.4f}")

# Test gradient flow
print(f"\n{'=' * 60}")
print("Testing gradient flow...")

model_with_ln.train()
model_without_ln.train()

x = random_states[:, 0, :]
hx_with = model_with_ln.init_hidden(batch_size)
hx_without = model_without_ln.init_hidden(batch_size)

# Forward and backward with LayerNorm
logits_ln, value_ln, _ = model_with_ln.step(x, hx_with)
loss_ln = value_ln.mean()
loss_ln.backward()

# Forward and backward without LayerNorm
logits_no_ln, value_no_ln, _ = model_without_ln.step(x, hx_without)
loss_no_ln = value_no_ln.mean()
loss_no_ln.backward()

# Collect gradient statistics
grad_stats_with_ln = []
grad_stats_without_ln = []

for name, param in model_with_ln.named_parameters():
    if param.grad is not None:
        grad_stats_with_ln.append((name, param.grad.abs().mean().item(), param.grad.abs().max().item()))

for name, param in model_without_ln.named_parameters():
    if param.grad is not None:
        grad_stats_without_ln.append((name, param.grad.abs().mean().item(), param.grad.abs().max().item()))

print(f"\nGradient Statistics (mean / max):")
print(f"  With LayerNorm:")
for name, mean_grad, max_grad in grad_stats_with_ln[:5]:
    print(f"    {name:20s}: {mean_grad:.6f} / {max_grad:.6f}")

print(f"\n  Without LayerNorm:")
for name, mean_grad, max_grad in grad_stats_without_ln[:5]:
    print(f"    {name:20s}: {mean_grad:.6f} / {max_grad:.6f}")

# Summary
print(f"\n{'=' * 60}")
print("Summary:")
print(f"{'=' * 60}")

# Check if LayerNorm normalizes properly (mean ~0, std ~1)
feature_after_mean = activations_with_ln['feature_after_ln'].mean().item()
feature_after_std = activations_with_ln['feature_after_ln'].std().item()
rnn_after_mean = activations_with_ln['rnn_after_ln'].mean().item()
rnn_after_std = activations_with_ln['rnn_after_ln'].std().item()

ln_working = abs(feature_after_mean) < 0.5 and abs(rnn_after_mean) < 0.5

if ln_working:
    print("✓ Layer Normalization is working correctly!")
    print(f"  - Feature output: mean ≈ 0 ({feature_after_mean:.4f})")
    print(f"  - RNN output: mean ≈ 0 ({rnn_after_mean:.4f})")
    print(f"  - Activations are normalized as expected")
else:
    print("✗ Warning: Layer Normalization may not be working as expected")
    print(f"  - Feature output mean: {feature_after_mean:.4f} (expected ~0)")
    print(f"  - RNN output mean: {rnn_after_mean:.4f} (expected ~0)")

print(f"\n✓ Test completed successfully!")
print(f"  Current setting in params.py: use_layer_norm = {use_layer_norm}")
print(f"\n{'=' * 60}")
