#!/usr/bin/env python3
"""Test script to verify feedforward network configuration"""

import torch
from drl_framework.networks import ActorCritic, RecurrentActorCritic
from drl_framework.params import use_recurrent, use_layer_norm, hidden_dim, device
from drl_framework.custom_env import make_env
from drl_framework.params import ENV_PARAMS
from drl_framework.utils import flatten_dict_values

# Test environment dimensions
temp_env_fn = make_env(**ENV_PARAMS)
temp_env = temp_env_fn()
sample_obs = temp_env.reset()[0]
state_dim = len(flatten_dict_values(sample_obs))
action_dim = temp_env.action_space.n

print("=" * 60)
print("Network Configuration Test")
print("=" * 60)
print(f"use_recurrent: {use_recurrent}")
print(f"use_layer_norm: {use_layer_norm}")
print(f"hidden_dim: {hidden_dim}")
print(f"device: {device}")
print(f"state_dim: {state_dim}")
print(f"action_dim: {action_dim}")
print()

# Instantiate model based on configuration
if use_recurrent:
    model = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=use_layer_norm).to(device)
    print(f"✓ Created RecurrentActorCritic with LayerNorm={use_layer_norm}")
else:
    model = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
    print(f"✓ Created ActorCritic (feedforward)")

print()
print("Model architecture:")
print(model)
print()

# Test forward pass
print("Testing forward pass...")
test_input = torch.randn(1, state_dim).to(device)
hx = model.init_hidden(batch_size=1, device=device)

# Test step method
logits, value, hx_new = model.step(test_input, hx)
print(f"✓ step() output shapes: logits={logits.shape}, value={value.shape}, hx={hx_new.shape}")

# Test rollout method
test_seq = torch.randn(1, 5, state_dim).to(device)  # batch=1, time=5
done_seq = torch.zeros(1, 5).bool().to(device)
logits_seq, values_seq, hx_final = model.rollout(test_seq, hx, done_seq)
print(f"✓ rollout() output shapes: logits_seq={logits_seq.shape}, values_seq={values_seq.shape}")

print()
print("=" * 60)
print("✓ All tests passed! Configuration is correct.")
print("=" * 60)
