"""
Simple test to verify Layer Normalization implementation.
"""

print("=" * 60)
print("Layer Normalization Implementation Test")
print("=" * 60)

# Test import
try:
    from drl_framework.networks import RecurrentActorCritic
    from drl_framework.params import hidden_dim, use_layer_norm
    import torch
    print("\n✓ Imports successful")
except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    exit(1)

# Test model creation
try:
    state_dim = 50  # Example
    action_dim = 3

    print(f"\nCreating models:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  use_layer_norm (from params): {use_layer_norm}")

    model_with_ln = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=True)
    model_without_ln = RecurrentActorCritic(state_dim, action_dim, hidden_dim, use_layer_norm=False)

    print(f"\n✓ Models created successfully")
    print(f"  With LN params: {sum(p.numel() for p in model_with_ln.parameters()):,}")
    print(f"  Without LN params: {sum(p.numel() for p in model_without_ln.parameters()):,}")

except Exception as e:
    print(f"\n✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check LayerNorm components
print(f"\nChecking LayerNorm components:")
has_ln_feature = hasattr(model_with_ln, 'ln_feature')
has_ln_rnn = hasattr(model_with_ln, 'ln_rnn')
is_ln_feature = isinstance(model_with_ln.ln_feature, torch.nn.LayerNorm)
is_ln_rnn = isinstance(model_with_ln.ln_rnn, torch.nn.LayerNorm)

print(f"  ✓ Has ln_feature: {has_ln_feature}")
print(f"  ✓ ln_feature is LayerNorm: {is_ln_feature}")
print(f"  ✓ Has ln_rnn: {has_ln_rnn}")
print(f"  ✓ ln_rnn is LayerNorm: {is_ln_rnn}")

# Test forward pass
try:
    print(f"\nTesting forward pass:")
    batch_size = 4
    x = torch.randn(batch_size, state_dim)
    hx = model_with_ln.init_hidden(batch_size)

    logits, value, next_hx = model_with_ln.step(x, hx)

    print(f"  ✓ Forward pass successful")
    print(f"    Logits shape: {logits.shape} (expected: [{batch_size}, {action_dim}])")
    print(f"    Value shape: {value.shape} (expected: [{batch_size}, 1])")
    print(f"    Hidden shape: {next_hx.shape}")

except Exception as e:
    print(f"\n✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test that LN normalizes activations
print(f"\nTesting LayerNorm normalization:")
with torch.no_grad():
    # Get activations before and after LN
    z = model_with_ln.feature(x)
    print(f"  Before ln_feature: mean={z.mean():.4f}, std={z.std():.4f}")

    z_ln = model_with_ln.ln_feature(z)
    print(f"  After ln_feature:  mean={z_ln.mean():.4f}, std={z_ln.std():.4f}")

    # Check per-sample normalization (each sample should have mean~0, std~1)
    sample_means = z_ln.mean(dim=1)
    sample_stds = z_ln.std(dim=1)
    print(f"  Per-sample means: {sample_means.abs().mean():.4f} (should be close to 0)")
    print(f"  Per-sample stds: {sample_stds.mean():.4f} (should be close to 1)")

print(f"\n{'=' * 60}")
print("Summary:")
print(f"{'=' * 60}")

if is_ln_feature and is_ln_rnn:
    print("✓ Layer Normalization successfully implemented!")
    print("  - Feature layer has LayerNorm")
    print("  - RNN output has LayerNorm")
    print("  - Forward pass works correctly")
    print(f"\nCurrent setting: use_layer_norm = {use_layer_norm}")
    print("\nTo use LayerNorm in training:")
    print("  1. Ensure params.py has use_layer_norm = True")
    print("  2. Run main_train.py as usual")
else:
    print("✗ Layer Normalization not properly configured")

print(f"{'=' * 60}")
