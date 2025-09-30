# Work Summary - UAV A3C Project

## Issues Identified and Solutions

### 1. Model Extreme Bias Issue (2025-09-29)

**Problem**: Model shows extreme bias toward OFFLOAD action (99.99% probability), ignoring other actions regardless of input.

**Analysis Results**:
- Action 1 (OFFLOAD): ~21 logits → 99.99% probability
- Action 0 (LOCAL), 2 (DISCARD): ~-4 logits → near 0% probability
- Model outputs identical results for different random inputs

### 2. Root Cause: Broken Cloud Resource Sharing Implementation

**Critical Discovery**: The intended cloud resource sharing mechanism between workers is completely broken.

#### 2.1 Intended vs Actual Implementation
- **Intended**: 5 workers compete for shared cloud resources (10,000 units total)
- **Actual**: Each worker has independent cloud resources (1,000 units each = 5,000 total)

#### 2.2 Specific Issues Found

**Issue 1: make_env() Function Disabled**
```python
# custom_env.py:346-351 - COMMENTED OUT
def make_env(**kwargs):
    # network_state = kwargs.pop('network_state', None)  ← DISABLED!
    # def _make():
        # return CustomEnv(**kwargs, network_state=network_state)  ← DISABLED!
    # return _make
    return partial(CustomEnv, **kwargs)  ← Only this active
```

**Issue 2: Cloud Resource Competition Missing**
```python
# custom_env.py OFFLOAD action - Missing shared resource check
case_action = ((self.available_computation_units_for_cloud >= self.queue_comp_units) and
               (self.cloud_comp_units[self.cloud_comp_units == 0].size > 0) and
               (self.queue_comp_units > 0) and
               (self.channel_quality == 1))
# NO shared cloud resource verification!
```

**Issue 3: Each Worker Uses Independent Resources**
```python
# custom_env.py:182 - Each worker gets 1000 cloud units independently
self.available_computation_units_for_cloud = self.max_available_computation_units_for_cloud
```

#### 2.3 Environment Reward Imbalance
- **LOCAL completion**: `reward += done_comp`
- **OFFLOAD completion**: `reward += (done_comp - BETA*consumed_time)` where BETA=0.5 (minimal penalty)
- **Failure penalties**: LOCAL/OFFLOAD = -10, DISCARD = -5
- **Energy cost**: Set to 0.0 (no difference between LOCAL/OFFLOAD)

### 3. File Collection Error Fix

**Problem**: Notebook trying to access `/mnt/data/*_actions.csv` (non-existent path)

**Solution**: Updated to use current directory pattern:
```python
# Before
files = sorted(f for f in glob.glob("/mnt/data/*_actions.csv") if pattern.search(f))

# After
files = sorted(f for f in glob.glob("*_actions.csv") if pattern.search(f))
pattern = re.compile(r"(a3c_global|individual_w\d+)_env(\d+)_actions\.csv$")
```

### 4. Tensor Shape Mismatch Error Fix

**Problem**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x49 and 48x128)`

**Root Cause**: Test code created 49-dimensional tensor but model expects 48-dimensional input

**Solution**: Use dynamic dimension detection:
```python
# Get expected dimension from model
expected_input_dim = model.feature[0].in_features
obs_tensor = torch.randn(1, expected_input_dim, dtype=torch.float32, device=device)
```

### 5. Recommended Fixes for Cloud Resource Sharing

#### Fix 1: Enable make_env() function
```python
def make_env(**kwargs):
    network_state = kwargs.pop('network_state', None)
    worker_id = kwargs.pop('worker_id', None)
    def _make():
        return CustomEnv(**kwargs, network_state=network_state, worker_id=worker_id)
    return _make
```

#### Fix 2: Add shared resource check in OFFLOAD action
```python
elif action == OFFLOAD:   # Offload
    # Check shared cloud resource first
    can_consume_cloud = (self.network_state is None or
                        self.network_state.consume_cloud_resource(self.worker_id, self.queue_comp_units))

    case_action = (can_consume_cloud and
                   (self.cloud_comp_units[self.cloud_comp_units == 0].size > 0) and
                   (self.queue_comp_units > 0) and
                   (self.channel_quality == 1))
```

#### Fix 3: Rebalance reward structure
- Increase BETA value for meaningful time costs
- Adjust cloud resource capacity (currently too generous)
- Rebalance failure penalties between actions

### 6. Impact Assessment

**Current State**:
- Workers operate with ~5000 total cloud resources (practically unlimited)
- No competition for resources
- OFFLOAD always succeeds → Model learns extreme bias

**Expected After Fix**:
- True resource competition between 5 workers
- OFFLOAD failures when resources depleted
- Balanced action selection based on resource availability
- Significantly improved model training dynamics

## Implementation Status (2025-09-30)

### ✅ Cloud Resource Sharing Implementation Complete

**Changes Made**:

1. **Fixed make_env() function** (custom_env.py:346-351)
   - Re-enabled network_state and worker_id passing
   - Now properly passes shared resources to each environment instance

2. **Added shared cloud resource check** (custom_env.py:248-275)
   - OFFLOAD action now checks NetworkState.consume_cloud_resource()
   - Local conditions checked first to avoid unnecessary resource consumption
   - Maintains backward compatibility when network_state is None (for testing)

3. **Updated reset() with clarifying comments** (custom_env.py:176-187)
   - Clarified that available_computation_units_for_cloud is only for non-shared mode
   - NetworkState.available_cloud_capacity manages shared resources during training

4. **Verification Complete**
   - Created test_simple_cloud_sharing.py
   - Verified NetworkState correctly manages shared resources
   - Confirmed resource competition and depletion works as expected

**Test Results**:
```
Initial cloud resources: 1000.0
Worker 0-4: 200 each → All succeed (1000 → 0)
Worker 0: 200 more → FAILED (resource depleted)
After releasing 400: Worker 1: 300 → SUCCESS
```

### Code Changes Summary

**custom_env.py**:
- Line 346-351: make_env() now properly handles network_state
- Line 249-275: OFFLOAD action checks shared cloud resources
- Line 185-187: Added clarifying comments for reset()

**Test Files Created**:
- test_simple_cloud_sharing.py: Basic NetworkState verification
- test_cloud_resource_sharing.py: Full environment integration test (needs refinement)

## Critical Bug Fix (2025-09-30 - Evening)

### 🚨 **Root Cause of Training Collapse Identified**

**Problem**: Cloud resources were NOT being reset between episodes!

**Impact**:
```
Episode 1: Cloud 1000 → 100 (consumed)
Episode 2: Started with 100 (NO RESET!) → depleted
Episode 3: Started with 0 → all OFFLOADs fail forever
Result: Model learned "OFFLOAD always fails" → DISCARD bias
```

**Evidence**:
- Training reward graph showed catastrophic collapse after 50 episodes
- Action distribution heavily biased toward DISCARD (>90%)
- Cloud resource testing revealed resources never recovered between episodes

**Fix Applied** (network_state.py:106-107):
```python
def reset_for_new_episode(self):
    """새 에피소드를 위한 네트워크 상태 리셋"""
    with self.lock:
        self.total_offloading_load.value *= 0.1
        # ⭐ ADDED: Reset cloud resources completely
        self.available_cloud_capacity.value = self.max_cloud_capacity
        self.current_episode.value += 1
```

**Verification**:
- Created test_episode_reset.py
- Confirmed resources reset to max_cloud_capacity after each episode
- Expected outcome: Model can now learn proper offloading strategies

## Reward Scaling Fix (2025-09-30 - Night)

### 🔧 **High Loss Value Problem Solved**

**Problem**: Loss values were extremely high (50k-600k), causing training instability.

**Root Cause**:
```python
# Episode rewards: 4000-7000
# Value predictions: 0-500 (slowly learning)
# Value Loss = MSE(500, 5000)^2 = 20,250,000!
# Total Loss = policy_loss + 0.5 × 20M ≈ 10M
```

**Analysis**:
- Reward magnitude (thousands) vs Value network scale mismatch
- Large advantage values (5000 - 500 = 4500) caused unstable policy gradients
- Value network learned too slowly due to massive target values

**Fix Applied**:
1. **custom_env.py:349-351** - Added reward scaling before return
2. **params.py:39** - Set `REWARD_SCALE = 0.01` (divide by 100)

```python
# Before
return next_obs, self.reward, done, False, {}
# Reward range: 0-7000

# After
scaled_reward = self.reward * 0.01
return next_obs, scaled_reward, done, False, {}
# Reward range: 0-70
```

**Expected Impact**:
- Loss: 100,000 → 1,000 (100x reduction)
- Value predictions: 0-70 range (reasonable)
- Advantages: -70 to +70 (stable policy gradients)
- Faster convergence of value network
- Better numerical stability

## Next Steps

1. ✅ ~~Implement the cloud resource sharing fixes~~ **COMPLETE**
2. ✅ ~~Fix critical cloud resource reset bug~~ **COMPLETE**
3. ✅ ~~Fix high loss values with reward scaling~~ **COMPLETE**
4. **Retrain model with all fixes** (NEXT PRIORITY)
5. Verify loss values are in reasonable range (100-10k)
6. Verify balanced action selection in new training
7. Compare performance metrics before/after all fixes
8. Consider Knowledge Distillation framework (cloud teacher → edge student model)

## Future Directions: Knowledge Distillation Framework

Based on analysis with GPT-4, potential enhancement direction:

**Concept**: Cloud (complex teacher model) + Knowledge Distillation + UAV (lightweight student model)

**Advantages over Individual models**:
- Single global model → Easy deployment via distillation
- Fast warm start for new UAVs (no cold start)
- Communication efficiency (1 student vs N individual models)
- Hierarchical adaptation: Cloud (generalization) + Edge (personalization)

**Trade-offs**:
- Implementation complexity increases 2-3x
- Central server dependency
- Requires distillation pipeline maintenance

This could be a significant contribution for the paper: "Hierarchical Cloud-Edge RL for UAV MEC with Resource Competition"

---
*Last Updated: 2025-09-30*