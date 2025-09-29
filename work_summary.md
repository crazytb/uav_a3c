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

## Next Steps

1. Implement the cloud resource sharing fixes
2. Retrain model with corrected environment
3. Verify balanced action selection in new training
4. Compare performance metrics before/after fix

---
*Last Updated: 2025-09-29*