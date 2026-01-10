# State Transition Function for ATLAS

**Reference**: Based on DeepAAQM IEEE IoT Journal paper format
**Purpose**: Document state transition dynamics for ATLAS framework
**Last Updated**: 2026-01-11

---

## State Representation

We denote two arbitrary states, the current state $s$ and the next state $s'$ in $\mathcal{S}$ as:

$$
s = \{C_l, E_r, C_q, T_q, I_l, I_o, \mathbf{C}_m^K, \mathbf{T}_m^K, V, C\}
$$

and

$$
s' = \{C_l', E_r', C_q', T_q', I_l', I_o', \mathbf{C}_m^{K'}, \mathbf{T}_m^{K'}, V', C'\}
$$

where $\mathbf{C}_m, \mathbf{T}_m \in \mathbb{R}^K$ are $K$-dimensional vectors representing MEC server queue states, and all other components are scalars.

**Cloud Resource Management**: Cloud computing capacity is shared among all workers in the distributed A3C framework. Unlike local and MEC resources that are part of the observable state, cloud capacity acts as an exogenous constraint managed through atomic operations (via `NetworkState` class). From each worker's perspective, cloud availability influences offloading action feasibility ($I_o$) but is not explicitly included in the state representation to maintain the single-agent POMDP formulation.

**Hidden State**: Channel quality $Q$ is not directly observable but influences state transitions (particularly $I_o$) and is implicitly inferred by the recurrent neural network through observation sequences.

---

## State Transition Function

The state transition function is the probability of reaching state $s'$ from state $s$ when the agent takes action $a \in \{0, 1, 2\}$ (LOCAL, OFFLOAD, DISCARD).

The complete transition function can be decomposed into **endogenous** (action-dependent) and **exogenous** (action-independent) components:

$$
\Pr[s' | s, a] = \Pr[s_{\text{endo}}' | s, a] \times \Pr[s_{\text{exo}}' | s]
$$

---

## Endogenous State Transitions (Action-Dependent)

The endogenous components are directly influenced by the agent's actions and evolve according to controllable system dynamics:

$$
\begin{split}
\Pr[s_{\text{endo}}' | s, a] &= \Pr[C_l' | C_l, C_q, a] \\
&\times \Pr[E_r' | E_r] \\
&\times \Pr[I_l' | C_l, C_q, \mathbf{C}_m, a] \\
&\times \Pr[I_o' | C_q, a] \\
&\times \prod_{k=1}^{K} \Pr[C_{m,k}' | C_{m,k}, T_{m,k}, a] \\
&\times \prod_{k=1}^{K} \Pr[T_{m,k}' | T_{m,k}]
\end{split}
$$

### 1. Local Computation Units ($C_l$)

The local computation capacity transitions based on task processing completion and new task allocation:

$$
C_l' = \begin{cases}
C_l - C_q, & \text{if } a = 0 \text{ (LOCAL)} \land C_l \geq C_q \land \exists k: C_{m,k} = 0 \\
C_l + \sum_{k: T_{m,k}=1} C_{m,k}, & \text{if local tasks complete} \\
C_l, & \text{otherwise}
\end{cases}
$$

**Explanation**:
- When LOCAL action is taken and feasible, computation units are consumed
- When local processing completes (processing time reaches 1), units are released
- Otherwise, capacity remains unchanged

---

### 2. Remaining Epochs ($E_r$)

Time progression is deterministic:

$$
E_r' = E_r - 1
$$

**Explanation**: Each time step decrements the remaining epoch counter until episode termination ($E_r = 0$).

---

### 3. Success Indicators ($I_l, I_o$)

#### Local Success Indicator:

$$
I_l' = \begin{cases}
1, & \text{if } a = 0 \land C_l \geq C_q \land \exists k: C_{m,k} = 0 \land C_q > 0 \\
0, & \text{otherwise}
\end{cases}
$$

**Conditions for success**:
- LOCAL action selected
- Sufficient local computation units
- Available slot in MEC queue
- Non-empty task queue

#### Offload Success Indicator:

$$
I_o' = \begin{cases}
1, & \text{if } a = 1 \land C_c^{\text{global}} \geq C_q \land C_q > 0 \\
0, & \text{otherwise}
\end{cases}
$$

**Conditions for success**:
- OFFLOAD action selected
- Sufficient cloud capacity (queried atomically from shared resource)
- Non-empty task queue
- Note: Channel quality $Q$ (hidden) also affects success but is not explicitly shown in worker's state

**Cloud Capacity Check**: The condition $C_c^{\text{global}} \geq C_q$ is evaluated atomically via `NetworkState.consume_cloud_resource()`, which returns true if resources are available and atomically reserves them.

---

### 4. MEC Server State ($\mathbf{C}_m^K, \mathbf{T}_m^K$)

#### MEC Computation Units:

$$
C_{m,k}' = \begin{cases}
C_q, & \text{if } k = \text{first\_zero}(\mathbf{C}_m) \land a = 0 \land I_l' = 1 \\
0, & \text{if } T_{m,k} = 1 \text{ (task completes)} \\
C_{m,k}, & \text{otherwise}
\end{cases}
$$

#### MEC Processing Times:

$$
T_{m,k}' = \begin{cases}
T_q, & \text{if } k = \text{first\_zero}(\mathbf{T}_m) \land a = 0 \land I_l' = 1 \\
0, & \text{if } T_{m,k} = 1 \text{ (task completes)} \\
\max(0, T_{m,k} - 1), & \text{otherwise (decrement)}
\end{cases}
$$

**Explanation**:
- When LOCAL action succeeds, task is added to first available MEC slot
- Tasks complete when processing time reaches 1
- All active tasks decrement their processing time each step

---

## Exogenous State Transitions (Action-Independent)

The exogenous components evolve independently of the agent's actions and model external task arrivals and environmental conditions:

$$
\Pr[s_{\text{exo}}' | s] = \Pr[C_q'] \times \Pr[T_q'] \times \Pr[V'] \times \Pr[C']
$$

### 5. Queue State ($C_q, T_q$)

New tasks arrive at each time step with random computational requirements and processing times:

$$
C_q' \sim \text{Uniform}(1, C_{\max})
$$

$$
T_q' \sim \text{Uniform}(1, T_{\max})
$$

**Explanation**:
- Queue state is sampled from uniform distributions
- $C_{\max} = 200$ (maximum computation units per task)
- $T_{\max} = 50$ (maximum processing time)
- This stochastic arrival process models unpredictable workload in real-world UAV applications
- Task arrivals are i.i.d. (independent and identically distributed)

---

### 6. Environmental Context ($V, C$)

#### Velocity Context:

$$
\Pr[V'] = \delta(V' - V)
$$

where $V$ is the configured velocity for the current training/evaluation session, normalized as:

$$
V = \frac{v - v_{\min}}{v_{\max} - v_{\min}}, \quad v_{\min} = 30 \text{ km/h}, \quad v_{\max} = 100 \text{ km/h}
$$

**Explanation**: Velocity remains constant within an episode (deterministic transition).

#### Computation Context:

$$
\Pr[C'] = \delta(C' - C)
$$

where $C = C_{\max} / 120$ represents the normalized maximum computation capacity.

**Explanation**: Computation capacity context remains constant within an episode.

---

## Hidden State Transitions (Not Observable)

### Channel Quality ($Q$) - Hidden

Channel quality follows a Gilbert-Elliott two-state Markov model based on IEEE 802.11bd V2X:

$$
\Pr[Q' | Q] = \begin{cases}
\text{TRAN}_{01}, & \text{if } Q' = 1, Q = 0 \\
1 - \text{TRAN}_{01}, & \text{if } Q' = 0, Q = 0 \\
\text{TRAN}_{10}, & \text{if } Q' = 0, Q = 1 \\
1 - \text{TRAN}_{10}, & \text{if } Q' = 1, Q = 1
\end{cases}
$$

where transition probabilities depend on Doppler frequency:

$$
f_d = \frac{v}{3600 \cdot c} f_0, \quad f_0 = 5.9 \times 10^9 \text{ Hz (IEEE 802.11bd)}
$$

$$
\text{TRAN}_{01} = \frac{f_d \cdot T_p \cdot \sqrt{2\pi \rho}}{e^\rho - 1}, \quad \text{TRAN}_{10} = f_d \cdot T_p \cdot \sqrt{2\pi \rho}
$$

where:
- $T_p$ = packet time (duration per task)
- $\rho$ = SNR threshold normalized to average SNR (baseline: 15 dB)
- $c$ = speed of light (300,000 km/s)

**Explanation**: Higher velocity increases Doppler frequency, leading to more frequent channel state changes.

---

### Cloud Server State ($\mathbf{C}_c^K, \mathbf{T}_c^K$) - Shared Resource (Not in State)

Cloud server capacity is managed as a **shared exogenous constraint** rather than a state component:

$$
C_c^{\text{global}}' = C_c^{\text{global}} - \sum_{w=1}^{W} \mathbb{1}[a_w = 1 \land I_{o,w}' = 1] \cdot C_{q,w} + \sum_{\text{completed}} C_{c,k}
$$

where $W$ is the number of parallel workers competing for cloud resources.

**Why Not in State**:
- Cloud capacity evolves based on **all workers' actions** simultaneously
- Cannot be expressed as $\Pr[\mathbf{C}_c' | \mathbf{C}_c, a_i]$ for individual worker $i$
- Would require multi-agent POMDP formulation
- Instead, treated as **atomic constraint** queried during action execution

**Implementation**: `NetworkState` class provides:
- `consume_cloud_resource(worker_id, amount)`: Atomic check-and-reserve
- `release_cloud_resource(amount)`: Return resources on task completion
- Thread-safe operations with locks

---

## Action-Dependent Transitions Summary

### Action 0: LOCAL Processing

**Affected states** (Endogenous):
- $C_l' \leftarrow C_l - C_q$ (if feasible)
- $C_{m,k}' \leftarrow C_q$ (first zero slot)
- $T_{m,k}' \leftarrow T_q$ (first zero slot)
- $I_l' \leftarrow 1$ (if successful)

**Conditions**: $C_l \geq C_q \land \exists k: C_{m,k} = 0 \land C_q > 0$

---

### Action 1: OFFLOAD to Cloud

**Affected states** (Endogenous):
- $I_o' \leftarrow 1$ (if successful)
- Cloud resources consumed (external, not in state)

**Conditions**:
- Atomic check: $C_c^{\text{global}} \geq C_q$ via `NetworkState`
- $C_q > 0$
- Hidden: $Q = 1$ (good channel)

---

### Action 2: DISCARD

**Affected states**: None (no-op action)

**Rationale**: When both local and offload are infeasible (e.g., insufficient resources, poor channel, saturated queues), discarding minimizes immediate resource waste.

---

## Complete Transition Function Expression

Combining endogenous and exogenous components:

$$
\begin{split}
\Pr[s' | s, a] &= \underbrace{\Pr[C_l' | C_l, C_q, a]}_{\text{Endogenous}} \\
&\times \Pr[E_r' | E_r] \\
&\times \Pr[I_l' | C_l, C_q, \mathbf{C}_m, a] \\
&\times \Pr[I_o' | C_q, a] \\
&\times \prod_{k=1}^{K} \Pr[C_{m,k}' | C_{m,k}, T_{m,k}, a] \\
&\times \prod_{k=1}^{K} \Pr[T_{m,k}' | T_{m,k}] \\
&\times \underbrace{\Pr[C_q'] \times \Pr[T_q']}_{\text{Exogenous}} \\
&\times \Pr[V'] \times \Pr[C']
\end{split}
$$

---

## Markov Property Verification

The state transition satisfies the Markov property:

$$
\Pr[s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0] = \Pr[s_{t+1} | s_t, a_t]
$$

**Justification**:
- All relevant information for decision-making is captured in current state $s_t$
- RNN hidden state $h_t$ (maintained separately) captures temporal dependencies for hidden states
- Channel quality transitions depend only on current quality (Gilbert-Elliott model)
- Task arrivals are i.i.d. samples
- Cloud capacity is queried atomically at decision time

---

## Implementation Notes

### Normalization

All continuous state components are normalized to $[0, 1]$:

$$
C_l^{\text{norm}} = \frac{C_l}{C_{\max}}, \quad T_m^{\text{norm}} = \frac{T_m}{T_{\max}}
$$

### Deterministic vs. Stochastic Transitions

**Deterministic (Endogenous)**:
- Epoch progression: $E_r' = E_r - 1$
- Processing time decrement: $T_{m,k}' = \max(0, T_{m,k} - 1)$
- Resource allocation (conditional on action)

**Deterministic (Exogenous)**:
- Context variables: $V', C'$ (constant within episode)

**Stochastic (Exogenous)**:
- Queue arrivals: $C_q', T_q' \sim \text{Uniform}$

**Stochastic (Hidden)**:
- Channel quality: $Q' \sim \text{Gilbert-Elliott}$

---

## Code Reference

**Implementation**: `drl_framework/custom_env.py`
- Line 283-347: `step()` function with state transitions
- Line 318-320: Environment step execution
- Line 332-347: Processing phase (task completion, endogenous)
- Line 139-176: `change_channel_quality()` (hidden state transition)

**Network State Management**: `drl_framework/network_state.py`
- `NetworkState` class: Shared cloud resource management
- Atomic operations with multiprocessing locks

---

## Summary

The state transition function models:

1. **Endogenous Dynamics** (Action-Dependent):
   - Resource allocation (computation units)
   - Queue management (MEC server state)
   - Success indicators (action outcomes)
   - Temporal progression (epoch countdown)

2. **Exogenous Dynamics** (Action-Independent):
   - Task arrivals (stochastic workload)
   - Environmental context (constant within episode)

3. **Hidden States**:
   - Channel quality (Gilbert-Elliott model)
   - Cloud capacity (managed externally as atomic constraint)

4. **Multi-Agent Coordination**:
   - Shared cloud resources with atomic access
   - Single-agent POMDP formulation maintained

The combination of controllable endogenous dynamics, uncontrollable exogenous stochasticity, and hidden state inference creates a partially observable, continuous-state MDP suitable for deep reinforcement learning with recurrent architectures.

**Key Design Choice**: Cloud capacity as an **exogenous constraint** rather than a state component allows maintaining single-agent POMDP formulation while acknowledging multi-agent resource competition in the distributed A3C framework.
