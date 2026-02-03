# Session Summary — Observable State Transitions (2026-02-04)

Target file: `manuscript/paper.tex`, Section "Observable State Transitions" (around line 305)

---

## 1. Per-Element Transition Functions 작성

`drl_framework/custom_env.py` 코드를 근거로 각 subspace 요소별 transition을 LaTeX로 작성했습니다.

### 작성된 내용 (paper.tex 약 line 321~367)

| 변수 | 핵심 수식 | 코드 근거 |
|---|---|---|
| $c_l'$ | $c_l - c_q\cdot\mathbf{1}[a{=}0]\cdot\phi_l + \sum_k \tilde{c}_{m,k}\cdot\mathbf{1}[\tilde{t}_{m,k}=1]$ | `available_computation_units` 소비 + MEC completion 회수 |
| $e_r'$ | $e_r - 1$ | `remain_epochs` 단순 decrement |
| $i_l'$ | $\phi_l$ if $a=0$, else $i_l$ (retain) | `local_success`는 LOCAL action 시에만 갱신 |
| $i_o'$ | $\phi_o$ if $a=1$, else $i_o$ (retain) | `offload_success`는 OFFLOAD action 시에만 갱신 |
| $\mathbf{c}_m', \mathbf{t}_m'$ | 3-phase: enqueue → completion → decrement | `fill_first_zero` → `proc_times==1` 체크 → `clip(-1, 0)` |

### Key Definitions Introduced

- **$\phi_l$** (local feasibility): $\mathbf{1}[c_l \geq c_q]\cdot\mathbf{1}[\exists k: c_{m,k}=0]\cdot\mathbf{1}[c_q > 0]$
- **$\phi_o$** (offload feasibility): $\mathbf{1}[\exists k: c_{c,k}=0]\cdot\mathbf{1}[c_q > 0]\cdot\mathbf{1}[Q=1]\cdot\mathbf{1}[C_c^{\mathrm{avail}} \geq c_q]$
- **$\kappa$**: $\min\{k: c_{m,k}=0\}$ — MEC queue의 첫 빈 슬롯
- **$\tilde{c}_{m,k}, \tilde{t}_{m,k}$**: enqueue 후 completion 전의 intermediate MEC state

### MEC 3-Phase Update 구조

코드 실행 순서와 수식 대응:
1. **Enqueue**: LOCAL + $\phi_l$이면 슬롯 $\kappa$에 $(c_q, t_q)$ 배치 → $\tilde{c}_{m,k}, \tilde{t}_{m,k}$
2. **Completion**: $\tilde{t}_{m,k}=1$인 task 완료, comp_units를 $c_l$에 회수
3. **Decrement**: $t_{m,k}' = \max(\tilde{t}_{m,k}-1, 0)$, 활성 task를 앞으로 compact

---

## 2. Decomposition Equation 수정 (약 line 309)

실제 dependency를 정확히 반영하여 conditioning set을 수정했습니다.

| 항 | 기존 | 수정 후 | 변경 이유 |
|---|---|---|---|
| $P[c_l']$ | $c_l, c_q, a$ | $c_l, c_q, t_q, \mathbf{c}_m, \mathbf{t}_m, a$ | completion 회수항이 $\mathbf{t}_m, t_q$에 의존 |
| $P[i_l']$ | $c_l, c_q, \mathbf{c}_m, a$ | $i_l, c_l, c_q, \mathbf{c}_m, a$ | $a \neq 0$이면 $i_l'=i_l$ (retain) |
| $P[i_o']$ | $c_q, a$ | $i_o, c_q, z_t, a$ | retain + $\phi_o$가 $Q, \mathbf{c}_c \in z_t$에 의존 |
| MEC | $\prod_k P[c_{m,k}'\|\cdot] \times \prod_k P[t_{m,k}'\|\cdot]$ | $P[\mathbf{c}_m', \mathbf{t}_m' \| \mathbf{c}_m, \mathbf{t}_m, c_l, c_q, t_q, a]$ | $\kappa$가 slot 간 결합 → per-slot 독립 불성립 |
| $c_q', t_q'$ | (없음) | $\times\, P[c_q'] \times P[t_q']$ 추가 | $o'$의 구성 요소인데 빠져 있었음 |

---

## 3. 변수명 매핑 (논문 ↔ custom_env.py)

### Observable — $\mathcal{O}$

**$S_{local}$**

| 논문 | custom_env.py | get_obs() key | 정규화 |
|---|---|---|---|
| $c_l$ | `available_computation_units` | `"available_computation_units"` | `/ max_available_computation_units` |
| $e_r$ | `remain_epochs` | `"remain_epochs"` | `/ max_remain_epochs` |
| $c_q$ | `queue_comp_units` | `"queue_comp_units"` | `/ max_comp_units` |
| $t_q$ | `queue_proc_times` | `"queue_proc_times"` | `/ max_proc_times` (=50) |
| $i_l$ | `local_success` | `"local_success"` | 없음 (binary) |
| $i_o$ | `offload_success` | `"offload_success"` | 없음 (binary) |

**$S_{context}$**

| 논문 | custom_env.py | 정규화 |
|---|---|---|
| $V$ | `agent_velocities` | `(v - 30) / 70` |
| $C$ | `max_comp_units` | `/ 120.0` (`_COMP_MAX`) |

**$S_{mec}$**

| 논문 | custom_env.py | 정규화 |
|---|---|---|
| $\mathbf{c}_m$ | `mec_comp_units` | `/ max_comp_units` |
| $\mathbf{t}_m$ | `mec_proc_times` | `/ max_proc_times` (=50) |

### Hidden — $\mathcal{Z}$ (get_obs()에 포함되지 않음)

| 논문 | custom_env.py |
|---|---|
| $\mathbf{c}_c$ | `cloud_comp_units` |
| $\mathbf{t}_c$ | `cloud_proc_times` |
| $Q$ | `channel_quality` |

### Transition 수식에만 등장하는 변수

| 논문 | custom_env.py | 설명 |
|---|---|---|
| $C_c^{\mathrm{avail}}$ | `available_computation_units_for_cloud` | 클라우드 잔여 용량 (network_state 없을 때) |
| $\phi_l$ | `case_action` (LOCAL) | 조건식, attribute 아님 |
| $\phi_o$ | `local_conditions and can_consume_cloud` | 조건식, attribute 아님 |
| $\kappa$ | `fill_first_zero` 내부 로직 | 첫 빈 슬롯 인덱스 |

---

## 4. $c_q, t_q$의 독립 확률분포 표현

**질문:** $c_q, t_q$는 매 스텝마다 state/action과 무관하게 랜덤 생성되는데, factorization에서 어떻게 표현하나?

**핵심 구분 — dual role:**

| 역할 | 시점 | 수식에서의 표현 |
|---|---|---|
| 현재 스텝의 관측값 $c_q^{(t)}$ | action phase에서 사용됨 | RHS 조건부: `P[c_l' \| ..., c_q, ...]` |
| 다음 스텝의 독립 random $c_q^{(t+1)}$ | step 끝에서 새로 생성 | 비조건부: `P[c_q']` (or bar 없음) |

**결론:** `|` 기호가 없는 것 자체가 독립임을 표현. 구체적 분포식은 Exogenous 섹션에 별도로 기술됨.

---

## 5. LaTeX Overfull 수정

intermediate MEC state 수식을 1행 (`\qquad` 구분)에서 2행 `equation + aligned`로 변환하여 overfull 오류 해결.

---

## 미해결 / 추후 확인 항목

1. **구조적 불일치 (미수정):** line 296의 상위 decomposition이 `P(o'|o,z,a) · P(z'|z)` (observable/hidden축)이지만, 실제 섹션 내용과 하단 종합식(line 389)은 endogenous/exogenous축. 사용자가 순차적으로 작성 중이므로 현재 미수정.
2. **$C$ 정규화 상수 불일치:** paper.tex 약 line 227에서 computation capacity를 "0 to 200 units"로 작성되었나, 코드의 정규화 상수는 `_COMP_MAX = 120.0`. `200`은 `queue_comp_units`의 sampling range와 혼동된 것으로 보음. 확인 필요.
3. **Cloud queue transition 미기술:** OFFLOAD action이 cloud queue를 수정하는 것($\mathbf{c}_c', \mathbf{t}_c'$ transition)은 현재 논문의 어떤 섹션에서도 명시적으로 작성되지 않음.
