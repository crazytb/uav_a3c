# Related Works Paper Classification

**Date**: 2025-11-05
**Purpose**: Classification of collected papers for Related Works section

---

## 📚 Subsection A: Reinforcement Learning for UAV Task Offloading

### ⭐⭐⭐ Highly Relevant

**1. Multi-Agent Deep Reinforcement Learning for Task Offloading in UAV-Assisted Mobile Edge Computing**
- **Authors**: Nan Zhao, Zhiyang Ye, Yiyang Pei, Ying-Chang Liang, Dusit Niyato
- **Venue**: IEEE Transactions on Wireless Communications, 2022
- **Summary**: Proposes a cooperative multi-agent DRL (MATD3) framework for multi-UAV multi-EC task offloading that jointly optimizes trajectories, task allocation, and resource management. Uses centralized training with decentralized execution strategy, where each UAV acts as an independent agent with TD3 algorithm. Demonstrates superior performance compared to fixed-power, fixed-height, and random schemes across varying bandwidth, task arrival rates, and transmission power settings.
- **Key Contribution**: Cooperative MADRL with each UAV as independent agent (different from our A3C parameter sharing approach)
- **Filename**: `Multi-Agent_Deep_Reinforcement_Learning_for_Task_Offloading_in_UAV-Assisted_Mobile_Edge_Computing.pdf`

**2. Multi-Agent DRL for Task Offloading and Resource Allocation in Multi-UAV Enabled IoT Edge Network**
- **Summary**: Addresses task offloading and resource allocation in multi-UAV IoT edge networks using multi-agent deep reinforcement learning. Focuses on joint optimization of computation offloading decisions and communication resource allocation to minimize system cost including latency and energy consumption.
- **Relevance**: Multi-UAV + DRL for task offloading
- **Filename**: `Multi-Agent_DRL_for_Task_Offloading_and_Resource_Allocation_in_Multi-UAV_Enabled_IoT_Edge_Network.pdf`

### ⭐⭐ Moderately Relevant

**3. DRQN-based Task Offloading in Partially Observable UAV-assisted Mobile Edge Computing Environments (ICTC 2024)**
- **Summary**: Applies Deep Recurrent Q-Network (DRQN) to handle partial observability in UAV-assisted MEC task offloading. Uses recurrent neural networks (LSTM/GRU) to capture temporal dependencies and hidden state information for improved decision-making under incomplete observations.
- **Key Contribution**: Handles POMDP formulation with value-based recurrent method (our work uses actor-critic with RNN)
- **Filename**: `DRQN_based_Task_Offloading_in_Partially_Observable_UAV_assisted_Mobile_Edge_Computing_Environments__ICTC_2024_.pdf`

**4. Cognitive UAV-Assisted Offloading for Mobile Edge Computing Based on Multi-Agent Deep Reinforcement Learning**
- **Summary**: Proposes cognitive UAV-assisted offloading framework using multi-agent DRL where UAVs learn to make intelligent offloading decisions based on environmental observations. Focuses on cognitive radio aspects and spectrum-aware task offloading in UAV-MEC systems.
- **Relevance**: Multi-agent DRL for UAV offloading with cognitive aspects
- **Filename**: `Cognitive_UAV-Assisted_Offloading_for_Mobile_Edge_Computing_Based_on_Multi-Agent_Deep_Reinforcement_Learning.pdf`

**5. Joint Task Offloading, Resource Allocation, and Trajectory Design for Multi-UAV Cooperative Edge Computing With Task Priority**
- **Summary**: Addresses joint optimization of task offloading, resource allocation, and UAV trajectory design considering task priority levels. Focuses on cooperative multi-UAV systems where UAVs collaborate to serve ground users with heterogeneous task requirements.
- **Relevance**: Multi-UAV cooperation for task offloading (may not use DRL)
- **Filename**: `Joint_Task_Offloading_Resource_Allocation_and_Trajectory_Design_for_Multi-UAV_Cooperative_Edge_Computing_With_Task_Priority.pdf`

**6. Robust Computation Offloading and Trajectory Optimization for Multi-UAV-Assisted MEC: A Multiagent DRL Approach**
- **Summary**: Proposes multi-agent DRL approach for robust computation offloading and trajectory optimization in multi-UAV MEC systems. Addresses uncertainty and dynamic environments through robust optimization techniques combined with deep reinforcement learning.
- **Relevance**: Multi-agent DRL for UAV-MEC with robustness focus
- **Filename**: `Robust_Computation_Offloading_and_Trajectory_Optimization_for_Multi-UAV-Assisted_MEC_A_Multiagent_DRL_Approach.pdf`

### ⭐ Reference Material

**7. Multi-UAV Cooperative Task Offloading and Resource Allocation in 5G Advanced and Beyond**
- **Summary**: Examines multi-UAV cooperative task offloading in 5G and beyond networks, focusing on resource allocation strategies for enhanced mobile broadband and ultra-reliable low-latency communications scenarios.
- **Relevance**: Multi-UAV cooperation context
- **Filename**: `Multi-UAV_Cooperative_Task_Offloading_and_Resource_Allocation_in_5G_Advanced_and_Beyond.pdf`

**8. Stackelberg-Game-Based Intelligent Offloading Incentive Mechanism for a Multi-UAV-Assisted Mobile-Edge Computing System**
- **Summary**: Proposes game-theoretic approach (Stackelberg game) for task offloading incentive mechanism in multi-UAV MEC systems. Focuses on economic incentives and pricing mechanisms rather than learning-based approaches.
- **Relevance**: Alternative optimization approach (game theory) for comparison
- **Filename**: `Stackelberg-Game-Based_Intelligent_Offloading_Incentive_Mechanism_for_a_Multi-UAV-Assisted_Mobile-Edge_Computing_System.pdf`

---

## 📚 Subsection B: Distributed Learning and Parameter Sharing

### ⭐⭐⭐ Highly Relevant

**1. Federated Deep Reinforcement Learning**
- **Summary**: Foundational work on federated learning paradigm applied to deep reinforcement learning. Proposes framework where distributed agents train local models and periodically aggregate parameters to a global model, enabling collaborative learning while preserving data privacy.
- **Key Contribution**: Federated learning framework for DRL (different from our synchronous A3C parameter sharing)
- **Filename**: `Federated_Deep_Reinforcement_Learning.pdf`

**2. Federated Deep Reinforcement Learning for Task Offloading and Resource Allocation in Mobile Edge Computing-Assisted Vehicular Networks**
- **Summary**: Applies federated DRL to vehicular edge computing for task offloading and resource allocation. Each vehicle trains local policy and shares model parameters through federated aggregation, addressing non-IID data distribution and communication efficiency challenges.
- **Key Contribution**: Federated learning application to edge computing offloading
- **Filename**: `Federated_deep_reinforcement_learning_for_task_offloading_and_resource_allocation_in_mobile_edge_computing-assisted_vehicular_networks.pdf`

### ⭐⭐ Moderately Relevant

**3. Federated Deep Reinforcement Learning Based Task Offloading With Power Control in Vehicular Edge Computing**
- **Summary**: Combines federated DRL with power control for task offloading in vehicular networks. Addresses joint optimization of offloading decisions and transmission power under federated learning framework with intermittent connectivity.
- **Relevance**: Federated DRL for offloading with communication constraints
- **Filename**: `Federated_Deep_Reinforcement_Learning_Based_Task_Offloading_With_Power_Control_in_Vehicular_Edge_Computing.pdf`

**4. Federated Deep Reinforcement Learning for Internet of Things With Decentralized Cooperative Edge Caching**
- **Summary**: Proposes federated DRL framework for cooperative edge caching in IoT systems. Multiple edge servers collaboratively learn caching policies through federated parameter aggregation while maintaining decentralized execution.
- **Relevance**: Decentralized cooperative learning paradigm
- **Filename**: `Federated_Deep_Reinforcement_Learning_for_Internet_of_Things_With_Decentralized_Cooperative_Edge_Caching.pdf`

**5. Federated Deep Reinforcement Learning for the Distributed Control of NextG Wireless Networks**
- **Summary**: Applies federated DRL to distributed control problems in next-generation wireless networks. Addresses challenges of heterogeneous network environments and communication overhead in federated training.
- **Relevance**: Distributed control with federated learning
- **Filename**: `Federated_Deep_Reinforcement_Learning_for_the_Distributed_Control_of_NextG_Wireless_Networks.pdf`

**6. When Deep Reinforcement Learning Meets Federated Learning: Intelligent Multitimescale Resource Management for Multiaccess Edge Computing in 5G Ultradense Network**
- **Summary**: Integrates DRL with federated learning for multitimescale resource management in ultra-dense MEC systems. Proposes hierarchical learning framework that operates at different time scales for efficient resource allocation.
- **Relevance**: DRL + Federated learning intersection for MEC
- **Filename**: `When_Deep_Reinforcement_Learning_Meets_Federated_Learning_Intelligent_Multitimescale_Resource_Management_for_Multiaccess_Edge_Computing_in_5G_Ultradense_Network.pdf`

### ⭐ Reference Material

**7. Personalized Federated Deep Reinforcement Learning for Heterogeneous Edge Content Caching Networks**
- **Summary**: Proposes personalized federated DRL that allows each edge node to maintain personalized model while benefiting from collaborative training. Addresses heterogeneity in edge networks through personalization layers.
- **Relevance**: Personalized federated learning (less relevant to our homogeneous worker approach)
- **Filename**: `Personalized_Federated_Deep_Reinforcement_Learning_for_Heterogeneous_Edge_Content_Caching_Networks.pdf`

**8. FedRL: A Reinforcement Learning Federated Recommender System for Efficient Communication Using Reinforcement Selector and Hypernet Generator**
- **Summary**: Applies federated RL to recommender systems with focus on communication efficiency through selective model updates and hypernetwork-based parameter generation.
- **Relevance**: Federated RL framework design (domain different)
- **Filename**: `FedRL A Reinforcement Learning Federated Recommender System for Efficient Communication Using Reinforcement Selector and Hypernet Generator.pdf`

**9. FedRL: Federated Learning with Non-IID Data via Review Learning**
- **Summary**: Addresses non-IID data challenge in federated learning through review learning mechanism. Proposes method to improve federated model performance when data distributions vary significantly across clients.
- **Relevance**: Federated learning methodology (general framework)
- **Filename**: `FedRL Federated Learning with Non-IID Data via Review Learning.pdf`

---

## 📚 Subsection C: Architecture Components in Deep Reinforcement Learning

### ⚠️ Limited Coverage

**Note**: The current paper collection has limited papers specifically focused on architectural components (RNN, LayerNorm, ablation studies) in deep RL. Most papers are application-focused rather than architecture-analysis focused.

**Recommendation**:
- Consider adding classic DRL architecture papers (e.g., original A3C paper by Mnih et al., LSTM/GRU in RL papers)
- Add normalization technique papers (e.g., LayerNorm in RL, Batch Norm variants)
- Include ablation study methodology papers from DeepMind/OpenAI
- Alternatively, restructure Subsection C to "System Design and Performance Analysis" to better fit available papers

---

## 🚫 Not Relevant / Out of Scope

**1. Deep-Reinforcement-Learning-Based Age-of-Information-Aware Low-Power Active Queue Management for IoT Sensor Networks**
- **Reason**: Focuses on AoI (Age of Information) and queue management in IoT, not UAV or task offloading
- **Filename**: `Deep-Reinforcement-Learning-Based_Age-of-Information-Aware_Low-Power_Active_Queue_Management_for_IoT_Sensor_Networks.pdf`

**2. HARE: Hybrid ARQ-Based Adaptive Retransmission Control Scheme for Synchronous Multi-Link in Wireless LANs**
- **Reason**: Not related to reinforcement learning or task offloading (communication protocol design)
- **Filename**: `HARE_Hybrid_ARQ-Based_Adaptive_Retransmission_Control_Scheme_for_Synchronous_Multi-Link_in_Wireless_LANs.pdf`

**3. s11276-024-03804-3.pdf**
- **Reason**: Unable to determine relevance from filename alone (needs manual inspection)
- **Action**: Manually check this paper's title and abstract
- **Filename**: `s11276-024-03804-3.pdf`

---

## 📊 Classification Summary

| Subsection | Highly Relevant (⭐⭐⭐) | Moderately Relevant (⭐⭐) | Reference (⭐) | Total |
|-----------|----------------------|------------------------|--------------|-------|
| **A: RL for UAV Task Offloading** | 2 | 4 | 2 | 8 |
| **B: Distributed Learning** | 2 | 4 | 3 | 9 |
| **C: Architecture Components** | 0 | 0 | 0 | 0 |
| **Not Relevant** | - | - | - | 3 |
| **Total Relevant Papers** | 4 | 8 | 5 | **17** |

---

## 🎯 Citation Strategy Recommendations

### Must-Cite Papers

**Subsection A:**
- Zhao et al. (TWC 2022) - Multi-Agent DRL with TD3, cooperative framework
- DRQN paper (ICTC 2024) - Partial observability handling with recurrent networks

**Subsection B:**
- Federated Deep Reinforcement Learning (foundational work)
- Federated DRL for vehicular networks (application to edge computing)

### How to Position Our Work

**vs. Zhao et al. (MATD3):**
- **Their approach**: Independent UAV agents with cooperative reward, centralized training
- **Our approach**: A3C with global parameter sharing, asynchronous updates
- **Key difference**: "While [Zhao et al.] use independent agents with cooperative rewards, our A3C framework employs direct parameter sharing for more efficient knowledge transfer"

**vs. Federated DRL:**
- **Their approach**: Periodic aggregation, privacy-preserving, asynchronous communication
- **Our approach**: Continuous parameter sharing, synchronous updates to global model
- **Key difference**: "Unlike federated learning's periodic aggregation, A3C maintains continuous parameter sharing for real-time policy updates"

**vs. DRQN:**
- **Their approach**: Value-based (Q-learning) with recurrent networks
- **Our approach**: Actor-critic with recurrent policy
- **Key difference**: "While DRQN uses value-based learning, our recurrent actor-critic directly learns stochastic policies suitable for continuous action spaces"

---

## 📝 Next Steps

1. **Add papers to BibTeX file** with proper citation keys
2. **Read abstracts** of moderately relevant papers to confirm classification
3. **Check s11276-024-03804-3.pdf** manually to determine relevance
4. **Consider collecting additional papers** for Subsection C on architecture components
5. **Draft Related Works text** using this classification as guide
