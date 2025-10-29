"""
Visualize Neural Network Architecture for UAV Task Offloading

Creates a comprehensive diagram showing:
1. Input state structure (48-dim)
2. RecurrentActorCritic architecture
3. Information flow with and without Layer Normalization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig = plt.figure(figsize=(20, 14))

# ===== Subplot 1: Input State Composition (48-dim) =====
ax1 = plt.subplot(2, 3, 1)
ax1.axis('off')
ax1.set_title('Input State Composition (48 dimensions)', fontsize=14, fontweight='bold', pad=20)

# Define state components
state_components = [
    ('Queue Info (MEC)', 40, '#3498db'),
    ('Context Features', 2, '#e74c3c'),
    ('Task Flags', 2, '#f39c12'),
    ('Scalars', 4, '#2ecc71')
]

y_start = 0.9
bar_width = 0.6
total_dims = 48

for name, dims, color in state_components:
    height = (dims / total_dims) * 0.7
    rect = FancyBboxPatch((0.1, y_start - height), 0.8, height,
                           boxstyle="round,pad=0.01",
                           edgecolor=color, facecolor=color, alpha=0.3,
                           linewidth=2)
    ax1.add_patch(rect)

    # Add text
    ax1.text(0.5, y_start - height/2, f'{name}\n{dims} dims',
             ha='center', va='center', fontsize=11, fontweight='bold')

    y_start -= height

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Add detailed breakdown
breakdown_y = 0.1
ax1.text(0.05, breakdown_y, 'Breakdown:', fontsize=10, fontweight='bold')
breakdown_y -= 0.05
details = [
    '• MEC comp_units: 20',
    '• MEC proc_times: 20',
    '• ctx_vel, ctx_comp: 2',
    '• local/offload success: 2',
    '• available_units, remain_epochs,',
    '  queue_comp, queue_proc: 4'
]
for detail in details:
    ax1.text(0.05, breakdown_y, detail, fontsize=8)
    breakdown_y -= 0.04

# ===== Subplot 2: Network Architecture Without LN =====
ax2 = plt.subplot(2, 3, 2)
ax2.axis('off')
ax2.set_title('RecurrentActorCritic WITHOUT Layer Normalization',
              fontsize=14, fontweight='bold', pad=20)

# Define layers
layers_no_ln = [
    ('Input', 'State\n48-dim', 0.9, '#ecf0f1', 0.08),
    ('Feature', 'Linear\n48→128', 0.75, '#3498db', 0.08),
    ('Activation', 'ReLU', 0.68, '#95a5a6', 0.04),
    ('RNN', 'GRU\n128→128', 0.57, '#9b59b6', 0.08),
    ('Split', 'Split', 0.47, '#95a5a6', 0.02),
    ('Actor', 'Linear\n128→3', 0.30, '#2ecc71', 0.08),
    ('Critic', 'Linear\n128→1', 0.30, '#e74c3c', 0.08),
    ('Output1', 'Policy Logits\n3-dim', 0.15, '#27ae60', 0.08),
    ('Output2', 'Value\n1-dim', 0.15, '#c0392b', 0.08)
]

# Draw layers
for i, (layer_id, label, y, color, height) in enumerate(layers_no_ln):
    if layer_id == 'Split':
        # Special handling for split
        ax2.plot([0.5, 0.3], [y, y-0.05], 'k-', linewidth=2, alpha=0.5)
        ax2.plot([0.5, 0.7], [y, y-0.05], 'k-', linewidth=2, alpha=0.5)
        ax2.text(0.5, y, label, ha='center', va='center', fontsize=9,
                style='italic', color='gray')
    elif 'Output' in layer_id:
        # Outputs (side by side)
        x_pos = 0.3 if layer_id == 'Output1' else 0.7
        rect = FancyBboxPatch((x_pos - 0.15, y - height/2), 0.3, height,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, alpha=0.7,
                             linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x_pos, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold')
    else:
        # Regular layers
        rect = FancyBboxPatch((0.2, y - height/2), 0.6, height,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, alpha=0.7,
                             linewidth=2)
        ax2.add_patch(rect)
        ax2.text(0.5, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Draw arrows
        if i < len(layers_no_ln) - 1:
            next_layer = layers_no_ln[i + 1]
            if 'Split' not in layer_id and 'Split' not in next_layer[0]:
                if 'Output' not in next_layer[0]:
                    arrow = FancyArrowPatch((0.5, y - height/2),
                                          (0.5, next_layer[2] + next_layer[4]/2),
                                          arrowstyle='->', mutation_scale=20,
                                          linewidth=2, color='black', alpha=0.5)
                    ax2.add_patch(arrow)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Add parameter count
ax2.text(0.5, 0.05, 'Total: 105,860 parameters',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ===== Subplot 3: Network Architecture With LN =====
ax3 = plt.subplot(2, 3, 3)
ax3.axis('off')
ax3.set_title('RecurrentActorCritic WITH Layer Normalization',
              fontsize=14, fontweight='bold', pad=20)

# Define layers with LN
layers_with_ln = [
    ('Input', 'State\n48-dim', 0.9, '#ecf0f1', 0.06),
    ('Feature', 'Linear\n48→128', 0.78, '#3498db', 0.06),
    ('LN1', 'LayerNorm', 0.71, '#f1c40f', 0.03),
    ('Activation', 'ReLU', 0.66, '#95a5a6', 0.03),
    ('RNN', 'GRU\n128→128', 0.57, '#9b59b6', 0.06),
    ('LN2', 'LayerNorm', 0.50, '#f1c40f', 0.03),
    ('Split', 'Split', 0.45, '#95a5a6', 0.02),
    ('Actor', 'Linear\n128→3', 0.30, '#2ecc71', 0.06),
    ('Critic', 'Linear\n128→1', 0.30, '#e74c3c', 0.06),
    ('Output1', 'Policy Logits\n3-dim', 0.18, '#27ae60', 0.06),
    ('Output2', 'Value\n1-dim', 0.18, '#c0392b', 0.06)
]

# Draw layers
for i, (layer_id, label, y, color, height) in enumerate(layers_with_ln):
    if layer_id == 'Split':
        ax3.plot([0.5, 0.3], [y, y-0.05], 'k-', linewidth=2, alpha=0.5)
        ax3.plot([0.5, 0.7], [y, y-0.05], 'k-', linewidth=2, alpha=0.5)
        ax3.text(0.5, y, label, ha='center', va='center', fontsize=9,
                style='italic', color='gray')
    elif 'Output' in layer_id:
        x_pos = 0.3 if layer_id == 'Output1' else 0.7
        rect = FancyBboxPatch((x_pos - 0.15, y - height/2), 0.3, height,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, alpha=0.7,
                             linewidth=2)
        ax3.add_patch(rect)
        ax3.text(x_pos, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold')
    else:
        rect = FancyBboxPatch((0.2, y - height/2), 0.6, height,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, alpha=0.7,
                             linewidth=2 if 'LN' not in layer_id else 3)
        ax3.add_patch(rect)
        ax3.text(0.5, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Draw arrows
        if i < len(layers_with_ln) - 1:
            next_layer = layers_with_ln[i + 1]
            if 'Split' not in layer_id and 'Split' not in next_layer[0]:
                if 'Output' not in next_layer[0]:
                    arrow = FancyArrowPatch((0.5, y - height/2),
                                          (0.5, next_layer[2] + next_layer[4]/2),
                                          arrowstyle='->', mutation_scale=20,
                                          linewidth=2, color='black', alpha=0.5)
                    ax3.add_patch(arrow)

# Highlight LayerNorm
ax3.text(0.92, 0.71, '⭐', fontsize=20)
ax3.text(0.92, 0.50, '⭐', fontsize=20)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Add parameter count
ax3.text(0.5, 0.10, 'Total: 106,372 parameters',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.text(0.5, 0.05, '+512 params (LayerNorm)',
         ha='center', fontsize=9, style='italic')

# ===== Subplot 4: Action Space =====
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
ax4.set_title('Action Space & Output', fontsize=14, fontweight='bold', pad=20)

# Action space
actions = [
    ('0: LOCAL', 'Process task\nlocally', '#2ecc71'),
    ('1: OFFLOAD', 'Offload to\nMEC server', '#3498db'),
    ('2: DISCARD', 'Discard\ntask', '#e74c3c')
]

y_pos = 0.8
for action_id, desc, color in actions:
    # Action box
    rect = FancyBboxPatch((0.1, y_pos - 0.15), 0.35, 0.12,
                         boxstyle="round,pad=0.01",
                         edgecolor=color, facecolor=color, alpha=0.3,
                         linewidth=2)
    ax4.add_patch(rect)
    ax4.text(0.275, y_pos - 0.09, action_id, ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Description
    ax4.text(0.55, y_pos - 0.09, desc, ha='left', va='center',
            fontsize=10)

    y_pos -= 0.2

# Policy output
ax4.text(0.1, 0.3, 'Policy Output:', fontsize=11, fontweight='bold')
ax4.text(0.1, 0.24, 'Logits → Softmax → π(a|s)', fontsize=10)
ax4.text(0.1, 0.19, 'Sample action: a ~ Categorical(π)', fontsize=10)

# Value output
ax4.text(0.1, 0.1, 'Value Output:', fontsize=11, fontweight='bold')
ax4.text(0.1, 0.04, 'V(s): Expected return from state s', fontsize=10)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# ===== Subplot 5: Training Flow (A3C) =====
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
ax5.set_title('A3C Training Flow', fontsize=14, fontweight='bold', pad=20)

# Training steps
steps = [
    'Environment → State(48)',
    'RecurrentActorCritic',
    'Sample Action ~ π(a|s)',
    'Execute & Collect (s,a,r,s\')',
    'Compute Advantage',
    'Calculate Losses',
    'Backprop (5 workers)',
    'Aggregate Gradients',
    'Update Global Model'
]

y = 0.95
for i, step in enumerate(steps):
    color = '#3498db' if i % 2 == 0 else '#2ecc71'
    rect = FancyBboxPatch((0.1, y - 0.07), 0.8, 0.06,
                         boxstyle="round,pad=0.01",
                         edgecolor=color, facecolor=color, alpha=0.3,
                         linewidth=1.5)
    ax5.add_patch(rect)
    ax5.text(0.5, y - 0.04, step, ha='center', va='center',
            fontsize=10, fontweight='bold' if i in [1, 7] else 'normal')

    if i < len(steps) - 1:
        arrow = FancyArrowPatch((0.5, y - 0.07), (0.5, y - 0.1),
                              arrowstyle='->', mutation_scale=15,
                              linewidth=1.5, color='black', alpha=0.5)
        ax5.add_patch(arrow)

    y -= 0.1

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)

# ===== Subplot 6: Layer Normalization Effect =====
ax6 = plt.subplot(2, 3, 6)
ax6.set_title('Layer Normalization Impact', fontsize=14, fontweight='bold', pad=20)

# Data from experiments
categories = ['Value Loss\nReduction', 'Training\nReward', 'Generalization\n(Test)']
a3c_improvements = [61.5, 9.6, 251]
ind_improvements = [91.1, 1.8, -13.9]

x = np.arange(len(categories))
width = 0.35

bars1 = ax6.bar(x - width/2, a3c_improvements, width,
                label='A3C', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x + width/2, ind_improvements, width,
                label='Individual', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax6.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(categories, fontsize=10)
ax6.legend(fontsize=11, loc='upper left')
ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

# Add title annotation
ax6.text(0.5, 0.95, 'Why A3C benefits MORE from LN than Individual',
         transform=ax6.transAxes, ha='center', fontsize=10,
         style='italic', color='darkred',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('network_architecture_diagram.png', dpi=200, bbox_inches='tight')
print(f"[Saved] network_architecture_diagram.png")

# ===== Create a separate detailed architecture diagram =====
fig2, ax = plt.subplots(figsize=(16, 12))
ax.axis('off')
ax.set_title('Detailed RecurrentActorCritic Architecture with Data Flow',
             fontsize=16, fontweight='bold', pad=30)

# Define vertical positions for each layer
layer_info = [
    # (y_position, layer_name, shape, params, color)
    (0.95, 'Input State', '(B, 48)', '-', '#ecf0f1'),
    (0.88, 'Linear Layer', '(B, 48) → (B, 128)', '6,272', '#3498db'),
    (0.81, 'LayerNorm (optional)', '(B, 128)', '256', '#f1c40f'),
    (0.76, 'ReLU Activation', '(B, 128)', '-', '#95a5a6'),
    (0.69, 'GRU (Recurrent)', '(B, 1, 128) + hx', '98,688', '#9b59b6'),
    (0.62, 'LayerNorm (optional)', '(B, 128)', '256', '#f1c40f'),
    (0.53, 'Actor Head', '(B, 128) → (B, 3)', '387', '#2ecc71'),
    (0.53, 'Critic Head', '(B, 128) → (B, 1)', '129', '#e74c3c'),
    (0.44, 'Policy Logits', '(B, 3)', '-', '#27ae60'),
    (0.44, 'State Value', '(B, 1)', '-', '#c0392b'),
]

# Draw layers
for y, name, shape, params, color in layer_info:
    if name in ['Actor Head', 'Policy Logits']:
        x_center = 0.25
    elif name in ['Critic Head', 'State Value']:
        x_center = 0.75
    else:
        x_center = 0.5

    # Layer box
    box_width = 0.35 if x_center == 0.5 else 0.3
    rect = FancyBboxPatch((x_center - box_width/2, y - 0.03), box_width, 0.05,
                         boxstyle="round,pad=0.01",
                         edgecolor='black', facecolor=color, alpha=0.7,
                         linewidth=2.5 if 'LayerNorm' in name else 2)
    ax.add_patch(rect)

    # Layer name
    ax.text(x_center, y - 0.005, name, ha='center', va='center',
           fontsize=12, fontweight='bold')

    # Shape annotation (left)
    ax.text(x_center - box_width/2 - 0.02, y - 0.005, shape,
           ha='right', va='center', fontsize=9, style='italic')

    # Parameters (right)
    if params != '-':
        ax.text(x_center + box_width/2 + 0.02, y - 0.005, f'{params} params',
               ha='left', va='center', fontsize=9, color='blue')

# Draw arrows
arrow_connections = [
    (0.5, 0.92, 0.5, 0.85),  # Input → Linear
    (0.5, 0.85, 0.5, 0.78),  # Linear → LN
    (0.5, 0.78, 0.5, 0.73),  # LN → ReLU
    (0.5, 0.73, 0.5, 0.66),  # ReLU → GRU
    (0.5, 0.64, 0.5, 0.59),  # GRU → LN
    (0.5, 0.59, 0.25, 0.50), # Split to Actor
    (0.5, 0.59, 0.75, 0.50), # Split to Critic
    (0.25, 0.50, 0.25, 0.41), # Actor → Output
    (0.75, 0.50, 0.75, 0.41), # Critic → Output
]

for x1, y1, x2, y2 in arrow_connections:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=20,
                          linewidth=2, color='black', alpha=0.6)
    ax.add_patch(arrow)

# Add annotations
ax.text(0.5, 0.35, 'Total Parameters:', ha='center', fontsize=13, fontweight='bold')
ax.text(0.5, 0.31, 'With LN: 106,372  |  Without LN: 105,860',
       ha='center', fontsize=11)

# Add legend for layer types
legend_y = 0.20
ax.text(0.1, legend_y, 'Layer Types:', fontsize=11, fontweight='bold')
legend_items = [
    ('Input/Output', '#ecf0f1'),
    ('Fully Connected', '#3498db'),
    ('Normalization', '#f1c40f'),
    ('Activation', '#95a5a6'),
    ('Recurrent', '#9b59b6'),
    ('Actor', '#2ecc71'),
    ('Critic', '#e74c3c'),
]
for i, (label, color) in enumerate(legend_items):
    y = legend_y - 0.04 - i * 0.03
    rect = mpatches.Rectangle((0.1, y), 0.02, 0.02, facecolor=color,
                              edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    ax.text(0.13, y + 0.01, label, fontsize=9, va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('detailed_architecture_flow.png', dpi=200, bbox_inches='tight')
print(f"[Saved] detailed_architecture_flow.png")

print("\n" + "="*80)
print("Network Architecture Visualization Complete!")
print("="*80)
print("\nGenerated files:")
print("  1. network_architecture_diagram.png - 6-panel overview")
print("  2. detailed_architecture_flow.png - Detailed layer-by-layer diagram")
print("\nKey insights:")
print("  • Input: 48 heterogeneous dimensions")
print("  • Hidden: 128-dim GRU with optional LayerNorm")
print("  • Output: 3 actions (LOCAL/OFFLOAD/DISCARD) + value")
print("  • LayerNorm adds only 512 parameters (0.5% increase)")
print("  • A3C benefits +251% generalization with LN")
print("  • Individual suffers -13.9% generalization with LN")
print("="*80)
