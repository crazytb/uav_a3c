import torch
import sys

# .pth 파일 로드
pth_path = 'runs/a3c_20251005_024042/models/global_final.pth'
checkpoint = torch.load(pth_path, map_location='cpu')

# checkpoint가 딕셔너리인지, 직접 state_dict인지 확인
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    has_metadata = True
else:
    state_dict = checkpoint
    has_metadata = False

print('='*60)
print('=== .pth 파일 최상위 키 ===')
print('='*60)
print(list(checkpoint.keys()))
print()

print('='*60)
print('=== 모델 state_dict 레이어 목록 및 Shape ===')
print('='*60)
for key in state_dict.keys():
    shape = state_dict[key].shape
    dtype = state_dict[key].dtype
    num_params = state_dict[key].numel()
    print(f'{key:40s} | Shape: {str(shape):25s} | params: {num_params:>8,} | dtype: {dtype}')
print()

print('='*60)
print('=== 총 파라미터 수 ===')
print('='*60)
total_params = sum(p.numel() for p in state_dict.values())
print(f'총 파라미터: {total_params:,}')
print()

if has_metadata:
    print('='*60)
    print('=== 메타데이터 ===')
    print('='*60)
    for key in sorted(checkpoint.keys()):
        if key != 'model_state_dict':
            print(f'{key:20s}: {checkpoint[key]}')
    print()
else:
    print('='*60)
    print('=== 메타데이터 ===')
    print('='*60)
    print('메타데이터 없음 (가중치만 저장됨)')
    print()

# 모델 아키텍처 추론
print('='*60)
print('=== 추론된 모델 아키텍처 정보 ===')
print('='*60)
has_rnn = any('rnn' in key.lower() or 'gru' in key.lower() or 'lstm' in key.lower()
              for key in state_dict.keys())
has_feature = any('feature' in key.lower() for key in state_dict.keys())
has_policy = any('policy' in key.lower() or 'actor' in key.lower() for key in state_dict.keys())
has_value = any('value' in key.lower() or 'critic' in key.lower() for key in state_dict.keys())

print(f'RNN/GRU/LSTM 레이어: {has_rnn}')
print(f'Feature 레이어: {has_feature}')
print(f'Policy (Actor) 레이어: {has_policy}')
print(f'Value (Critic) 레이어: {has_value}')
print(f'총 레이어 개수: {len(state_dict.keys())}')

# 입력/출력 차원 추론
if 'feature.0.weight' in state_dict:
    input_dim = state_dict['feature.0.weight'].shape[1]
    print(f'입력 차원: {input_dim}')
if 'policy.weight' in state_dict:
    action_dim = state_dict['policy.weight'].shape[0]
    print(f'행동 차원 (출력): {action_dim}')
if 'rnn.weight_ih_l0' in state_dict:
    hidden_dim = state_dict['rnn.weight_hh_l0'].shape[0]
    print(f'RNN 은닉 차원: {hidden_dim}')
