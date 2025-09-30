import torch.multiprocessing as mp
import numpy as np
import time

class NetworkState:
    """
    다중 워커 간 공유되는 네트워크 상태를 관리하는 클래스
    Episode-level 동기화에서 네트워크 혼잡도를 추적합니다.
    """
    
    def __init__(self, max_workers, max_network_capacity=1000.0, max_cloud_capacity=10000):
        self.max_workers = max_workers
        self.max_capacity = max_network_capacity
        self.max_cloud_capacity = max_cloud_capacity

        # 공유 메모리 변수들
        self.total_offloading_load = mp.Value('f', 0.0)  # 현재 총 오프로딩 부하
        self.current_episode = mp.Value('i', 0)          # 현재 에피소드 번호
        self.active_workers = mp.Value('i', 0)           # 현재 활성 워커 수
        self.available_cloud_capacity = mp.Value('f', max_cloud_capacity)

        # 워커별 통계 (Array는 고정 크기여야 함)
        self.worker_offload_counts = mp.Array('i', [0] * max_workers)
        self.worker_total_loads = mp.Array('f', [0.0] * max_workers)
        
        # 동기화용 락
        self.lock = mp.Lock()
        
        print(f"[NetworkState] Initialized for {max_workers} workers, max capacity: {max_network_capacity}")
    
    def register_worker_start(self, worker_id):
        """워커가 에피소드를 시작할 때 호출"""
        with self.lock:
            self.active_workers.value += 1
            # print(f"[NetworkState] Worker {worker_id} started episode {self.current_episode.value}, active workers: {self.active_workers.value}")
    
    def register_worker_end(self, worker_id):
        """워커가 에피소드를 끝낼 때 호출"""
        with self.lock:
            self.active_workers.value -= 1
            # print(f"[NetworkState] Worker {worker_id} finished episode {self.current_episode.value}, active workers: {self.active_workers.value}")
    
    def add_offloading_load(self, worker_id, load_amount):
        """워커가 오프로딩할 때 네트워크 부하 추가"""
        with self.lock:
            self.total_offloading_load.value += load_amount
            self.worker_offload_counts[worker_id] += 1
            self.worker_total_loads[worker_id] += load_amount
            
            # 네트워크 용량 초과 방지
            if self.total_offloading_load.value > self.max_capacity:
                self.total_offloading_load.value = self.max_capacity
    
    def remove_offloading_load(self, load_amount):
        """처리 완료된 오프로딩 부하 제거"""
        with self.lock:
            self.total_offloading_load.value = max(0.0, self.total_offloading_load.value - load_amount)
    
    def get_congestion_level(self):
        """현재 네트워크 혼잡도 반환 (0.0 ~ 1.0)"""
        with self.lock:
            congestion = min(self.total_offloading_load.value / self.max_capacity, 1.0)
            return congestion
        
    def consume_cloud_resource(self, worker_id, resource_amount):
        """클라우드 자원 소모"""
        with self.lock:
            if self.available_cloud_capacity.value >= resource_amount:
                self.available_cloud_capacity.value -= resource_amount
                return True  # 자원 할당 성공
            else:
                return False  # 자원 부족
    
    def release_cloud_resource(self, resource_amount):
        """클라우드 자원 회복"""
        with self.lock:
            self.available_cloud_capacity.value = min(
                self.max_cloud_capacity, 
                self.available_cloud_capacity.value + resource_amount
            )
    
    def get_cloud_utilization(self):
        """클라우드 사용률 반환"""
        with self.lock:
            utilization = 1.0 - (self.available_cloud_capacity.value / self.max_cloud_capacity)
            return utilization
    
    def get_network_stats(self):
        """현재 네트워크 통계 반환"""
        with self.lock:
            return {
                'total_load': self.total_offloading_load.value,
                'congestion_level': self.get_congestion_level(),
                'active_workers': self.active_workers.value,
                'current_episode': self.current_episode.value,
                'worker_offload_counts': list(self.worker_offload_counts[:self.max_workers]),
                'worker_total_loads': list(self.worker_total_loads[:self.max_workers])
            }
    
    def reset_for_new_episode(self):
        """새 에피소드를 위한 네트워크 상태 리셋"""
        with self.lock:
            # 이전 에피소드의 잔여 부하는 일정 비율만 유지 (현실적 모델링)
            self.total_offloading_load.value *= 0.1  # 10%만 다음 에피소드로 이월

            # ⭐ 클라우드 자원 완전 회복 (에피소드마다 리셋)
            self.available_cloud_capacity.value = self.max_cloud_capacity

            # 에피소드 번호 증가
            self.current_episode.value += 1

            # 워커별 통계는 누적 유지 (전체 학습 과정 추적용)

            # print(f"[NetworkState] Reset for episode {self.current_episode.value}, cloud capacity reset to {self.max_cloud_capacity}")
    
    def print_episode_summary(self):
        """에피소드 종료 시 요약 정보 출력"""
        stats = self.get_network_stats()
        print(f"\n[Episode {stats['current_episode']} Summary]")
        print(f"  Final network load: {stats['total_load']:.2f}")
        print(f"  Final congestion: {stats['congestion_level']:.3f}")
        print(f"  Worker offload counts: {stats['worker_offload_counts']}")
        print(f"  Worker total loads: {[f'{x:.1f}' for x in stats['worker_total_loads']]}")
        print()


class NetworkCongestionCalculator:
    """
    네트워크 혼잡도 기반 보상 계산을 위한 유틸리티 클래스
    """
    
    def __init__(self, base_penalty=20.0, congestion_threshold=0.5):
        self.base_penalty = base_penalty
        self.threshold = congestion_threshold
    
    def calculate_congestion_penalty(self, congestion_level):
        """
        혼잡도에 따른 페널티 계산
        
        Args:
            congestion_level: 0.0 ~ 1.0 사이의 혼잡도
            
        Returns:
            penalty: 혼잡도에 따른 페널티 값
        """
        if congestion_level <= self.threshold:
            # 낮은 혼잡도에서는 페널티 없음
            return 0.0
        else:
            # 임계값 초과 시 지수적 페널티 증가
            excess_congestion = congestion_level - self.threshold
            penalty = self.base_penalty * (excess_congestion / (1.0 - self.threshold)) ** 2
            return penalty
    
    def calculate_offload_reward_modifier(self, congestion_level):
        """
        혼잡도에 따른 오프로딩 보상 수정자
        
        Returns:
            modifier: 1.0 기준으로 보상을 조정하는 배수
        """
        if congestion_level <= 0.3:
            return 1.2  # 네트워크 여유 시 오프로딩 장려
        elif congestion_level <= 0.7:
            return 1.0  # 보통 수준
        else:
            return 0.5  # 혼잡 시 오프로딩 억제