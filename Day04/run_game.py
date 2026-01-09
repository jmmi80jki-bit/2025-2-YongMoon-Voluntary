#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이에서 실행할 완전한 Pong AI 게임
Frame Skip 적용 + TFLite 모델 사용
"""

import numpy as np
import tflite_runtime.interpreter as tflite
import time
import sys

# pong_game.py에서 PongEnv 임포트
from pong_game import PongEnv


class PongAgent:
    """Frame Skip이 적용된 TFLite AI 에이전트"""
    
    def __init__(self, model_path, frame_skip=4):
        """
        Args:
            model_path: TFLite 모델 파일 경로
            frame_skip: N 프레임마다 한 번 추론 (기본값: 4)
        """
        print(f"AI 에이전트 초기화 중...")
        
        # TFLite 인터프리터 로드
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=2  # 라즈베리파이 성능 최적화
        )
        self.interpreter.allocate_tensors()
        
        # 입/출력 텐서 정보
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Frame skip 설정
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.last_action = 1  # 초기 행동 (정지)
        
        # 성능 모니터링
        self.inference_times = []
        self.total_inferences = 0
        
        print(f"AI 로드 완료!")
        print(f"  └─ Frame skip: {frame_skip} (매 {frame_skip}프레임마다 추론)")
        print(f"  └─ 입력 shape: {self.input_details[0]['shape']}")
        print(f"  └─ 출력 shape: {self.output_details[0]['shape']}")
        
    def get_action(self, state):
        """
        Frame skip 적용된 행동 선택
        
        Args:
            state: numpy array [공x, 공y, 패들x, 공dx, 공dy]
        
        Returns:
            action: 0(왼쪽), 1(정지), 2(오른쪽)
        """
        self.frame_count += 1
        
        # Frame skip: N프레임마다 한 번만 추론
        if self.frame_count % self.frame_skip == 0:
            # 추론 시간 측정
            start_time = time.time()
            
            # 입력 데이터 준비
            input_data = np.array([state], dtype=np.float32)
            
            # TFLite 추론
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                input_data
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # 행동 선택 (Q값이 가장 큰 행동)
            self.last_action = np.argmax(output[0])
            
            # 추론 시간 기록
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            self.total_inferences += 1
        
        # Skip된 프레임에서는 이전 행동 재사용
        return self.last_action
    
    def get_stats(self):
        """성능 통계 반환"""
        if not self.inference_times:
            return None
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'total_inferences': self.total_inferences,
            'total_frames': self.frame_count
        }


def print_header():
    """게임 시작 헤더 출력"""
    print("\n" + "="*60)
    print("PONG AI")
    print("="*60)


def print_episode_start(episode, max_episodes):
    """에피소드 시작 메시지"""
    print(f"\n{'─'*60}")
    print(f"에피소드 {episode}/{max_episodes} 시작")
    print(f"{'─'*60}")


def print_episode_end(episode_score, episode_frames, episode_time):
    """에피소드 종료 메시지"""
    fps = episode_frames / episode_time if episode_time > 0 else 0
    print(f"   ✓ 점수: {episode_score}")
    print(f"   ✓ 프레임: {episode_frames}")
    print(f"   ✓ 시간: {episode_time:.2f}초")
    print(f"   ✓ FPS: {fps:.1f}")


def print_final_stats(total_score, total_frames, total_time, episodes, agent):
    """최종 통계 출력"""
    avg_score = total_score / episodes if episodes > 0 else 0
    avg_fps = total_frames / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print("🏆  최종 통계  🏆")
    print("="*60)
    print(f"📊 총 에피소드:     {episodes}회")
    print(f"📈 평균 점수:       {avg_score:.2f}점")
    print(f"🎯 최고 점수:       {max(total_score, 0)}점")
    print(f"⏱️  총 실행 시간:    {total_time:.2f}초")
    print(f"🖼️  평균 FPS:        {avg_fps:.1f}")
    
    # AI 성능 통계
    stats = agent.get_stats()
    if stats:
        skip_ratio = (stats['total_frames'] - stats['total_inferences']) / stats['total_frames'] * 100
        print(f"\n  AI 성능")
        print(f"{'─'*60}")
        print(f"   평균 추론 시간:  {stats['avg_inference_time']:.2f} ms")
        print(f"   최대 추론 시간:  {stats['max_inference_time']:.2f} ms")
        print(f"   최소 추론 시간:  {stats['min_inference_time']:.2f} ms")
        print(f"   총 추론 횟수:    {stats['total_inferences']}회")
        print(f"   Frame Skip 비율: {skip_ratio:.1f}%")
    
    print("="*60 + "\n")


def main():
    """메인 게임 루프"""
    
    # 설정
    MODEL_PATH = 'pong_model.tflite'
    FRAME_SKIP = 4  # 2~8 사이에서 조절 가능
    MAX_EPISODES = 10  # 실행할 에피소드 수
    RENDER_MODE = None  # 'human'으로 변경하면 화면 출력 (pygame 필요)
    
    # 헤더 출력
    print_header()
    print(f"📁 모델 파일: {MODEL_PATH}")
    print(f"🎲 Frame Skip: {FRAME_SKIP}")
    print(f"🎮 에피소드 수: {MAX_EPISODES}")
    print(f"🖥️  렌더링 모드: {'활성화' if RENDER_MODE == 'human' else '비활성화 (헤드리스)'}")
    
    try:
        # 1. AI 에이전트 초기화
        agent = PongAgent(
            model_path=MODEL_PATH,
            frame_skip=FRAME_SKIP
        )
        
        # 2. 게임 환경 초기화
        print(f"\n🎮 게임 환경 초기화 중...")
        env = PongEnv(render_mode=RENDER_MODE)
        print(f"✅ 게임 환경 로드 완료!")
        
        # 3. 게임 통계 변수
        total_score = 0
        total_frames = 0
        total_time = 0
        all_scores = []
        
        # 4. 게임 루프
        for episode in range(1, MAX_EPISODES + 1):
            print_episode_start(episode, MAX_EPISODES)
            
            # 에피소드 초기화
            state = env.reset()
            done = False
            episode_score = 0
            episode_frames = 0
            episode_start_time = time.time()
            
            # 에피소드 실행
            while not done:
                # AI가 행동 선택 (Frame skip 자동 적용)
                action = agent.get_action(state)
                
                # 환경에서 행동 실행
                state, reward, done, info = env.step(action)
                episode_score = info['score']
                episode_frames += 1
                
                # 렌더링 (RENDER_MODE='human'일 때만)
                if RENDER_MODE == 'human':
                    env.render()
            
            # 에피소드 종료
            episode_time = time.time() - episode_start_time
            print_episode_end(episode_score, episode_frames, episode_time)
            
            # 통계 업데이트
            total_score += episode_score
            total_frames += episode_frames
            total_time += episode_time
            all_scores.append(episode_score)
        
        # 5. 최종 통계 출력
        print_final_stats(total_score, total_frames, total_time, MAX_EPISODES, agent)
        
        # 6. 환경 종료
        env.close()
        
        return 0
        
    except FileNotFoundError:
        print(f"\n❌ 에러: '{MODEL_PATH}' 파일을 찾을 수 없습니다.")
        print(f"   └─ 현재 디렉토리에 TFLite 모델 파일이 있는지 확인하세요.")
        return 1
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 게임을 중단했습니다.")
        sys.exit(0)