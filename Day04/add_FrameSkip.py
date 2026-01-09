#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이에서 실행할 Pong 게임 추론 코드
Frame skip 적용으로 성능 최적화
"""

import numpy as np
import tflite_runtime.interpreter as tflite
import time

class PongAgent:
    def __init__(self, model_path, frame_skip=4):
        """
        Args:
            model_path: TFLite 모델 경로 (예: 'pong_model.tflite')
            frame_skip: N 프레임마다 한 번 추론 (기본값: 4)
        """
        # TFLite 인터프리터 초기화
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=2  # 라즈베리파이 성능에 맞춤
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Frame skip 설정
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.last_action = 1  # 초기 행동 (정지)

        print(f"✅ TFLite 모델 로드 완료!")
        print(f"📊 Frame skip: {frame_skip} (매 {frame_skip}프레임마다 추론)")

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

        # Skip된 프레임에서는 이전 행동 재사용
        return self.last_action


# ============================================
# 사용 예시 (실제 게임 환경에 맞춰 수정)
# ============================================

if __name__ == "__main__":
    # 1. 에이전트 초기화
    agent = PongAgent(
        model_path='pong_model.tflite',
        frame_skip=4  # 4프레임마다 1번 추론
    )

    # 2. 게임 루프 시뮬레이션
    print("\n🎮 게임 시작!")

    for frame in range(100):
        # 현재 게임 상태 가져오기 (실제 환경에서는 센서나 게임 엔진에서 가져옴)
        # 예시: [공x, 공y, 패들x, 공dx, 공dy]
        state = np.array([
            0.5 + np.random.randn() * 0.1,  # 공 x
            0.5 + np.random.randn() * 0.1,  # 공 y
            0.5 + np.random.randn() * 0.1,  # 패들 x
            0.1,  # 공 dx
            0.1   # 공 dy
        ], dtype=np.float32)

        # 행동 선택
        action = agent.get_action(state)

        # 행동 실행 (실제 환경에서는 모터 제어 등)
        action_names = ['←왼쪽', '정지', '→오른쪽']
        if frame % 10 == 0:  # 10프레임마다 출력
            print(f"Frame {frame}: 행동 = {action_names[action]}")

        # 짧은 딜레이 (실제 게임 프레임레이트 시뮬레이션)
        time.sleep(0.033)  # ~30 FPS

    print("\n✅ 게임 종료!")