import numpy as np

class PongEnv:
    def __init__(self, render_mode=None, target_fps = 60):
        """
        환경 초기화
        
        Args:
            render_mode: 'human'이면 화면 렌더링, None이면 헤드리스 모드
        """
        self.width = 800
        self.height = 600

        self.ball_x = None
        self.ball_y = None
        self.ball_dx = None
        self.ball_dy = None

        self.paddle_x = None  # 패들 가로 위치 (0~1)
        self.paddle_width = 0.2  # 패들 너비 (화면 대비 비율)
        
        # 점수 추가
        self.score = 0

        # 액션 스페이스와 관측 스페이스 정의
        self.action_space_n = 3  # 0: 왼쪽, 1: 정지, 2: 오른쪽
        self.observation_space_shape = (5,)
        
        # 렌더링 설정
        self.render_mode = render_mode
        self.target_fps = target_fps

        # 화면 렌더링 변수
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        if render_mode == 'human':
            try:
                import pygame
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Pong 게임")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 74)
                self.small_font = pygame.font.Font(None, 48)
            except ImportError:
                print("pygame이 설치되지 않았습니다. headless mode로 실행하거나 pygame을 설치하세요.")
                self.render_mode = None

    def reset(self):
        """환경을 초기 상태로 리셋"""
        self.ball_x = 0.5  # 화면 중앙
        self.ball_y = 0.1  # 화면 상단
        self.paddle_x = 0.5  # 패들 중앙
        self.ball_dx = np.random.choice([-0.01, 0.01])  # 좌우 랜덤
        self.ball_dy = 0.01  # 아래로 이동
        
        self.score = 0

        return self._get_state()

    def step(self, action):
        """
        action:
        0 = 왼쪽으로 이동
        1 = 정지
        2 = 오른쪽으로 이동

        반환값:
        state : 현재 상태
        reward : 보상
        done : 에피소드 종료 여부
        info : 추가 정보 (딕셔너리)
        """
        # ✏️ TODO 1: 패들 이동 로직 구현
        # action이 0이면 왼쪽으로 0.01 이동
        # action이 2이면 오른쪽으로 0.01 이동
        # hint: self.paddle_x를 증가 또는 감소시키세요
       
        if action == 0:
            self.paddle_x -= 0.01 # 왼쪽
        elif action == 2:
            self.paddle_x += 0.01 # 오른쪽
              
         
         
         
        
        
        
        
        # 패들이 화면 밖으로 나가지 않도록 제한
        self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

        # ✏️ TODO 2: 공 이동 로직 구현
        # 공의 x 위치에 dx 속도를 더하기
        # 공의 y 위치에 dy 속도를 더하기
        # hint: self.ball_x += ?, self.ball_y += ?
            
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        
        reward = 0.0
        done = False

        # ✏️ TODO 3: 좌우 벽 충돌 구현
        # 공의 x 위치가 0.0 이하이거나 1.0 이상이면
        # ball_dx의 부호를 반대로 바꾸기
        # hint: self.ball_dx *= -1
          
        if self.ball_x <=0.0 or self.ball_x >= 1.0:
            self.ball_dx *= -1     
        
        
        
        
        
        
        # ✏️ TODO 4: 위쪽 벽 충돌 구현
        # 공의 y 위치가 0.0 이하이면
        # ball_dy의 부호를 반대로 바꾸기
        
      
        if self.ball_y <=0.0:
            self.ball_dy *= -1

        # ✏️ TODO 5: 패들 충돌 및 점수 시스템 구현
        # 공의 y 위치가 0.95 이상이면:
        #   - 공과 패들의 거리가 paddle_width/2 이하이면:
        #     * ball_dy의 부호를 반대로 (공이 튕김)
        #     * score를 1 증가
        #     * reward를 1.0으로 설정
        #   - 그렇지 않으면 (공을 놓침):
        #     * reward를 -1.0으로 설정
        #     * done을 True로 설정
        # hint: abs(self.ball_x - self.paddle_x)로 거리 계산
        
        if  self.ball_y >=0.95:
            if abs(self.ball_x - self.paddle_x) <= self.paddle_width / 2:
                #패들에맞음 
                self.ball_dy *= -1
                self.score +=1
                reward = 1.0
            else:
        #  패들놓침 게임오버
                reward = -1.0
                done = True
        
        
        
        
        
        
        
        
        
        
        

        # info 딕셔너리
        info = {
            'score': self.score,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """
        ✏️ TODO 6: AI에게 줄 입력값(관측값) 추출
        현재 상태를 numpy 배열로 반환
        
        반환할 값들:
        - self.ball_x: 공의 x 위치
        - self.ball_y: 공의 y 위치
        - self.paddle_x: 패들의 x 위치
        - self.ball_dx: 공의 x 방향 속도
        - self.ball_dy: 공의 y 방향 속도
        
        hint: np.array([값1, 값2, ...], dtype=np.float32)
        """
        return np.array([self. ball_x, self. ball_y, self.paddle_x, self.ball_dx, self.ball_dy], dtype=np.float32)


    # 화면 렌더링 관련 함수
    def render(self):
        """
        게임 화면 렌더링 (render_mode='human'일 때만 작동)
        """
        if self.render_mode != 'human' or self.screen is None:
            return
        
        import pygame
        
        # 배경
        self.screen.fill((0, 0, 0))
        
        # 패들 (아래쪽, 가로로)
        paddle_pixel_x = int(self.paddle_x * self.width)
        paddle_pixel_width = int(self.paddle_width * self.width)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (paddle_pixel_x - paddle_pixel_width // 2, self.height - 30,
                         paddle_pixel_width, 15))
        
        # 공
        ball_pixel_x = int(self.ball_x * self.width)
        ball_pixel_y = int(self.ball_y * self.height)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (ball_pixel_x - 7, ball_pixel_y - 7, 15, 15))
        
        # 점수 표시
        score_text = self.font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (self.width // 2 - 30, 50))

        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(self.target_fps)
    
    def render_game_over(self):
        """게임 오버 화면 렌더링"""
        if self.render_mode != 'human' or self.screen is None:
            return None
        
        import pygame
        
        # 반투명 오버레이
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # GAME OVER 텍스트
        game_over_font = pygame.font.Font(None, 120)
        game_over_text = game_over_font.render("GAME OVER", True, (255, 50, 50))
        text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 100))
        self.screen.blit(game_over_text, text_rect)
        
        # 최종 점수 표시
        score_font = pygame.font.Font(None, 60)
        score_text = score_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 - 20))
        self.screen.blit(score_text, score_rect)
        
        # 다시 시작 버튼
        button_width = 250
        button_height = 60
        button_x = self.width // 2 - button_width // 2
        button_y = self.height // 2 + 50
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        
        # 마우스 호버 체크
        mouse_pos = pygame.mouse.get_pos()
        is_hover = button_rect.collidepoint(mouse_pos)
        
        # 버튼 모양
        button_color = (100, 255, 100) if is_hover else (50, 200, 50)
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=15)
        pygame.draw.rect(self.screen, (255, 255, 255), button_rect, 4, border_radius=15)
        
        # 버튼 텍스트
        button_font = pygame.font.Font(None, 48)
        button_text = button_font.render("RESTART", True, (0, 0, 0))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        pygame.display.flip()
        
        return button_rect

    def close(self):
        """환경 종료 및 리소스 정리"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None


# ====================================================================
# 테스트 코드
# ====================================================================
if __name__ == "__main__":
    import pygame
    
    # 렌더링 모드로 환경 생성
    env = PongEnv(render_mode='human')
    
    # 게임 초기화
    state = env.reset()
    print(f"초기 상태: {state}")
    print(f"상태 형태: {state.shape}")
    print(f"액션 스페이스: {env.action_space_n}")
    print("\n조작법: 왼쪽 화살표=왼쪽, 오른쪽 화살표=오른쪽")
    
    # 게임 플레이 (키보드 조작)
    running = True
    total_reward = 0
    steps = 0
    action = 1  # 초기 액션: 정지
    
    while running:
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 키보드 입력 받기
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0  # 왼쪽
        elif keys[pygame.K_RIGHT]:
            action = 2  # 오른쪽
        else:
            action = 1  # 정지
        
        # 스텝 실행
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # 렌더링
        env.render()
        
        # 상태 정보 출력
        if steps % 100 == 0:
            print(f"Steps: {steps}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"Score: {info['score']}")
        
        # 게임 종료 시 리셋
        if done:
            # GAME OVER 화면 표시 및 버튼 대기
            waiting = True
            while waiting:
                button_rect = env.render_game_over()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    # 마우스 클릭 체크
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect and button_rect.collidepoint(event.pos):
                            waiting = False
            
            print(f"\n게임 종료! 총 스텝: {steps}, 총 보상: {total_reward:.2f}")
            print(f"최종 점수 - Score: {info['score']}")
            state = env.reset()
            total_reward = 0
            steps = 0
    
    env.close()
