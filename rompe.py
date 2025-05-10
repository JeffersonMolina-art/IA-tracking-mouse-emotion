import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

pygame.init()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Juego Rompe Bloques")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)
big_font = pygame.font.SysFont(None, 72)

# Recursos
paddle_image = pygame.image.load("paddle.png")
paddle_image = pygame.transform.scale(paddle_image, (150, 40))
ball_image = pygame.image.load("esfera.png")
ball_image = pygame.transform.scale(ball_image, (40, 40))
block_image = pygame.image.load("ladrillo.png")
block_rows, block_colums = 4, 8
block_width, block_height = WIDTH // block_colums, 50
block_image = pygame.transform.scale(block_image, (block_width, block_height))

# Variables del juego
paddle_x, paddle_y = WIDTH // 2, HEIGHT - 100
paddle_width, paddle_height = 150, 40
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = 8, 8
ball_radius = 20
score = 0
lives = 3
powerups = []
powerup_size = 30
powerup_effect_time = 300  # frames

def create_blocks():
    return [pygame.Rect(col * block_width, row * block_height, block_width, block_height)
            for row in range(block_rows) for col in range(block_colums)]

blocks = create_blocks()
powerup_active = False
powerup_timer = 0

def show_message(text):
    screen.fill((0, 0, 0))
    message = big_font.render(text, True, (255, 255, 255))
    screen.blit(message, (WIDTH//2 - message.get_width()//2, HEIGHT//2 - message.get_height()//2))
    pygame.display.flip()
    pygame.time.delay(3000)

def draw_info():
    score_text = font.render(f"Puntos: {score}", True, (255, 255, 255))
    lives_text = font.render(f"Vidas: {lives}", True, (255, 100, 100))
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (WIDTH - lives_text.get_width() - 10, 10))

running = True
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        mano_detectada = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mano_detectada = True
                index_finger = hand_landmarks.landmark[8]
                paddle_x = int(index_finger.x * WIDTH) - paddle_width // 2
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if mano_detectada:
            ball_x += ball_speed_x
            ball_y += ball_speed_y

        if ball_x - ball_radius < 0 or ball_x + ball_radius > WIDTH:
            ball_speed_x *= -1
        if ball_y - ball_radius < 0:
            ball_speed_y *= -1
        if ball_y + ball_radius > HEIGHT:
            lives -= 1
            if lives == 0:
                show_message("¡Perdiste!")
                running = False
            else:
                ball_x, ball_y = WIDTH // 2, HEIGHT // 2

        if paddle_x < 0:
            paddle_x = 0
        if paddle_x + paddle_width > WIDTH:
            paddle_x = WIDTH - paddle_width

        if paddle_x < ball_x < paddle_x + paddle_width and            paddle_y < ball_y + ball_radius < paddle_y + paddle_height and            ball_speed_y > 0:
            ball_speed_y *= -1

        for block in blocks:
            if block.collidepoint((ball_x, ball_y)):
                blocks.remove(block)
                ball_speed_y *= -1
                score += 10
                if random.random() < 0.2:
                    powerups.append(pygame.Rect(block.x + block_width//2 - powerup_size//2, block.y, powerup_size, powerup_size))
                break

        paddle_rect = pygame.Rect(paddle_x, paddle_y, paddle_width, paddle_height)
        for p in powerups[:]:
            p.y += 5
            if p.colliderect(paddle_rect):
                powerup_active = True
                powerup_timer = powerup_effect_time
                paddle_width = 200
                powerups.remove(p)
            elif p.y > HEIGHT:
                powerups.remove(p)

        if powerup_active:
            powerup_timer -= 1
            if powerup_timer <= 0:
                powerup_active = False
                paddle_width = 150

        if not blocks:
            show_message("¡Ganaste!")
            running = False

        rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))
        rgb_frame = np.rot90(rgb_frame)
        surface = pygame.surfarray.make_surface(rgb_frame)
        screen.blit(surface, (0, 0))

        screen.blit(paddle_image, (paddle_x, paddle_y))
        screen.blit(ball_image, (ball_x - ball_radius, ball_y - ball_radius))
        for block in blocks:
            screen.blit(block_image, (block.x, block.y))
        for p in powerups:
            pygame.draw.rect(screen, (255, 215, 0), p)
        draw_info()

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

cap.release()
cv2.destroyAllWindows()