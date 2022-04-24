import imp
import pygame
from player import Player
from enemy import Enemy
from sounds import SoundManager


class Game():
    def __init__(self):

        # définir si le jeu a commencé
        self.is_playing = False

        # groupe joueur
        self.all_players = pygame.sprite.Group()

        # création de joueur
        self.player = Player(self)

        # ajout du joueur au groupe joueur
        self.all_players.add(self.player)

        # groupe d'enemy
        self.all_enemy = pygame.sprite.Group()

        # gerer son
        self.sound_manager = SoundManager()

        # mettre score à 0
        self.score = 0

        # score de la partie précédente
        self.previous_score = 0

        # liste de touches pressées
        self.pressed = {}


    def start(self):
        self.is_playing = True
        # spawn enemy
        self.spawn_enemy()
        self.spawn_enemy()
        self.spawn_enemy()


    def game_over(self):
        # remettre jeu à neuf: retirer monstres joueur à 100 jeu attente
        self.all_enemy = pygame.sprite.Group()
        self.player.health = self.player.max_health
        self.is_playing = False
        self.previous_score = self.score
        self.score = 0
        print(self.previous_score)

    def update(self, screen):
        # afficher le score sur l'écran
        font = pygame.font.SysFont('monospace', 16)
        score_text = font.render(f'Score : {self.score}', 1, (0,0,0))
        screen.blit(score_text, (20, 20))
        # appliquer l'image du joueur
        screen.blit(self.player.image, self.player.rect)

        # actualiser jauge du joueur
        self.player.update_health_bar(screen)


        # récupérer enemy
        for enemy in self.all_enemy:
            # Faire avancer
            enemy.forward()

        # appliquer l'image de l'enemy
        self.all_enemy.draw(screen)

        # verif si le joueur veut aller à gauche ou à droite
        if self.pressed.get(pygame.K_RIGHT) and self.player.rect.x + self.player.rect.width <= screen.get_width():
            self.move_right()

        elif self.pressed.get(pygame.K_LEFT) and self.player.rect.x >= 0:
            self.move_left()


    def move_right(self):
        self.player.rect.x += self.player.velocity

    def move_left(self):
        self.player.rect.x -= self.player.velocity

    def check_collision(self, sprite, group):
        return pygame.sprite.spritecollide(sprite, group, False, pygame.sprite.collide_mask)

    def spawn_enemy(self):
        enemy = Enemy(self)
        self.all_enemy.add(enemy)
