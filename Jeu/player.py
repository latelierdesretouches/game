from matplotlib.pyplot import cla
import pygame


class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        # accès a la classe game
        self.game = game

        #vitesse
        self.velocity = 40

        #image et coordonnées joueur
        self.image = pygame.image.load('assets/player.png')
        self.rect = self.image.get_rect()
        self.rect.x = 200
        self.rect.y = 150

        # health
        self.max_health=100
        self.health=100


# barre de vie joueur
    def update_health_bar(self, surface):

        # dessiner arrière plan jauge
        pygame.draw.rect(surface, (60,63,60), [self.rect.x+50, self.rect.y+15, self.max_health, 7])

        #dessiner la jauge
        pygame.draw.rect(surface, (111, 210, 46), [self.rect.x+50, self.rect.y+15, self.health, 7])

    def damage(self, amount):
        #dégats joueur
        self.health -= amount

        # vérifier si joueur mort
        if self.health < 1:
            self.game.game_over()




