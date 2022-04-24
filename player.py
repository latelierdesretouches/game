from matplotlib.pyplot import cla
import pygame


class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        # accès a la classe game
        self.game = game

        #vitesse
        self.velocity = 10

        #image et coordonnées joueur
        self.image = pygame.image.load('assets/PJ.png')
        self.image = pygame.transform.scale(self.image, (120,150))
        self.rect = self.image.get_rect()
        self.rect.x = 400
        self.rect.y = 500

        # health
        self.max_health=50
        self.health=50


# barre de vie joueur
    def update_health_bar(self, surface):

        # dessiner arrière plan jauge
        pygame.draw.rect(surface, (60,63,60), [self.rect.x+35, self.rect.y, self.max_health, 5])

        #dessiner la jauge
        pygame.draw.rect(surface, (111, 210, 46), [self.rect.x+35, self.rect.y, self.health, 5])

    def damage(self, amount):
        #dégats joueur
        self.health -= amount

        # vérifier si joueur mort
        if self.health < 1:
            self.game.game_over()       