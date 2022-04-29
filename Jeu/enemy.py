import pygame
from random import *
from PIL import Image

class Enemy(pygame.sprite.Sprite):

# initialisation class
	def __init__(self, game):
		# méthode game
		self.game = game

		super().__init__()

		#load image et gestion position
		self.image = pygame.image.load('assets/comet.png')
		self.image = pygame.transform.scale(self.image, (80,60))
		self.rect = self.image.get_rect()
		self.rect.x  = randint(0,500)
		self.rect.y = randint(-200, -100)

		#vitesse de l'enemy
		self.velocity = randint(2,4)

		# health
		self.health = 50
		self.max_health = 50

		#attack
		self.attack = 34

	def damage(self, amount):
		#infliger dégats
		self.health -= amount

		#vérifier si enemy est mort
		if self.health <= 0:
			#nouveau monstre
			self.health = self.max_health
			self.rect.x  = randint(0,500)
			self.rect.y = randint(-200, -100)

			#vitesse de l'enemy
			self.velocity = randint(2,4)
#chute enemy
	def forward(self):

		# vérifier collision avec joueur
		if not self.game.check_collision(self, self.game.all_players):

			# vérifier si pas collision avec sol
			if self.rect.y<250:
				self.rect.y += self.velocity

			# sinon dégat
			else:
				self.game.player.damage(self.attack)
				self.damage(self.health)

		else:
			self.damage(self.health)
			self.game.score +=1