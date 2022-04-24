import pygame


class SoundManager:
    def __init__(self):
        self.sounds = {
            'banner_sound': pygame.mixer.Sound('assets/sounds/soundspause.mp3'),
            'wow': pygame.mixer.Sound('assets/sounds/wow.mp3'),
            'beurk' : pygame.mixer.Sound('assets/sounds/beurk.mp3')}

    def play(self, name):
        self.sounds[name].play()
