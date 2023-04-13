# REFERENCES:
# https://rmgi.blog/pygame-2d-car-tutorial.html
# https://www.pygame.org/docs/tut/PygameIntro.html

import pygame
from math import sin
from Ship import Ship

car_img = pygame.image.load("car.png")
car_img = pygame.transform.scale(car_img, (200, 100))

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        # self.clock is an object that help tracking time (https://www.pygame.org/docs/ref/time.html#pygame.time.Clock)
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.tot = 0  # Total time
        self.car_img = car_img
        self.car_rect = self.car_img.get_rect()
        self.speed = [2,2]
    def run(self):
        while not self.exit:
            dt = self.clock.get_time() / 1000  # self.clock.get_time() gives the time in millisec
            self.tot = self.tot + dt

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()

            # Logic
            # Car logic goes here

            # The following snippet moves a car image around the screen, bouncing against the walls.
            self.car_rect = self.car_rect.move(self.speed)
            if self.car_rect.left < 0 or self.car_rect.right > self.screen.get_width():
                self.speed[0] = -self.speed[0]
            if self.car_rect.top < 0 or self.car_rect.bottom > self.screen.get_height():
                self.speed[1] = -self.speed[1]

            # Drawing
            """
            self.screen.fill((255*sin(self.tot/5)*sin(self.tot/5),
                              round((self.tot*12) % 255),
                              round((self.tot*6) % 255)))  # Just screen background color
            
            self.screen.fill((255 * sin(self.tot / 5) * sin(self.tot / 5),
                              255 * sin(self.tot / 4) * sin(self.tot / 4),
                              255 * sin(self.tot / 7) * sin(self.tot / 7)))  # Just screen background color
            """
            self.screen.fill((0, 0, 0))  # Just screen background color

            self.screen.blit(self.car_img, self.car_rect)
            pygame.display.flip()

            self.clock.tick(self.ticks)  # This function will set the framerate to a max of 60fps

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()

