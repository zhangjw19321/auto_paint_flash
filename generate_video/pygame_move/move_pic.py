import pygame,sys
from pygame.locals import *
pygame.init()
FPS = 30 #frames per second setting
fpsClock = pygame.time.Clock()
# set background picture
background = pygame.image.load('xingkong2.png')

#set up the window
DISPLAYSURF = pygame.display.set_mode((948,532,),0,32)
# DISPLAYSURF=pygame.image.load('img/katong.png').convert()
pygame.display.set_caption('Animation')
GREEN = (0,255,0)
WHITE = (255,255,255)
catImg = pygame.image.load('img_alpha.png')
catx=10
caty=10
direction='right'
while True:
	# DISPLAYSURF.fill(GREEN)
	DISPLAYSURF.blit(background,(0,0))
	if direction == 'right':
		catx+=10
		if catx == 620:
			direction = 'down'
	elif direction=='down':
		caty+=10
		if caty == 300:
			direction='left'
	elif direction == 'left':
		catx-=10
		if catx==10:
			direction='up'
	elif direction == 'up':
		caty -=10
		if caty==10:
			direction = 'right'
	DISPLAYSURF.blit(catImg,(catx,caty))
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			sys.exit()
	pygame.display.update()
	fpsClock.tick(FPS)