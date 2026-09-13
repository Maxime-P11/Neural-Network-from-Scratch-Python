import numpy as np
from os import path
from NN_Numpy_class import NNFromScratch
import pygame
from cv2 import resize
import cv2
from math import ceil

FCNN = NNFromScratch(0.03, [])

CURRENTDIR = path.dirname(path.realpath(__file__))
MODELSAVEPATH = path.join(CURRENTDIR,'model.pkl')
if path.exists(MODELSAVEPATH):
    FCNN.load_model(MODELSAVEPATH)



pygame.init()
SIZE = 512
OFFSETY = 80
GRIDNUM = 128
GRIDSIZE = SIZE//GRIDNUM

RANGEPENCIL = 8

screen = pygame.display.set_mode((SIZE, SIZE+OFFSETY))
pygame.display.set_caption("AI")
clock = pygame.time.Clock()

font = pygame.font.SysFont('arial', OFFSETY-5)
griddraw = np.zeros((GRIDNUM, GRIDNUM))


def draw(x, y, color, surface, grid):
    color = max(color, grid[y,x])

    pygame.draw.rect(surface, (color*255, color*255, color*255), (x * GRIDSIZE, y * GRIDSIZE+OFFSETY, GRIDSIZE, GRIDSIZE))

    grid[y,x] = color 
    pass



def calc_output(model:NNFromScratch, grid:np.array):
    #make drawing insensible to translation and size diff
    numonly = np.copy(grid)
    numonly = np.copy(numonly[~np.all(numonly == 0, axis=1)])
    numonly = numonly[:, ~np.all(numonly == 0, axis=0)]


    width, height = numonly.shape
    max_dim = max(width, height)

    pad_amount_w = max_dim - width
    pad_amount_h = max_dim - height
    pad_width = ((pad_amount_w//2, ceil(pad_amount_w/2)), (pad_amount_h//2, ceil(pad_amount_h/2)))

    numonly = np.pad(numonly, pad_width, mode='constant', constant_values=0)

    #resize from cv2
    numonly = resize(numonly, (20,20))

    #sentimage = np.repeat(np.repeat(numonly, 8, axis=0), 8, axis=1)
    sentimage = numonly*255
    sentimage = np.array([[(item, item, item) for item in row] for row in sentimage])
    sentimage = np.rot90(sentimage)
    sentimage = np.flip(sentimage, axis=0)


    numonly = np.pad(numonly, 4)


    #forward pass
    numonly = numonly.flatten()
    numonly = np.reshape(numonly, (1, 784))
    return model.forward(numonly), sentimage

    


pygame.draw.rect(screen, (120,120,120), (OFFSETY, 0, SIZE-OFFSETY, OFFSETY))
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        elif event.type == pygame.KEYDOWN:
            pygame.draw.rect(screen, (0,0,0), (0, OFFSETY, SIZE, SIZE))
            griddraw = np.zeros((GRIDNUM, GRIDNUM))




    
    if pygame.mouse.get_pressed()[0]:
        pygame.draw.rect(screen, (120,120,120), (OFFSETY, 0, SIZE-OFFSETY, OFFSETY))
        pos=pygame.mouse.get_pos()
        x = int(pos[0]/GRIDSIZE)
        y = int((pos[1]-OFFSETY)/GRIDSIZE)
        x = max(min(x, GRIDNUM-1), 0)
        y = max(min(y, GRIDNUM-1), 0)

        for xoffset in range(-RANGEPENCIL, RANGEPENCIL+1):
            for yoffset in range(-RANGEPENCIL, RANGEPENCIL+1):
                xpos = max(min(x+xoffset, GRIDNUM-1), 0)
                ypos = max(min(y+yoffset, GRIDNUM-1), 0)


                color = abs(xoffset) + abs(yoffset) - 2
                color = 1 / max(color, 1)

                draw(xpos, ypos, color, screen, griddraw)

        result, image_sent = calc_output(FCNN, griddraw)
        best = np.argmax(result)
        text_surface = font.render('Prediction : '+str(best) , False, (255, 255, 255))
        screen.blit(text_surface, (OFFSETY,0))

        surf = pygame.Surface((20, 20))

        pygame.surfarray.blit_array(surf, image_sent)
        surf = pygame.transform.scale(surf, (OFFSETY, OFFSETY))
        screen.blit(surf)

    pygame.display.flip()