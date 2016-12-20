#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import os
# import kezmenu

from tetrominoes import list_of_tetrominoes
from tetrominoes import rotate

from scores import load_score, write_score

class BrokenMatrixException(Exception):
    pass


# def get_sound(filename):
#     return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

BGCOLOR = (0, 0, 0)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

WIDTH = 700
HEIGHT = 20*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22
VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2


class Matris(object):
    def __init__(self, size=(MATRIX_WIDTH, MATRIX_HEIGHT), blocksize=BLOCKSIZE):
        self.size = {'width': size[0], 'height': size[1]}
        self.blocksize = blocksize
        self.surface = Surface((self.size['width']  * self.blocksize,
                                (self.size['height']-2) * self.blocksize))


        self.matrix = dict()
        for y in range(self.size['height']):
            for x in range(self.size['width']):
                self.matrix[(y,x)] = None


        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4 # Move down every 400 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.paused = False
        self.gameover = False

        self.highscore = load_score()

        self.action_space = []
        for i in range(4):
            for j in range(-4,7):
                self.action_space.append([i,j])
        # self.played_highscorebeaten_sound = False

        # self.levelup_sound  = get_sound("levelup.wav")
        # self.gameover_sound = get_sound("gameover.wav")
        # self.linescleared_sound = get_sound("linecleared.wav")
        # self.highscorebeaten_sound = get_sound("highscorebeaten.wav")


    def set_tetrominoes(self):
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        #self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

    
    def hard_drop(self):
        amount = 0
        while self.request_movement('down'):
            amount += 1

        self.lock_tetromino()
        self.score += 10*amount

    def update(self, timepassed, action):
        # pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
        # unpressed = lambda key: event.type == pygame.KEYUP and event.key == key
        if len(action)>2:
            action = self.decode_action(action)
        events = pygame.event.get()
        for i in range(action[0]):
            self.request_rotation()
        if action[1] < 0:
            for i in range(-action[1]):
                self.request_movement('left')
                self.movement_keys['left'] = 0
            self.movement_keys_timer = (-self.movement_keys_speed)*2
        elif action[1] > 0:
            for i in range(action[1]):
                self.request_movement('right')
                self.movement_keys['right'] = 0
            self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.hard_drop()
            
            # self.movement_keys_timer = (-self.movement_keys_speed)*2

        
        # elif action == 1:
        #     self.request_rotation()
        # elif action == 2:
        #     self.request_movement('left')
        #     self.movement_keys['left'] = 1
        #     self.movement_keys['left'] = 0
        #     self.movement_keys_timer = (-self.movement_keys_speed)*2
        # elif action == 3:
        #     self.request_movement('right')
        #     self.movement_keys['right'] = 1
        #     self.movement_keys['right'] = 0
        #     self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.downwards_speed = self.base_downwards_speed ** (1 + 1/10.)

        self.downwards_timer += timepassed
        downwards_speed = self.downwards_speed
        
        # if self.downwards_timer > downwards_speed:
        #     if not self.request_movement('down'):
        #         self.lock_tetromino()
        #     self.downwards_timer %= downwards_speed



        # if any(self.movement_keys.values()):
        #     self.movement_keys_timer += timepassed
        # if self.movement_keys_timer > self.movement_keys_speed:
        #     result = self.request_movement('right' if self.movement_keys['right'] else 'left')
        #     self.movement_keys_timer %= self.movement_keys_speed

        #with_shadow = self.place_shadow()

        try:
            with_tetromino = self.blend(self.rotated(), allow_failure=False) #, matrix=with_shadow)
            # if type(with_tetromino) == bool:
            #     self.prepare_and_execute_gameover()
            #     return
        except BrokenMatrixException:
            self.prepare_and_execute_gameover()
            return

        for y in range(self.size['height']):
            for x in range(self.size['width']):

                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x*self.blocksize, (y*self.blocksize - 2*self.blocksize), self.blocksize, self.blocksize)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)
        image_data = pygame.surfarray.array3d(self.surface)
        return image_data

    def encode_action(self, action): # [2, -1] -> [0...1...0]
        a = np.zeros(len(self.action_space))
        loc = self.action_space.index(action)
        a[loc] = 1
        return a

    def decode_action(self, action): # [0...1...0] -> [2, -1]
        loc = list(action).index(1)
        return self.action_space[loc]

                    
    def prepare_and_execute_gameover(self, playsound=False):
        if playsound:
            self.gameover_sound.play()
        write_score(self.score)
        self.gameover = True


    def fits_in_matrix(self, shape, position):
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]: # outside matrix
                    return False

        return position
                    

    def request_rotation(self):
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ Thats how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            return self.tetromino_rotation
        else:
            return False
            
    def request_movement(self, direction):
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            self.tetromino_position = (posY, posX-1)
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)):
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        colors = {'blue':   (27, 34, 224),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}


        if shadow:
            end = [40] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((self.blocksize, self.blocksize), pygame.SRCALPHA, 32)
        border.fill(map(lambda c: c*0.5, colors[color]) + end)

        borderwidth = 2

        box = Surface((self.blocksize-borderwidth*2, self.blocksize-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color]) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))


        return border

    def lock_tetromino(self):

        self.matrix = self.blend()

        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        if lines_cleared:
            # if lines_cleared >= 4:
            #     self.linescleared_sound.play()
            self.score += 100 * (lines_cleared**2) * self.combo

            # if not self.played_highscorebeaten_sound and self.score > self.highscore:
            #     if self.highscore != 0:
            #         self.highscorebeaten_sound.play()
            #     self.played_highscorebeaten_sound = True

        if self.lines >= self.level*10:
            # self.levelup_sound.play()
            self.level += 1

        self.combo = self.combo + 1 if lines_cleared else 1

        self.set_tetrominoes()

    def remove_lines(self):
        lines = []
        for y in range(self.size['height']):
            line = (y, [])
            for x in range(self.size['width']):
                # if type(self.matrix) == bool:
                #     self.prepare_and_execute_gameover()
                #     return len(lines)
                # else:
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == self.size['width']:
                lines.append(y)

        for line in sorted(lines):
            for x in range(self.size['width']):
                self.matrix[(line,x)] = None
            for y in range(0, line+1)[::-1]:
                for x in range(self.size['width']):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)

        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, block=None, allow_failure=True, shadow=False):
        # if type(self.matrix) == bool:
        #     return False
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        # self.blendrecord=[]
        # for x in range(self.size['width']):
        #     for y in range(self.size['height']):
        #         if copy[(y,x)] != None:
        #             self.blendrecord.append([y,x])
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if ((copy.get((y, x), False) is False and shape[y-posY][x-posX]) # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    (copy.get((y,x)) and shape[y-posY][x-posX])): 
                    if allow_failure:
                        return False
                    else:
                        raise BrokenMatrixException("Tried to blend a broken matrix. This should mean game over, if you see this it is certainly a bug. (or you are developing)")
                elif shape[y-posY][x-posX]:
                    copy[(y,x)] = ('block', self.tetromino_block if block is None else block)
                # elif shape[y-posY][x-posX] and shadow:
                #     copy[(y,x)] = ('shadow', block)

        return copy

##############################################################################
##############################################################################
    def Holes(self, fake_matrix):
        count = 0
        # start = len(self.current_tetromino.shape) + 1
        for x in range(self.size['width']):
            block = False
            for y in range(self.size['height']):
                if fake_matrix[(y,x)] != None:
                    block = True
                elif (fake_matrix[(y,x)] == None) & (block):
                    count += 1
        return count

    def columnHeight(self, fake_matrix, column):
        h = 0
        # start = len(self.current_tetromino.shape) + 1
        for h in range(self.size['height']):
            if fake_matrix[(h,column)] != None:
                return self.size['height'] - h
        return 0

    def aggregateHeight(self, fake_matrix):
        total = 0
        test = []
        for x in range(self.size['width']):
            total += self.columnHeight(fake_matrix, x)
            test.append(self.columnHeight(fake_matrix, x))
        return total,test

    def bumpness(self, fake_matrix):
        total = 0
        for x in range(self.size['width'] - 1):
            total += abs(self.columnHeight(fake_matrix, x) - self.columnHeight(fake_matrix, x+1))
        return total

    def clines(self, fake_matrix):
        lines = 0
        for y in range(self.size['height']):
            l = True
            for x in range(self.size['width']):
                if fake_matrix[(y,x)] == None:
                    l = False
                    break
            if l == True:
                lines += 1
        return lines

    def choose_action(self):
        bestscore = -float('inf')
        action = [0, 0]
        for rot in range(4):
            shape = self.rotated(rotation=rot)
            self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
            x = self.tetromino_position[1]
            i = 0
            while self.fits_in_matrix(shape, (0,x-i)) and self.blend(shape, (0,x-i)):
                x0 = x-i
                for y in range(self.size['height']):
                    if self.blend(shape,position=(y+1, x0)) == False:
                        break
                self.tetromino_position = (y, x0)
                fake_matrix = self.blend(shape)
                score = - 0.510066 * self.aggregateHeight(fake_matrix)[0] +  0.760666 * self.clines(fake_matrix) -  0.35663 * self.Holes(fake_matrix) -  0.184483 * self.bumpness(fake_matrix)
                record = [shape,score,(rot,x0),self.tetromino_position,self.aggregateHeight(fake_matrix),self.clines(fake_matrix),self.Holes(fake_matrix),self.bumpness(fake_matrix)]
                # print record
                if score > bestscore:
                    bestscore = score
                    action = [rot,-i]
                    ma = fake_matrix
                i += 1
                self.tetromino_position = (0,x0)

            j = 0
            while self.fits_in_matrix(shape, (0,x+j)) and self.blend(shape, (0,x+j)):
                x0 = x+j
                for y in range(self.size['height']):
                    if self.blend(shape,position=(y+1, x0)) == False:
                        break
                self.tetromino_position = (y, x0)
                # print self.blendrecord
                fake_matrix = self.blend(shape,self.tetromino_position)
                score = - 0.510066 * self.aggregateHeight(fake_matrix)[0] +  0.760666 * self.clines(fake_matrix) -  0.35663 * self.Holes(fake_matrix) -  0.184483 * self.bumpness(fake_matrix)
                record = [shape,score,(rot,x0),self.tetromino_position,self.aggregateHeight(fake_matrix),self.clines(fake_matrix),self.Holes(fake_matrix),self.bumpness(fake_matrix)]
                # print record
                # print self.aggregateHeight(fake_matrix),self.clines(fake_matrix),self.Holes(fake_matrix),self.bumpness(fake_matrix)
                if score > bestscore:
                    bestscore = score
                    action = [rot,j]
                    ma = fake_matrix

                j += 1
                self.tetromino_position = (0,x0)
        
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)

        # print record
        return action


##############################################################################
##############################################################################

    def construct_surface_of_next_tetromino(self):
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*self.blocksize, len(shape)*self.blocksize), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*self.blocksize, y*self.blocksize))
        return surf

class Game(object):
    def main(self, screen):
        clock = pygame.time.Clock()
        background = Surface(screen.get_size())
        state_batch = []

        #background.blit(construct_nightmare(background.get_size()), (0,0))

        self.matris = Matris()
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        avg_lines=0
        n_games = 1
        while 1:
            dt = clock.tick(45)
            
            # action = [0,-3]
            # events = pygame.event.get()
            # for event in events:
            #     pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
            #     if pressed(pygame.K_p):
            action = self.matris.choose_action()
            state = self.matris.update((dt / 1000.), action)

            # state_batch.append(state)
            if self.matris.gameover:
                avg_lines = avg_lines + (self.matris.lines-avg_lines)/n_games
                print avg_lines,n_games
                n_games += 1
                self.matris = Matris()
                # while True:
                #     events = pygame.event.get()
                #     for event in events:
                #         pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
                #         if pressed(pygame.K_p):
                #             return 


            # tricky_centerx = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

            background.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
            background.blit(self.matris.surface, (MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH))

            # nextts = self.next_tetromino_surf(self.matris.surface_of_next_tetromino)
            # background.blit(nextts, nextts.get_rect(top=MATRIS_OFFSET, centerx=tricky_centerx))

            # infos = self.info_surf()
            # background.blit(infos, infos.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=tricky_centerx))


            screen.blit(background, (0, 0))

            pygame.display.flip()

    

    # def info_surf(self):

    #     textcolor = (255, 255, 255)
    #     font = pygame.font.Font(None, 30)
    #     width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

    #     def renderpair(text, val):
    #         text = font.render(text, True, textcolor)
    #         val = font.render(str(val), True, textcolor)

    #         surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

    #         surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
    #         surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
    #         return surf

    #     scoresurf = renderpair("Score", self.matris.score)
    #     levelsurf = renderpair("Level", self.matris.level)
    #     linessurf = renderpair("Lines", self.matris.lines)
    #     combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

    #     height = 20 + (levelsurf.get_rect().height + 
    #                    scoresurf.get_rect().height +
    #                    linessurf.get_rect().height + 
    #                    combosurf.get_rect().height )

    #     area = Surface((width, height))
    #     area.fill(BORDERCOLOR)
    #     area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))

    #     area.blit(levelsurf, (0,0))
    #     area.blit(scoresurf, (0, levelsurf.get_rect().height))
    #     area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
    #     area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

    #     return area

    # def next_tetromino_surf(self, tetromino_surf):
    #     area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
    #     area.fill(BORDERCOLOR)
    #     area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

    #     areasize = area.get_size()[0]
    #     tetromino_surf_size = tetromino_surf.get_size()[0]
    #     # ^^ I'm assuming width and height are the same

    #     center = areasize/2 - tetromino_surf_size/2
    #     area.blit(tetromino_surf, (center, center))

    #     return area

def construct_nightmare(size):
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in xrange(0, len(arr), boxsize):
        for y in xrange(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in xrange(x, x+(boxsize - bordersize)):
                for LY in xrange(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # screen = pygame.Surface((WIDTH,HEIGHT))
    # pygame.display.set_caption("MaTris")
    Game().main(screen)
