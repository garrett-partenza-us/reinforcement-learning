import numpy as np

# Four rooms environment class copied and pasted from my ex0 submission
class Environment:
    
    def __init__(self, rows, cols, walls):
        
        self.rows = rows
        self.cols = cols
        self.walls = walls
        self.x = 0
        self.y = 0
        self.x_prev = 0
        self.y_prev = 0
        self.bank = 0
        
    def draw(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if ((self.rows-row-1, col)) in self.walls:
                    print("{}   ".format("1"), end="")
                elif ((self.rows-row-1, col)) == (self.rows-1, self.cols-1):
                    print("{}   ".format("F"), end="")
                elif ((self.rows-row-1, col)) == (self.x, self.y):
                    print("{}   ".format("P"), end="")
                else:
                    print("{}   ".format("0"), end="")
            print("\n")                
                
    def step(self, action):
        self.x_prev=self.x
        self.y_prev=self.y
        if action == "LEFT":
            self.y=self.y-1
            self.correct()
            return self.reward()
        elif action == "RIGHT":
            self.y=self.y+1
            self.correct()
            return self.reward()
        elif action == "UP":
            self.x=self.x+1
            self.correct()
            return self.reward()
        elif action == "DOWN":
            self.x=self.x-1
            self.correct()
            return self.reward()
        
    def reward(self):
        if ((self.x, self.y)) == (self.rows-1, self.cols-1):
            self.bank+=1
            return (10, True)
        else:
            return (0, False)
        
    def correct(self):
        if self.x<0:
            self.x=self.x_prev
        elif self.x==self.rows:
            self.x=self.x_prev
        elif self.y<0:
            self.y=self.y_prev
        elif self.y==self.cols:
            self.y=self.y_prev
        elif ((self.x, self.y)) in self.walls:
            self.x=self.x_prev
            self.y=self.y_prev
            
    def restart(self):
        self.x = 0
        self.y = 0
        self.x_prev = 0
        self.y_prev = 0
        
    def reset(self):
        self.restart()
        self.bank = 0
        
    def loc(self):
        return (self.x, self.y)