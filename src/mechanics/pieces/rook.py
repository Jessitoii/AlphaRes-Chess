# rook.py
import sys
from os.path import abspath, dirname

# Get the current and parent directories of the file
currentDir = dirname(abspath(__file__))
parentDir = dirname(currentDir)
sys.path.append(parentDir)

from piece import Piece

class Rook(Piece):
    def __init__(self, color, coordinate, isAlive):
        # Call the superclass constructor with relevant information
        super().__init__(color, coordinate, isAlive)
        self.type = "Rook"
        
        # Set the path for the rook's icon based on its color
        self.iconPath = "./assets/"
        if self.color == "white":
            self.iconPath += "whiteRook.png"
        else:
            self.iconPath += "blackRook.png"

    def moveableCoors(self, board):
        # Utilize the superclass method to get orthogonal coordinates
        return super().checkOrthogonal(board)
