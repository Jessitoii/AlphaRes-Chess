# bishop.py

# Import sys and os.path modules
import sys
from os.path import abspath, dirname

# Get the directory where the file is located
currentDir = dirname(abspath(__file__))

# Get the parent directory
parentDir = dirname(currentDir)

# Add the parent directory to the sys.path list
sys.path.append(currentDir)

# Import the Piece class from the piece module
from piece import Piece

# Inherit from the Piece class to create the Bishop class
class Bishop(Piece):
    def __init__(self, color, coordinate, isAlive):
        # Call the __init__ method of the Piece class
        super().__init__(color, coordinate, isAlive)
        self.type = "Bishop"
        # Determine the icon path for the piece
        self.iconPath = "./assets/"

        # Set the icon path based on the color of the piece
        if self.color == "white":
            self.iconPath += "whiteBishop.png"
        else:
            self.iconPath += "blackBishop.png"

    def moveableCoors(self, board):
        # Use the checkDiagonal method to determine the movable coordinates
        return super().checkDiagonal(board)
