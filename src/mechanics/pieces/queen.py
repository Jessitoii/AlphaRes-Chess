#queen.py
import sys
from os.path import abspath, dirname

# Get the current and parent directories of the file
currentDir = dirname(abspath(__file__))
parentDir = dirname(currentDir)
sys.path.append(currentDir)

from piece import Piece

class Queen(Piece):
    def __init__(self, color, coordinate, isAlive):
        # Call the superclass constructor with relevant information
        super().__init__(color, coordinate, isAlive)
        
        self.type = "Queen"
        # Set the path for the queen's icon based on its color
        self.iconPath = "./assets/"
        if self.color == "white":
            self.iconPath += "whiteQueen.png"
        else:
            self.iconPath += "blackQueen.png"

    def moveableCoors(self, board):
        # Initialize an empty list to store movable coordinates
        moveableCoordinates = []
        
        # Get diagonal and orthogonal coordinates using superclass methods
        diagonalCoordinates = super().checkDiagonal(board)
        orthogonalCoordinates = super().checkOrthogonal(board)

        # Combine diagonal and orthogonal coordinates
        moveableCoordinates.extend(diagonalCoordinates)
        moveableCoordinates.extend(orthogonalCoordinates)
        
        # Return the final list of movable coordinates
        return moveableCoordinates
