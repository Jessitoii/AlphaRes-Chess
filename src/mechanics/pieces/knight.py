# knight.py

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

# Inherit from the Piece class to create the Knight class
class Knight(Piece):
    def __init__(self, color, coordinate, isAlive):
        # Call the __init__ method of the Piece class
        super().__init__(color, coordinate, isAlive)

        self.type = "Knight"
        # Determine the icon path for the piece
        self.iconPath = "./assets/"

        # Set the icon path based on the color of the piece
        if self.color == "white":
            self.iconPath += "whiteKnight.png"
        else:
            self.iconPath += "blackKnight.png"

    def moveableCoors(self, board):
        # Get the current coordinates of the knight
        coor = self.coordinate
        x = coor[0]
        y = coor[1]

        # Define lists for the x and y coordinates
        xList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        yList = ["1", "2", "3", "4", "5", "6", "7", "8"]

        # List to store movable coordinates
        moveableCoordinates = []

        # Iterate over possible knight movements
        for i in [-2, -1, 1, 2]:
            for j in [-2, -1, 1, 2]:
                # Check if the movement is L-shaped
                if abs(i) != abs(j):
                    xIndex = xList.index(x)
                    yIndex = yList.index(y)

                    # Check if the new coordinates are within the board boundaries
                    if xIndex + i < 8 and xIndex + i >= 0 and yIndex + j < 8 and yIndex + j >= 0:
                        # Calculate the new coordinates
                        moveableCoordinate = xList[xIndex + i] + yList[yIndex + j]

                        # Check if the target position is empty or has an opponent's piece
                        if board[moveableCoordinate]["piece"] == None:
                            moveableCoordinates.append(moveableCoordinate)
                        else:
                            if board[moveableCoordinate]["piece"].color != self.color:
                                moveableCoordinates.append(moveableCoordinate)
                else:
                    continue

        return moveableCoordinates
