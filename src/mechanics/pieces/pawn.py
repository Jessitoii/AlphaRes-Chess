# pawn.py

# Import sys and os.path modules
import sys
from os.path import abspath, dirname

# Get the directory where the file is located
currentDir = dirname(abspath(__file__))

# Get the parent directory
parentDir = dirname(currentDir)

# Add the parent directory to the sys.path list
sys.path.append(parentDir)

# Import the Piece class from the piece module
from piece import Piece

# Inherit from the Piece class to create the Pawn class
class Pawn(Piece):
    def __init__(self, color, coordinate, isAlive):
        # Call the __init__ method of the Piece class
        super().__init__(color, coordinate, isAlive)
        
        self.type = "Pawn"
        # Determine the icon path for the piece
        self.iconPath = "./assets/"

        # Set the icon path based on the color of the piece
        if self.color == "white":
            self.iconPath += "whitePawn.png"
        else:
            self.iconPath += "blackPawn.png" 

    def moveableCoors(self, board):
        # Get the current coordinates, color, and position of the pawn
        coor = self.coordinate
        color = self.color
        x = coor[0]
        y = coor[1]
        xList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        moveableCoordinates = []  # It returns like this: ["a1", "a2", "a3"]

        # Check possible moves based on pawn color
        if color == "white":
            if y == "2":
                for i in [3, 4]:
                    if board[x + str(i)]["piece"] == None:
                        moveableCoordinates.append(x + str(i))
                    else:
                        break
            else:
                if board[x + str(int(y) + 1)]["piece"] == None:
                    a = int(y) + 1
                    moveableCoordinates.append(x + str(a))

            for i in [1, -1]:
                xIndex = xList.index(x)
                a = int(y) + 1
                if xIndex + i >= 0 and xIndex + i < 8:
                    moveableCoordinate = xList[xIndex + i] + str(a)
                    if (
                        board[moveableCoordinate]["piece"] != None
                        and board[moveableCoordinate]["piece"].color != self.color
                    ):
                        moveableCoordinates.append(moveableCoordinate)

        else:
            if y == "7":
                for i in [6, 5]:
                    if board[x + str(i)]["piece"] == None:
                        moveableCoordinates.append(x + str(i))
                    else:
                        break
            else:
                if board[x + str(int(y) - 1)]["piece"] == None:
                    a = int(y) - 1
                    moveableCoordinates.append(x + str(a))

            for i in [1, -1]:
                xIndex = xList.index(x)
                a = int(y) - 1
                if xIndex + i >= 0 and xIndex + i < 8:
                    moveableCoordinate = xList[xIndex + i] + str(a)
                    if (
                        board[moveableCoordinate]["piece"] != None
                        and board[moveableCoordinate]["piece"].color != self.color
                    ):
                        moveableCoordinates.append(moveableCoordinate)

        return moveableCoordinates
