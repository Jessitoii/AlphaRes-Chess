# board.py
import sys
from os.path import abspath, dirname
# Add parent directory to the Python path
currentDir = dirname(abspath(__file__))
parentDir = dirname(currentDir)
sys.path.append(parentDir)
from mechanics import game
from PyQt5.QtWidgets import QPushButton, QGridLayout, QWidget, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

class Board(QWidget):
    squareSize = 100
    
    def __init__(self):
        super().__init__()
        self.Game = game.Game()
        self.squares = {}

        self.layout = QGridLayout()
    
        # Create the chessboard grid
        for row in self.Game.xCoordinates:
            for col in  self.Game.yCoordinates:
                coordinate = row + col
                rowIndex = self.Game.xCoordinates.index(row)
                colIndex = self.Game.yCoordinates.index(col)

                # Create a QPushButton for each square on the chessboard
                square = QPushButton(coordinate)
                square.setFixedSize(self.squareSize, self.squareSize)
                square.setStyleSheet('background-color : brown; border: none;') if((colIndex + rowIndex) % 2 == 0) else square.setStyleSheet('background-color : white; border: none;')
                square.setContentsMargins(0,0,0,0)

                # Add the QPushButton to the layout
                layoutIndex = [8 - colIndex , rowIndex]
                self.layout.addWidget(square, layoutIndex[0], layoutIndex[1])
                self.squares.update({coordinate : {
                    "button": square,
                    "function" : None
                } })

        # Set layout properties
        self.layout.setSpacing(0)
        self.setFixedSize(self.squareSize * 8, self.squareSize * 8)
        self.setLayout(self.layout)

        self.Game.computerMove()

        # Connect the click event and set piece icons
        self.clickEvent()
        self.setIcons()

    def clickEvent(self):
        # Connect click events for each square
        for row in self.Game.xCoordinates:
            for col in  self.Game.yCoordinates:
                coordinate = row + col
                square = self.squares[coordinate]
                gameSquare = self.Game.board[coordinate]
                
                if square["function"] != None:
                    square["button"].clicked.disconnect()
                    square["function"] = None           

                if gameSquare["piece"] != None :
                    if gameSquare["piece"].color == self.Game.round:
                        firstClickLambda = lambda _, coord=coordinate: self.firstClickFunc(coord)
                        square["button"].clicked.connect(firstClickLambda)
                        square["function"] = firstClickLambda
                    else:
                        if square["function"] != None:
                            square["button"].clicked.disconnect()
                            square["function"] = None        

    def firstClickFunc(self, coordinate):
        # Handle logic for the first click event
        self.clickEvent()
        for row in self.Game.xCoordinates:
            for col in  self.Game.yCoordinates:
                if self.squares[row + col]["button"].text() == "X":
                    self.squares[row + col]["button"].setText(row + col)

        coordinateList = self.Game.checkMoveableCoordinates(coordinate)
        piece = self.Game.board[coordinate]["piece"]
        
        for coor in coordinateList:
            button = self.squares[coor]["button"]
            button.setText("X")
                
            if self.squares[coor]["function"] != None:
                try:
                    button.disconnect(self.squares[coor]["function"])
                except:
                    button.disconnect()
            secondClickedlambda = lambda _, coord=coor, piece=piece, oldCoordinate=coordinate, coordList=coordinateList: self.secondClickFunc(coord, piece, oldCoordinate, coordList)
            button.clicked.connect(secondClickedlambda)
            self.squares[coor]["function"] = secondClickedlambda

    def secondClickFunc(self, coordinate, piece, oldCoordinate, coordinateList):
        # Handle logic for the second click event
        self.Game.move(coordinate, piece, oldCoordinate, coordinateList)
        self.Game.computer.pushMove(oldCoordinate + coordinate)
        
        for coor in coordinateList:
            self.squares[coor]["button"].setText(self.Game.board[coor]["coordinate"])
            piece.coordinate = coordinate
        
        # Update the UI
        self.clickEvent()
        self.setIcons()
        
        if self.Game.winner:
            self.clearLayout()
            label = QLabel("Winner : " + self.Game.winner)
            self.layout.addWidget(label)
        
        self.Game.computerMove()
        
        self.clickEvent()
        self.setIcons()
        
        if self.Game.winner:
            self.clearLayout()
            label = QLabel("Winner : " + self.Game.winner)
            self.layout.addWidget(label)
        
        
    def setIcons(self):
        # Set piece icons on the board
        for row in self.Game.xCoordinates:
            for col in  self.Game.yCoordinates:
                coordinate = row + col
                if self.Game.board[coordinate]["piece"] != None:
                    icon = QIcon(self.Game.board[coordinate]["piece"].iconPath)
                    self.squares[coordinate]["button"].setIcon(icon)
                    size = QSize(50, 50)
                    self.squares[coordinate]["button"].setIconSize(size)
                else:
                    self.squares[coordinate]["button"].setIcon(QIcon())

    def clearLayout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                self.clearLayout(item.layout())