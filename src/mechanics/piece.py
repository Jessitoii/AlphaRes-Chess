class Piece:
    def __init__(self, color, coordinate, isAlive):
        # Initialize basic attributes of a chess piece
        self.color = color
        self.coordinate = coordinate
        self.isAlive = isAlive
    
    def checkDiagonal(self, board):
        """
        Calculate diagonal moveable coordinates for the piece.
        """
        coor = self.coordinate
        x = coor[0]
        y = coor[1]
        xList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        yList = ["1", "2", "3", "4", "5", "6", "7", "8"]
        xIndex = xList.index(x)
        yIndex = yList.index(y)
        moveableCoordinates = []

        # i changes the derivative of the row
        # j changes the derivative of the column
        for i in [1, -1]:
            for j in [1, -1]:
                col = yIndex
                if i == 1:
                    stop = 7
                    step = +1
                else:
                    step = -1
                    stop = 0
                for row in range(xIndex, stop, step):
                    if col + j < 8 and col + j >= 0: 
                        moveableCoordinate = xList[row + i] + yList[col + j]
                        if board[moveableCoordinate]["piece"] == None:
                            moveableCoordinates.append(moveableCoordinate)
                        else:
                            if board[moveableCoordinate]["piece"].color != self.color:
                                moveableCoordinates.append(moveableCoordinate)
                            break
                    col += j
        return moveableCoordinates
    
    def checkOrthogonal(self, board):
        """
        Calculate orthogonal moveable coordinates for the piece.
        """
        coor = self.coordinate
        x = coor[0]
        y = coor[1]
        xList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        yList = ["1", "2", "3", "4", "5", "6", "7", "8"]
        moveableCoordinates = []    

        # Check to the right of the piece
        for i in range(xList.index(x) + 1, 8):
            if board[xList[i] + y]["piece"] == None:
                moveableCoordinates.append(xList[i] + y)
            else:
                if board[xList[i] + y]["piece"].color != self.color:
                    moveableCoordinates.append(xList[i] + y)
                break
        
        # Check to the left of the piece
        for i in range(xList.index(x) - 1, -1, -1):
            if board[xList[i] + y]["piece"] == None:
                moveableCoordinates.append(xList[i] + y)
            else:
                if board[xList[i] + y]["piece"].color != self.color:
                    moveableCoordinates.append(xList[i] + y)
                break

        # Check above the piece
        for i in range(yList.index(y) + 1, 8):
            if board[x + yList[i]]["piece"] == None:
                moveableCoordinates.append(x + yList[i])
            else:
                if board[x + yList[i]]["piece"].color != self.color:
                    moveableCoordinates.append(x + yList[i])
                break

        # Check below the piece
        for i in range(yList.index(y) - 1, -1, -1):
            if board[x + yList[i]]["piece"] == None:
                moveableCoordinates.append(x + yList[i])
            else:
                if board[x + yList[i]]["piece"].color != self.color:
                    moveableCoordinates.append(x + yList[i])
                break

        return moveableCoordinates
