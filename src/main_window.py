# main_window.py
import sys
from ui_components.board import Board
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QIcon, QFont

# QMainWindow subclass for the main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the application icon
        appIcon = QIcon("/assets/blackKing.png")

        # Create the chessboard (Board widget)
        self.board = Board()
        self.board.setStyleSheet("")  # Adjust stylesheet if needed
        
        # Set up the main layout with the chessboard
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.board)
        self.layout.setAlignment(Qt.AlignCenter)

        # Set the main widget as the central widget of the QMainWindow
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)

        # Set window properties
        self.setWindowTitle("Chess App!")
        self.setWindowIcon(appIcon)

# Main entry point of the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
