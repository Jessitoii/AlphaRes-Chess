from PyQt5.QtWidgets import QLabel, QWidget, QGridLayout
from PyQt5.QtGui import QFont

#killed[0] holds white pieces, killed[1] holds black pieces
class RoundWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel("Round : White")
        self.label.setStyleSheet("color: brown;")
        
        

        font = QFont()
        font.setPixelSize(11)
        font.setBold(True)
        
        self.label.setFont(font)

        self.layout = QGridLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)