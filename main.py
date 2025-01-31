import os
import sys
import PyQt5
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap

from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout

from PyQt5 import QtCore # , QtGui, QtWidgets

import csv

PATH = 'C:/Users/ykest/Desktop/EngSci Year 3/ECE324/project_test'

#* Grabs *.jpg files in the directory PATH
def getImgFiles(PATH):
    directory = os.listdir(PATH)
    imgs = []
    for f in directory:
        name, ext = os.path.splitext(f)
        if ext == '.jpg':
            imgs.append(f)
    return imgs

class LabelApp(QWidget):
    
    def __init__(self,parent = None):
        super(LabelApp,self).__init__(parent)
        
        self.setWindowTitle('Label Triplets')
        self.setGeometry(0, 0, 600, 400)
        self.setFixedWidth(1200)
        self.output = []
        
        layout1 = QHBoxLayout()
        self.files = getImgFiles(PATH)
        for img_name in self.files:
            image = QLabel()
            pixmap = QPixmap(os.path.join(PATH, img_name))
            image.resize(200, 200)
            image.setPixmap(pixmap.scaled(image.size(), QtCore.Qt.KeepAspectRatio))
            layout1.addWidget(image)
        
        layout2 = QVBoxLayout()
        btnLabel = QLabel(
        "Select the expression that is the most different from the others:\n"
        )
        layout2.addWidget(btnLabel)

        layout3 = QHBoxLayout()

        layout4 = QVBoxLayout()
        self.b1 = QRadioButton("Image 1")
        self.b1.toggled.connect(lambda:self.btnstate(self.b1))
        layout4.addWidget(self.b1)

        self.b2 = QRadioButton("Image 2")
        self.b2.toggled.connect(lambda:self.btnstate(self.b2))
        layout4.addWidget(self.b2)

        self.b3 = QRadioButton("Image 3")
        self.b3.toggled.connect(lambda:self.btnstate(self.b3))
        layout4.addWidget(self.b3)
        layout3.addLayout(layout4)

        self.save = QPushButton("Save")
        self.save.clicked.connect(lambda:self.register(self.b1,self.b2,self.b3))
        layout3.addWidget(self.save)

        layout2.addLayout(layout3)

        layout1.addLayout(layout2)
        self.setLayout(layout1)
    
    #* Callback for save button
    def register(self,*argv):
        self.out = None
        for button in argv:
            if button.isChecked() == True:
                self.out = button.text() + " SAVED"
                break
        if self.out != None:
            self.label = [self.files,self.out]  #! Esther: here is the saved variable
            # print(self.label)
        else:
            print("Select Image before saving!")
        self.output = self.output + [self.label]
        return self.output

    #* Callback for buttons 1 2 3
    def btnstate(self,b):
        if b.isChecked() == True:
            print(b.text() + " is selected")
        else:
            print(b.text() + " is deselected") 

    #* Trigger for keypress
    def keyPressEvent(self, event):
        #* Press number keys to select image
        if event.key() == QtCore.Qt.Key_1:
            print("Image 1")
            self.b1.toggle()
        elif event.key() == QtCore.Qt.Key_2:
            print("Image 2")
            self.b2.toggle()
        elif event.key() == QtCore.Qt.Key_3:
            print("Image 3")
            self.b3.toggle()
        
        #* Pressing either enter key
        elif event.key() == QtCore.Qt.Key_Enter or event.key() == QtCore.Qt.Key_Return:
            print("ENTER")
            self.save.click()
        event.accept()

def main():
    app = QApplication(sys.argv)
    demo = LabelApp()
    demo.show()
    app.exec_()
    # sys.exit(app.exec_())
    if os.path.exists(PATH + "/output.csv"):
        mode = "a+"
    else:
        mode = "w"
    with open(PATH + "/output.csv", mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)
        for i in demo.output:
            label = int(i[1].split("Image ")[1].split(" ")[0])
            writer.writerow([i[0][0], i[0][1], i[0][2], label])
    sys.exit()
    
if __name__ == "__main__":
    main()





#*****************************************************
#TODO: Stuff to remove ...



#? Textbox in GUI -> Use QLineEdit
#? Button in GUI -> QPushButton


    
    # window = QWidget()
    # window.setWindowTitle('PyQt5 App')
    # #* (x position,y position,width,height)
    # # window.setGeometry(100, 100, 280, 80)
    # # window.move(60, 15)
    # # helloMsg = QLabel('<h1>Hello World!</h1>', parent=window)
    # # helloMsg.move(60, 15)
    # # helloMsg.show()

    # hlay = QHBoxLayout()
    # for img_name in ('26_K_1.jpg',
    #                  '26_K_2.jpg',
    #                  '26_K_3.jpg'):
    #     label = QLabel()
    #     pixmap = QPixmap(os.path.join(PATH, img_name))
    #     label.resize(250, 250)
    #     label.setPixmap(pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
    #     hlay.addWidget(label)

    

    # # label = QLabel('<h1>Hello World!</h1>', parent=window)
    # # pixmap = QPixmap(PATH)
    # # label.setPixmap(pixmap)

    # #button test
    # btnLabel = QLabel(
    #     "Identify which two of the three images shown have more similar facial"
    #     " expressions. Select the odd one out using the buttons:"
    # )
    # btn1 = QRadioButton("1")
    # btn2 = QRadioButton("2")
    # btn3 = QRadioButton("3")

    # # radio = RadioButton(hlay)
    # hlay.addWidget(btnLabel)
    # hlay.addWidget(btn1)
    # hlay.addWidget(btn2)
    # hlay.addWidget(btn3)

    # window.setLayout(hlay)
    
    # window.show()