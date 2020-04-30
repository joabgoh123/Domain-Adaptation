# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from resnet import SourceModel
from DeepCoral import coral
from DANN import dann
from DDC import ddc
import math

file_path = None

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(942, 627)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selectImage = QtWidgets.QPushButton(self.centralwidget)
        self.selectImage.setGeometry(QtCore.QRect(60, 440, 461, 71))
        self.selectImage.setObjectName("selectImage")
        self.imageGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.imageGroupBox.setGeometry(QtCore.QRect(60, 40, 461, 391))
        self.imageGroupBox.setObjectName("imageGroupBox")
        self.Image = QtWidgets.QLabel(self.imageGroupBox)
        self.Image.setGeometry(QtCore.QRect(10, 60, 431, 311))
        self.Image.setText("")
        self.Image.setScaledContents(True)
        self.Image.setObjectName("Image")
        self.label = QtWidgets.QLabel(self.imageGroupBox)
        self.label.setGeometry(QtCore.QRect(200, 10, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.chooseModelGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.chooseModelGroupBox.setGeometry(QtCore.QRect(620, 40, 271, 211))
        self.chooseModelGroupBox.setObjectName("chooseModelGroupBox")
        self.label_2 = QtWidgets.QLabel(self.chooseModelGroupBox)
        self.label_2.setGeometry(QtCore.QRect(70, 10, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.chooseModelComboBox = QtWidgets.QComboBox(self.chooseModelGroupBox)
        self.chooseModelComboBox.setGeometry(QtCore.QRect(30, 70, 221, 61))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.chooseModelComboBox.setFont(font)
        self.chooseModelComboBox.setObjectName("chooseModelComboBox")
        self.chooseModelComboBox.addItem("")
        self.chooseModelComboBox.addItem("")
        self.chooseModelComboBox.addItem("")
        self.chooseModelComboBox.addItem("")
        self.evaluateButton = QtWidgets.QPushButton(self.chooseModelGroupBox)
        self.evaluateButton.setGeometry(QtCore.QRect(30, 150, 221, 51))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(11)
        self.evaluateButton.setFont(font)
        self.evaluateButton.setObjectName("evaluateButton")
        self.resultGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.resultGroupBox.setGeometry(QtCore.QRect(630, 290, 261, 171))
        self.resultGroupBox.setObjectName("resultGroupBox")
        self.resultHeader = QtWidgets.QLabel(self.resultGroupBox)
        self.resultHeader.setGeometry(QtCore.QRect(90, 20, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(18)
        self.resultHeader.setFont(font)
        self.resultHeader.setObjectName("resultHeader")
        self.resultLabel = QtWidgets.QLabel(self.resultGroupBox)
        self.resultLabel.setGeometry(QtCore.QRect(20, 60, 221, 91))
        self.resultLabel.setObjectName("resultLabel")
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        self.resultLabel.setFont(font)

        # self.retrainButton = QtWidgets.QPushButton(self.centralwidget)
        # self.retrainButton.setGeometry(QtCore.QRect(630, 490, 261, 61))
        font = QtGui.QFont()
        font.setFamily("Trebuchet MS")
        font.setPointSize(14)
        # self.retrainButton.setFont(font)
        # self.retrainButton.setObjectName("retrainButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 942, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selectImage.setText(_translate("MainWindow", "Select Image"))
        #Select file path to image
        self.selectImage.clicked.connect(self.selectImage_handler)
        self.imageGroupBox.setTitle(_translate("MainWindow", ""))
        self.label.setText(_translate("MainWindow", "Image"))
        self.chooseModelGroupBox.setTitle(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "Choose Model"))
        self.chooseModelComboBox.setItemText(0, _translate("MainWindow", "Source Model"))
        self.chooseModelComboBox.setItemText(1, _translate("MainWindow", "Deep Coral"))
        self.chooseModelComboBox.setItemText(2, _translate("MainWindow", "DANN"))
        self.chooseModelComboBox.setItemText(3, _translate("MainWindow", "DDC"))
        self.evaluateButton.setText(_translate("MainWindow", "Evaluate"))
        #Check which model selected and evaluate
        self.evaluateButton.clicked.connect(self.evaluate_handler)
        self.resultGroupBox.setTitle(_translate("MainWindow", ""))
        self.resultHeader.setText(_translate("MainWindow", "Results"))
        self.resultLabel.setText(_translate("MainWindow", ""))
        # self.retrainButton.setText(_translate("MainWindow", "Retrain"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))

    def selectImage_handler(self):
        filename = QFileDialog.getOpenFileName()
        global file_path
        file_path = filename[0]
        self.set_image()

    def set_image(self):
        self.Image.setPixmap(QtGui.QPixmap(file_path))

    def model_selection_handler(self):
        current_selection = self.chooseModelComboBox.currentText()
        return current_selection
    
    def evaluate_handler(self):
        global file_path
        model_selected = self.model_selection_handler()
        self.show_evaluating()
        
        if file_path == None:
            self.show_no_image()
        else:
        
            if model_selected == "Source Model":
                model = SourceModel('cuda')
                prediction, prob = model.predict(file_path)
                results = [prediction, prob]
                self.update_results(results)

            elif model_selected == "Deep Coral":
                model = coral.CoralModel('cuda')
                prediction, prob = model.predict(file_path)
                results = [prediction, prob]
                self.update_results(results)

            elif model_selected == "DANN":
                model = dann.DannModel('cuda')
                prediction, prob = model.predict(file_path)
                results = [prediction, prob]
                self.update_results(results)

            elif model_selected == "DDC":
                model = ddc.DDCModel('cuda')
                prediction, prob = model.predict(file_path)
                results = [prediction, prob]
                self.update_results(results)

    def show_no_image(self):
        self.resultLabel.setText("No Image Selected ...")

    def show_evaluating(self):
        self.resultLabel.setText("Evaluating ...")

    def update_results(self, results):
        result_fields = ["Classification", "Confidence"]
        str_result = ""
        results[1] = round(results[1] * 100, 2)
        # for i in range(len(results)):
        #     str_result = str_result + " " + result_fields[i] + ": " + str(results[i]) +  "\n" 
        
        str_result = str_result + " " + result_fields[0] + ": " + results[0] + "\n"
        str_result = str_result + " " + result_fields[1] + ": " + str(results[1]) + "%" + "\n"
        self.resultLabel.setText(str_result)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
