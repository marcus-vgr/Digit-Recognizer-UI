from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
#plt.style.use('dark_background')

from PyQt5 import QtCore, QtWidgets, QtGui, Qt
import sys

from numpy import argmax
from predictions import DigitRecognizer


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(512, 512) # Create black window of 512x512 pixels
        pixmap.fill(color=QtGui.QColor('black')) # We can change colors..
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None  # create variables that give the coordinates of the click
        self.coordinates_img = [] # create list to save coordinates
        self.pen_color = QtGui.QColor('blue') # set pen color to blue

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        # Draw in the black window
        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(15)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()
        # Save the coordinates
        self.coordinates_img.append([self.last_x, self.last_y])

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def clearWindow(self):
        pixmap = QtGui.QPixmap(512, 512) # Create black window of 512x512 pixels
        self.setPixmap(pixmap)
        self.coordinates_img = []
        


class PredictionMplCanvas(MplCanvas):


    def compute_initial_figure(self):
        self.axes.set_ylabel('Probability', fontsize=15)
        self.axes.set_xlabel('Number', fontsize=15)
        self.axes.set_xticks(ticks=range(10))
        self.axes.set_ylim(0,1)
        self.axes.bar(range(10), 10*[0.1], color='blue')            


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # Set TIMER to update plot
        timer = QtCore.QTimer(self)
        timer.setInterval(4000) # Update every 5 seconds
        timer.timeout.connect(self.update_plot)
        timer.start() 

        self.setFixedSize(1040, 514)
        self.main_widget = QtWidgets.QWidget(self)
        self.w = QtWidgets.QWidget()
        self.l = QtWidgets.QGridLayout()
        self.w.setLayout(self.l)

        self.canvas = Canvas()
        self.mpl_canvas = PredictionMplCanvas()
        self.l.addWidget(self.canvas, 0,0)
        self.l.addWidget(self.mpl_canvas, 0,1)
        
        self.setCentralWidget(self.w)

    def Reset(self):
        self.l.removeWidget(self.canvas)
        self.canvas = Canvas()
        self.l.addWidget(self.canvas, 0,0)
        
    def update_plot(self):
        coordinates_img = self.canvas.coordinates_img
        self.mpl_canvas.axes.cla()
        self.mpl_canvas.axes.set_ylabel('Probability', fontsize=15)
        self.mpl_canvas.axes.set_xlabel('Number', fontsize=15)
        self.mpl_canvas.axes.set_xticks(ticks=range(10))
        self.mpl_canvas.axes.set_ylim(0,1)

        if len(coordinates_img) == 0:            
            self.mpl_canvas.axes.bar(range(10), 10*[0.1], color='blue')            
        else:
            pred = DigitRecognizer(coordinates_img)
            self.mpl_canvas.axes.bar(range(10), pred.flatten(), color='blue')
            self.mpl_canvas.axes.set_title(f'This is a {argmax(pred)}!', fontsize=20)

        self.mpl_canvas.draw()
        self.Reset()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
del window, app
