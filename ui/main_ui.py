# -*- coding: utf-8 -*-
# Copyright © 2021 Thong Duy Nguyen
# Email: thongngu@student.uef.vn

from PyQt5 import QtWidgets, QtCore, QtGui
from PIL.ImageQt import ImageQt, QImage
from PIL import Image
from PyQt5.QtCore import QBuffer
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QBrush, QPainter, QPen, QPixmap, QTextListFormat
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget, QHBoxLayout

import __init__
lib_path = __init__.get_libpath()
import config
import visualize
import io

import os
import sys
import warnings
import numpy as np
import cv2 as cv
from tifffile import TiffFile, TiffWriter
from tifffile.tifffile import NullContext

# This zoom function is referred from this solution: https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setFixedWidth(512)
        self.setFixedHeight(512)
        self.setStyleSheet("background-color: #C8C8C8; border: 0px;")

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Set size of the application window
        self.setGeometry(0, 0, 1280, 512)
        self.setWindowTitle("HSI Analyzer")
        self.initUI()

        # First initialization 
        self.list_paths = []
        self.list_mask = []
        self.list_gt_mask = []
        self.list_PIL_img = []
        self.list_PIL_mask = []
        self.list_PIL_gt = []
        self.gt_masks = []
        self.images = []
        self.curr_pth = ""
        self.current_index = -1
        self.capacity_value = 0
        self.show_mask = False
        self.is_segmented = False

    def initUI(self):
        # Create tab view
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300,200)
        self.setCentralWidget(self.tabs)
        self.tabs.setGeometry(QtCore.QRect(10, 10, 990, 700))
        self.tabs.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabBarAutoHide(False)
        # Add Tabs
        self.tabs.addTab(self.tab1, "Image Analysis")
        # self.tabs.addTab(self.tab2, "Live view")
        self.tabs.setObjectName("tab_view")
        self.tab1.setObjectName("image_analysis")
        # self.tab2.setObjectName("live_view")

        """Design features for tab 1"""
        # Create layout for tab1
        self.tab1.layout = QVBoxLayout()
        self.tab1.layout.setAlignment(QtCore.Qt.AlignCenter)

        # Create tool bar layout
        self.tool_bar = QWidget()
        self.tool_bar.layout = QtWidgets.QGridLayout()
        self.tool_bar.layout.setAlignment(QtCore.Qt.AlignLeft)

        # Create image display area layout
        self.original_image_viewer = PhotoViewer(self)
        self.segmented_image_viewer = PhotoViewer(self)
        self.image_display = QWidget()
        self.image_display.layout = QtWidgets.QGridLayout()
        self.image_display.layout.setAlignment(QtCore.Qt.AlignTop)

        # Add tool bar and display area to the layout of tab1
        self.tab1.layout.addLayout(self.tool_bar.layout)
        self.tab1.layout.addLayout(self.image_display.layout)
        self.tab1.setLayout(self.tab1.layout)

        # Design an "open directory" button
        self.open_dir_button = QPushButton()
        self.open_dir_button.setObjectName("open_dir_button")
        self.open_dir_button.setIcon(QtGui.QIcon(lib_path + "/icons/folder.png"))
        self.open_dir_button.setIconSize(QtCore.QSize(40, 40))
        self.open_dir_button.setToolTip("Open a directory")
        self.open_dir_button.setFixedWidth(100)
        self.open_dir_button.clicked.connect(self.openFiles)
        self.tool_bar.layout.addWidget(self.open_dir_button, 0,0, QtCore.Qt.AlignCenter)

        # Design an "open file" button
        self.open_file_button = QPushButton()
        self.open_file_button.setObjectName("open_file_button")
        self.open_file_button.setIcon(QtGui.QIcon(lib_path + "/icons/file.png"))
        self.open_file_button.setIconSize(QtCore.QSize(40, 40))
        self.open_file_button.setToolTip("Open an image file")
        self.open_file_button.setFixedWidth(100)
        self.open_file_button.clicked.connect(self.openFile)
        self.tool_bar.layout.addWidget(self.open_file_button, 0,1, QtCore.Qt.AlignCenter)

        # Desgin an "rotate" button
        self.rotate_button = QPushButton()
        self.rotate_button.setObjectName("rotate_button")
        self.rotate_button.setIcon(QtGui.QIcon(lib_path + "/icons/rotate.png"))
        self.rotate_button.setIconSize(QtCore.QSize(40, 40))
        self.rotate_button.setToolTip("Rotate the current image by 90 degree")
        self.rotate_button.setFixedWidth(100)
        self.rotate_button.clicked.connect(self.rotateImage)
        self.tool_bar.layout.addWidget(self.rotate_button, 0,2, QtCore.Qt.AlignCenter)

        # Desgin an "segmentation" button
        self.segment_button = QPushButton()
        self.segment_button.setObjectName("segment_button")
        self.segment_button.setIcon(QtGui.QIcon(lib_path + "/icons/segment.png"))
        self.segment_button.setIconSize(QtCore.QSize(40, 40))
        self.segment_button.setToolTip("Segment the current image")
        self.segment_button.setFixedWidth(100)
        self.segment_button.clicked.connect(self.segment)
        self.tool_bar.layout.addWidget(self.segment_button, 0,5, QtCore.Qt.AlignCenter)

        # Desgin an slider for mask blending
        self.blending_slider = QSlider()
        self.blending_slider.setObjectName("blending_slider")
        self.blending_slider.setToolTipDuration(-5)
        self.blending_slider.setFixedWidth(200)
        self.blending_slider.setTickInterval(10)
        self.blending_slider.setTickPosition(QSlider.TicksBelow)
        self.blending_slider.setMaximum(100)
        self.blending_slider.setMinimum(0)
        self.blending_slider.setToolTip("Adjust the blending percentage of labels over the current image")
        self.blending_slider.setOrientation(QtCore.Qt.Horizontal)
        self.blending_slider.valueChanged.connect(self.adjust_slider)
        self.tool_bar.layout.addWidget(self.blending_slider, 0, 8, 1, 2, QtCore.Qt.AlignTop)

        # Design an "save images" button
        self.save_file_button = QPushButton()
        self.save_file_button.setObjectName("save")
        self.save_file_button.setIcon(QtGui.QIcon(lib_path + "/icons/file.png"))
        self.save_file_button.setIconSize(QtCore.QSize(40, 40))
        self.save_file_button.setToolTip("Save mask images")
        self.save_file_button.setFixedWidth(100)
        self.save_file_button.clicked.connect(self.saveFile)
        self.tool_bar.layout.addWidget(self.save_file_button, 0,6, QtCore.Qt.AlignCenter)

        # Design labels for tool bar features
        self.capacity_percent = QLabel("0%")
        self.tool_bar.layout.addWidget(self.capacity_percent, 0, 8, 1, 2, QtCore.Qt.AlignHCenter)
        self.open_dir_label = QLabel("Open Directory")
        self.tool_bar.layout.addWidget(self.open_dir_label, 1, 0, QtCore.Qt.AlignCenter)
        self.open_file_label = QLabel("Open File")
        self.tool_bar.layout.addWidget(self.open_file_label, 1, 1, QtCore.Qt.AlignCenter)
        self.rotate_label = QLabel("Rotation")
        self.tool_bar.layout.addWidget(self.rotate_label, 1, 2, QtCore.Qt.AlignCenter)
        self.segment_label = QLabel("Segmentation")
        self.tool_bar.layout.addWidget(self.segment_label, 1, 5, QtCore.Qt.AlignCenter)
        self.segment_label = QLabel("Save")
        self.tool_bar.layout.addWidget(self.segment_label, 1, 6, QtCore.Qt.AlignCenter)
        self.blending_label = QLabel("Blending")
        self.tool_bar.layout.addWidget(self.blending_label, 1, 8, 1, 2, QtCore.Qt.AlignCenter)
        

        # Originial image viewer
        self.image_display.layout.addWidget(self.original_image_viewer, 0, 0, 4, 1,  QtCore.Qt.AlignCenter)
        # Segmented image viewer
        self.image_display.layout.addWidget(self.segmented_image_viewer, 0, 1, 4, 1,  QtCore.Qt.AlignCenter)

        self.original_image_label = QLabel("Original Image")
        self.image_display.layout.addWidget(self.original_image_label, 4, 0, QtCore.Qt.AlignCenter)
        self.converted_image_label = QLabel("Segmented Image")
        self.image_display.layout.addWidget(self.converted_image_label, 4, 1, QtCore.Qt.AlignCenter)

        # Create customized list widget
        self.labels_listbox = QListWidget()
        self.labels_listbox.setFixedWidth(256)   

        items = QtWidgets.QListWidgetItem() 
        #Create widget
        widget = QtWidgets.QWidget()

        widgetLayout = QtWidgets.QGridLayout()
        # for _labelname in config.LABELS:
        #     _l = QLabel(_labelname)
        #     _l.setFixedWidth(130)
        #     hex = self.rgb_to_hex(tuple(config.COLORS[config.LABELS[_labelname]+1]))
        #     _str = "background-color: #{}; border: 0.5px solid #{}; color: black;".format(hex,hex)
        #     _l.setStyleSheet(_str)
        #     widgetLayout.addWidget(_l, config.LABELS[_labelname], 0, QtCore.Qt.AlignCenter)

        for _labelname in config.LABELS:
            _l = QLabel(_labelname)
            _l.setFixedWidth(130)
            if config.MODEL_OPTION == "unet":
                hex = self.rgb_to_hex(tuple(config.COLORS[config.LABELS[_labelname]+1]))
            if config.MODEL_OPTION == "hrnet":
                hex = self.rgb_to_hex(tuple(config.COLORS[config.LABELS[_labelname]+1]))

            _str = "background-color: #{}; border: 0.5px solid #{}; color: black;".format(hex,hex)
            _l.setStyleSheet(_str)
            widgetLayout.addWidget(_l, config.LABELS[_labelname], 0, QtCore.Qt.AlignCenter)

        widgetLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        widget.setLayout(widgetLayout)  
        items.setSizeHint(widget.sizeHint())    
        #Add widget to QListWidget funList
        self.labels_listbox.addItem(items)
        self.labels_listbox.setItemWidget(items, widget)   
        self.labels_listbox.setStyleSheet("background-color: #C8C8C8; border: 0px;")
        self.image_display.layout.addWidget(self.labels_listbox, 1, 2, 1, 1,  QtCore.Qt.AlignCenter)

        self.filenames_listbox = QListWidget()
        self.filenames_listbox.setFixedHeight(372)
        self.filenames_listbox.setFixedWidth(256)
        self.filenames_listbox.clicked.connect(self.selectItem)
        self.filenames_listbox.setStyleSheet("background-color: #C8C8C8; border: 0px;")
        self.image_display.layout.addWidget(self.filenames_listbox, 3, 2, 1, 1,  QtCore.Qt.AlignCenter)

        self.filenames_list_label = QLabel("Label Annotation")
        self.image_display.layout.addWidget(self.filenames_list_label, 0, 2, QtCore.Qt.AlignLeft)
        self.labels_list_label = QLabel("Files")
        self.image_display.layout.addWidget(self.labels_list_label, 2, 2, QtCore.Qt.AlignLeft)
        
        self.toggle_gtmask = QPushButton("Show mask")
        self.toggle_gtmask.clicked.connect(self.changeColor)
        self.toggle_gtmask.setCheckable(True)
        self.toggle_gtmask.setStyleSheet("background-color : lightgrey")
        self.image_display.layout.addWidget(self.toggle_gtmask, 6, 0, QtCore.Qt.AlignLeft)

        self.current_path = QLabel("Current path: ")
        self.current_path.setAlignment(QtCore.Qt.AlignLeft)
        self.image_display.layout.addWidget(self.current_path, 5, 0, 1, 5, QtCore.Qt.AlignLeft)
 
    def changeColor(self):
        idx = self.filenames_listbox.currentRow()
        # if button is checked
        if self.toggle_gtmask.isChecked():
            # setting background color to light-blue
            self.toggle_gtmask.setStyleSheet("background-color : lightblue")
            self.show_mask = True
            self.show_gt_masks()
        # if it is unchecked
        else: 
            # set background color back to light-grey
            self.toggle_gtmask.setStyleSheet("background-color : lightgrey")
            self.show_mask = False
            if len(self.list_paths) != 0:
                self.original_image_viewer.setPhoto(QPixmap(self.list_paths[idx]))
                self.current_original_pixmap = QPixmap(self.list_paths[idx])
                idx = self.filenames_listbox.currentRow()
    
    def show_gt_masks(self):
        if self.show_mask == True:
            self.update_alpha()

    def rgb_to_hex(self, rgb):
        return '%02x%02x%02x' % rgb

    def rotateImage(self):
        transfrom = QtGui.QTransform()
        transfrom.rotate(90)
        self.rotated_original_image = self.current_original_pixmap.transformed(transfrom)
        self.current_original_pixmap = self.rotated_original_image 
        self.original_image_viewer.setPhoto(self.current_original_pixmap)

        if self.is_segmented == True:
            self.rotated_segmented_image = self.current_segmented_pixmap.transformed(transfrom)
            self.current_segmented_pixmap = self.rotated_segmented_image
            self.segmented_image_viewer.setPhoto(self.current_segmented_pixmap)

    
    def openFile(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '/home', "All Files (*);;TIFF (*.TIF *.tif *.TIFF *.tiff)")
        if self.file_path not in self.list_paths and self.file_path != "":
            self.list_paths.append(self.file_path)
            self.current_path.setText("Current path: {}".format(self.file_path))
            self.filenames_listbox.addItem(os.path.basename(self.file_path))
            self.original_image_viewer.setPhoto(QPixmap(self.file_path))
            self.current_original_pixmap = QPixmap(self.file_path)   
            self.current_segmented_pixmap = QPixmap()
            self.segmented_image_viewer.setPhoto(QPixmap())
            self.curr_pth = self.file_path
            self.list_mask.append(None)
            self.list_PIL_img.append(None)
            self.list_PIL_mask.append(None)
            self.list_gt_mask.append(None)
            self.list_PIL_gt.append(None)
        self.selectItem()

    def openFiles(self):
        self.file_paths, _ = QFileDialog.getOpenFileNames(self, 'Open File', '/home', "All Files (*);;TIFF (*.TIF *.tif *.TIFF *.tiff)")
        for path in self.file_paths:
            if path not in self.list_paths:   
                self.list_paths.append(path)         
                self.filenames_listbox.addItem(os.path.basename(path))
                self.original_image_viewer.setPhoto(QPixmap(path))
                self.current_original_pixmap = QPixmap(path) 
                self.current_segmented_pixmap = QPixmap()
                self.segmented_image_viewer.setPhoto(QPixmap())
                self.current_path.setText("Current path: {}".format(path))
                self.curr_pth = path
        self.list_mask = [None] * len(self.list_paths)
        self.list_PIL_img = [None] * len(self.list_paths)
        self.list_PIL_mask = [None] * len(self.list_paths)
        self.list_gt_mask = [None] * len(self.list_paths)
        self.list_PIL_gt = [None] * len(self.list_paths)
        self.selectItem()
    
    def saveFile(self):
        dir = QFileDialog.getExistingDirectory(self, caption="Select directory to save masks")
        for img in self.list_mask:
            if img != None:
                image_name = os.path.basename(self.list_paths[self.list_mask.index(img)])
                image_name_w_out_ext, ext = os.path.splitext(image_name)
                new_name = image_name_w_out_ext + "_mask.png"
                path = os.path.join(dir, new_name)
                img.save(path)

    def selectItem(self):
        idx = self.filenames_listbox.currentRow()
        self.original_image_viewer.setPhoto(QPixmap(self.list_paths[idx]))
        if self.list_mask[idx] is None:
            self.segmented_image_viewer.setPhoto(QPixmap())
        self.update_alpha()
        self.segmented_image_viewer.setPhoto(self.list_mask[idx])
        self.current_original_pixmap = QPixmap(self.list_paths[idx]) 
        self.curr_pth = self.list_paths[idx]
        self.current_path.setText("Current path: {}".format(self.list_paths[idx]))

    def segment(self):
        self.is_segmented = True
        # ret = QMessageBox.question(self, 'Question', "Segment all images?", QMessageBox.YesAll | QMessageBox.No | QMessageBox.Cancel)
        idx = self.filenames_listbox.currentRow()
           
        # if ret == QMessageBox.YesAll:
        if config.MODEL_OPTION == "unet":
            model = visualize.UNet()
        if config.MODEL_OPTION == "hrnet":
            model = visualize.HRNet()

        predmasks, gtmasks = model.visualize(self.list_paths)

        for img_path in self.list_paths:
            img = Image.open(img_path)
            predmask = predmasks[self.list_paths.index(img_path)]
            predmask = Image.fromarray(predmask.astype('uint8'), 'RGB') # convert numpy array to pil
            gtmask = gtmasks[self.list_paths.index(img_path)]
            gtmask = Image.fromarray(gtmask.astype('uint8'), 'RGB') # convert numpy array to pil

            blendimg = Image.blend(img, predmask, self.capacity_value/100)
            blendgt = Image.blend(img, gtmask, self.capacity_value/100)

            blendimg = blendimg.convert("RGB")
            blendgt = blendgt.convert("RGB")
            data = blendimg.tobytes("raw","RGB")
            data_gt = blendgt.tobytes("raw","RGB")
            blendimg = QtGui.QImage(data, blendimg.size[0], blendimg.size[1], QtGui.QImage.Format_RGB888)
            blendgt = QtGui.QImage(data_gt, blendgt.size[0], blendgt.size[1], QtGui.QImage.Format_RGB888)
            blendimg = QtGui.QImage(blendimg)
            blendgt = QtGui.QImage(blendgt)
            qpixmap_blendimg = QPixmap.fromImage(blendimg)
            qpixmap_blendgt = QPixmap.fromImage(blendgt)

            self.list_mask[self.list_paths.index(img_path)] = qpixmap_blendimg
            self.list_gt_mask[self.list_paths.index(img_path)] = qpixmap_blendgt
            self.list_PIL_img[self.list_paths.index(img_path)] = img
            self.list_PIL_mask[self.list_paths.index(img_path)] = predmask
            self.list_PIL_gt[self.list_paths.index(img_path)] = gtmask
        
            # Display
        self.segmented_image_viewer.setPhoto(self.list_mask[idx])
        self.current_segmented_pixmap = self.list_mask[idx]
        # self.original_image_viewer.setPhoto(QPixmap(self.list_paths[self.list_paths.index(img_path)]))

        # if ret == QMessageBox.No:
        #     visualize_result = visualize.visualization()
        #     mask = visualize_result.get_mask(image_path=self.list_paths[idx])

        #     img = Image.open(self.list_paths[idx])
        #     blend = Image.blend(img, mask, self.capacity_value/100)

        #     blend = blend.convert("RGB")
        #     data = blend.tobytes("raw","RGB")
        #     blend = QtGui.QImage(data, blend.size[0], blend.size[1], QtGui.QImage.Format_RGB888)
        #     blend = QtGui.QImage(blend)
        #     qpixmap_blend = QPixmap.fromImage(blend)

        #     self.list_mask[self.list_paths.index(self.list_paths[idx])] = qpixmap_blend
        #     self.list_PIL_img[self.list_paths.index(self.list_paths[idx])] = img
        #     self.list_PIL_mask[self.list_paths.index(self.list_paths[idx])] = mask
            
        #     # Display
        #     self.segmented_image_viewer.setPhoto(self.list_mask[idx])
    
    def adjust_slider(self):
        self.capacity_value = self.blending_slider.value()
        self.capacity_percent.setText("{}%".format(str(self.capacity_value)))
        self.update_alpha()
    
    def update_alpha(self):
        idx = self.filenames_listbox.currentRow()
        if len(self.list_PIL_img) != 0:     
            img = self.list_PIL_img[idx]
            mask = self.list_PIL_mask[idx]
            gt_mask = self.list_PIL_gt[idx]
            qpixmap_blend = QPixmap()
            qpixmap_blendgt = QPixmap()
            
            if img != None:
                blend = Image.blend(img, mask, self.capacity_value/100)
                blend_gt = Image.blend(img, gt_mask, self.capacity_value/100)
                blend = blend.convert("RGB")
                blend_gt = blend_gt.convert("RGB")
                data = blend.tobytes("raw","RGB")
                data_gt = blend_gt.tobytes("raw", "RGB")
                blend = QtGui.QImage(data, blend.size[0], blend.size[1], QtGui.QImage.Format_RGB888)
                blend_gt = QtGui.QImage(data_gt, blend_gt.size[0], blend_gt.size[1], QtGui.QImage.Format_RGB888)
                blend = QtGui.QImage(blend)
                blend_gt = QtGui.QImage(blend_gt)
                qpixmap_blend = QPixmap.fromImage(blend)
                qpixmap_blendgt = QPixmap.fromImage(blend_gt)
            
            self.list_mask[self.list_paths.index(self.list_paths[idx])] = qpixmap_blend       
            self.list_gt_mask[self.list_paths.index(self.list_paths[idx])] = qpixmap_blendgt  
            self.current_segmented_pixmap = qpixmap_blend
            self.segmented_image_viewer.setPhoto(self.list_mask[idx])
            if self.show_mask == True:
                if gt_mask == None:
                    qpixmap_blendgt = QPixmap(self.list_paths[idx])
                self.current_original_pixmap = qpixmap_blendgt
                self.original_image_viewer.setPhoto(qpixmap_blendgt)

def window():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    window()

