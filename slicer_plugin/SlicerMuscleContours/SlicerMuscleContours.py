import colorsys
import hashlib
import logging
import math

import numpy as np
import qt
import slicer
import vtk
from scipy import ndimage
from vtk.util.numpy_support import vtk_to_numpy
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

SLICE_VIEW_NAMES = ("Red", "Yellow", "Green")
LOCATOR_PLANE_COLORS = {
    "Red": (0.952, 0.297, 0.252),
    "Yellow": (0.952, 0.871, 0.255),
    "Green": (0.426, 0.748, 0.270),
}
LOCATOR_PLANE_MODEL_NAMES = {
    viewName: "MuscleContoursLocatorPlane{} (do not touch)".format(viewName)
    for viewName in SLICE_VIEW_NAMES
}
LOCATOR_AZIMUTH_MIN_MM = 1.5
LOCATOR_PICK_TOL_MM = 3.0
INSERT_POINT_TOLERANCE_MM = 12.0
THREE_D_POINT_PICK_TOLERANCE_PIXELS = 12.0


class SlicerMuscleContours(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Muscle Contours"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Markups", "Segmentations"]
        self.parent.contributors = ["SlicerNNInteractive contributors"]
        self.parent.helpText = (
            "Draw editable closed contours on key slices, inspect them in 3D, "
            "and interpolate a segmentation between slices."
        )
        self.parent.acknowledgementText = ""


class SlicerMuscleContoursWidget(
    ScriptedLoadableModuleWidget, VTKObservationMixin
):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._sliceObservers = []
        self._sceneObserversInstalled = False
        self._placementObservers = []
        self._placingNode = None
        self._placingViewName = None
        self._insertObservers = []
        self._locatorRotationObservers = []
        self._locatorDrag = None
        self._locatorPlaneModels = {}
        self._locatorPlaneSliceObservers = []
        self._threeDPointPickObservers = []
        self._lastSliceOffsets = {}
        self._drawShortcut = None
        self._legacyContoursAssigned = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SlicerMuscleContoursLogic()

        self.reloadModuleButton = qt.QPushButton("更新模块")
        self.reloadModuleButton.toolTip = (
            "热重载 SlicerMuscleContours 的 Python 代码和界面，保留当前场景数据。"
        )
        self.reloadModuleButton.connect("clicked()", self.onReloadModule)
        self.parent.layout().addWidget(self.reloadModuleButton)

        parameters = qt.QGroupBox("轮廓组")
        form = qt.QFormLayout(parameters)
        self.referenceSelector = slicer.qMRMLNodeComboBox()
        self.referenceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.referenceSelector.noneEnabled = False
        self.referenceSelector.addEnabled = False
        self.referenceSelector.removeEnabled = False
        self.referenceSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("参考影像：", self.referenceSelector)
        self.referenceStatusLabel = qt.QLabel("尚未选择参考影像")
        self.referenceStatusLabel.wordWrap = True
        form.addRow("当前状态：", self.referenceStatusLabel)

        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.addEnabled = True
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("输出分割：", self.segmentationSelector)

        self.groupName = qt.QLineEdit("Muscle")
        form.addRow("肌肉名称：", self.groupName)
        self.parent.layout().addWidget(parameters)

        drawing = qt.QGroupBox("关键层轮廓")
        drawingLayout = qt.QVBoxLayout(drawing)
        drawingViewRow = qt.QHBoxLayout()
        drawingViewRow.addWidget(qt.QLabel("绘制视图："))
        self.drawingViewComboBox = qt.QComboBox()
        self.drawingViewComboBox.addItems(list(SLICE_VIEW_NAMES))
        drawingViewRow.addWidget(self.drawingViewComboBox)
        drawingLayout.addLayout(drawingViewRow)
        self.drawButton = qt.QPushButton("在当前切片绘制闭合轮廓")
        self.drawButton.checkable = True
        self.drawButton.toolTip = "每次左键单击生成一个点；鼠标移动不会采样；右键结束。"
        self.insertButton = qt.QPushButton("在选中曲线上插入控制点")
        self.insertButton.checkable = True
        self.insertButton.toolTip = "启用后，在曲线附近单击即可插入新的可拖动控制点。"
        self.copyButton = qt.QPushButton("复制最近轮廓到当前切片")
        self.deleteButton = qt.QPushButton("删除选中轮廓")
        drawingLayout.addWidget(self.drawButton)
        drawingLayout.addWidget(self.insertButton)
        drawingLayout.addWidget(self.copyButton)
        drawingLayout.addWidget(self.deleteButton)

        self.contourList = qt.QListWidget()
        self.contourList.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        drawingLayout.addWidget(self.contourList)
        self.hintLabel = qt.QLabel(
            "左键单击逐点绘制，右键闭合；二维仅显示当前层轮廓，三维显示全部轮廓。"
        )
        self.hintLabel.wordWrap = True
        drawingLayout.addWidget(self.hintLabel)
        self.parent.layout().addWidget(drawing)

        positioning = qt.QGroupBox("单序列旋转定位")
        positioningLayout = qt.QVBoxLayout(positioning)
        self.locatorRotateButton = qt.QPushButton("拖动定位线旋转切片")
        self.locatorRotateButton.checkable = True
        self.locatorRotateButton.toolTip = (
            "启用后，在任一二维视图中抓住彩色切片交线并拖动。"
        )
        positioningLayout.addWidget(self.locatorRotateButton)
        self.alignAcquisitionButton = qt.QPushButton("对齐影像采集平面")
        self.alignAcquisitionButton.toolTip = (
            "使切片视图与 DICOM 原始采集切片方向一致。"
        )
        self.resetOrientationButton = qt.QPushButton("恢复标准解剖方向")
        self.resetOrientationButton.toolTip = (
            "恢复 Red=轴向、Yellow=矢状、Green=冠状。"
        )
        positioningLayout.addWidget(self.alignAcquisitionButton)
        positioningLayout.addWidget(self.resetOrientationButton)
        self.showLocatorPlanesCheckBox = qt.QCheckBox("显示 3D 定位面")
        self.showLocatorPlanesCheckBox.checked = True
        positioningLayout.addWidget(self.showLocatorPlanesCheckBox)
        brightnessRow = qt.QHBoxLayout()
        brightnessRow.addWidget(qt.QLabel("定位面亮度/透明度："))
        self.locatorPlaneBrightnessSlider = qt.QSlider(qt.Qt.Horizontal)
        self.locatorPlaneBrightnessSlider.minimum = 0
        self.locatorPlaneBrightnessSlider.maximum = 100
        self.locatorPlaneBrightnessSlider.value = 25
        self.locatorPlaneBrightnessSlider.toolTip = (
            "调整三维定位面的填充强度；0 为仅显示边框。"
        )
        brightnessRow.addWidget(self.locatorPlaneBrightnessSlider)
        positioningLayout.addLayout(brightnessRow)
        self.parent.layout().addWidget(positioning)

        output = qt.QGroupBox("体积生成")
        outputLayout = qt.QVBoxLayout(output)
        self.generateButton = qt.QPushButton("插值并生成分割")
        self.generateButton.toolTip = (
            "支持同方向的轴对齐或平行斜切关键层轮廓。"
        )
        outputLayout.addWidget(self.generateButton)
        refineRow = qt.QHBoxLayout()
        refineRow.addWidget(qt.QLabel("调整轮廓所在视图："))
        self.refineViewComboBox = qt.QComboBox()
        self.refineViewComboBox.addItems(list(SLICE_VIEW_NAMES))
        refineRow.addWidget(self.refineViewComboBox)
        outputLayout.addLayout(refineRow)
        self.addRefinementContourButton = qt.QPushButton(
            "添加当前层插值调整轮廓"
        )
        self.addRefinementContourButton.toolTip = (
            "从已生成分割截取所选视图当前层的边界，转换为可拖动闭合曲线。"
        )
        outputLayout.addWidget(self.addRefinementContourButton)
        self.showOutput3DCheckBox = qt.QCheckBox("显示当前输出 3D 模型")
        self.showOutput3DCheckBox.checked = True
        outputLayout.addWidget(self.showOutput3DCheckBox)
        self.deleteOutputModelsButton = qt.QPushButton(
            "删除当前输出的全部 3D 结果"
        )
        self.deleteOutputModelsButton.toolTip = (
            "删除当前输出 Segmentation 中的全部 Segment；"
            "保留轮廓和 Segmentation 节点，可稍后重新生成。"
        )
        outputLayout.addWidget(self.deleteOutputModelsButton)
        self.outputStatusLabel = qt.QLabel("等待生成")
        self.outputStatusLabel.wordWrap = True
        outputLayout.addWidget(self.outputStatusLabel)
        self.parent.layout().addWidget(output)
        self.parent.layout().addStretch(1)

        self.drawButton.connect("toggled(bool)", self.onDrawToggled)
        self.drawingViewComboBox.connect(
            "currentIndexChanged(int)", self.onDrawingViewIndexChanged
        )
        self.refineViewComboBox.connect(
            "currentIndexChanged(int)", self.onRefineViewIndexChanged
        )
        self.insertButton.connect("toggled(bool)", self.onInsertToggled)
        self.locatorRotateButton.connect(
            "toggled(bool)", self.onLocatorRotateToggled
        )
        self.alignAcquisitionButton.connect(
            "clicked()", self.alignViewsToAcquisitionPlane
        )
        self.resetOrientationButton.connect(
            "clicked()", self.resetViewsToStandardOrientation
        )
        self.showLocatorPlanesCheckBox.connect(
            "toggled(bool)", self.onLocatorPlanesVisibilityChanged
        )
        self.locatorPlaneBrightnessSlider.connect(
            "valueChanged(int)", self.onLocatorPlaneBrightnessChanged
        )
        self.copyButton.connect("clicked()", self.onCopy)
        self.deleteButton.connect("clicked()", self.onDelete)
        self.generateButton.connect("clicked()", self.onGenerate)
        self.addRefinementContourButton.connect(
            "clicked()", self.onAddRefinementContour
        )
        self.showOutput3DCheckBox.connect(
            "toggled(bool)", self.onOutput3DVisibilityChanged
        )
        self.deleteOutputModelsButton.connect(
            "clicked()", self.onDeleteAllOutputModels
        )
        self.contourList.connect("currentRowChanged(int)", self.onContourSelected)
        self.referenceSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.onReferenceVolumeChanged
        )
        self.segmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.onOutputSegmentationChanged,
        )
        self._drawShortcut = qt.QShortcut(
            qt.QKeySequence("D"), slicer.util.mainWindow()
        )
        self._drawShortcut.setContext(qt.Qt.ApplicationShortcut)
        self._drawShortcut.connect("activated()", self.toggleDrawShortcut)
        self.drawButton.toolTip += "　快捷键：D"

        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.NodeAddedEvent,
            self.onSceneNodeChanged,
        )
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.NodeRemovedEvent,
            self.onSceneNodeChanged,
        )
        self._sceneObserversInstalled = True
        self.enableSliceIntersections()
        self.installSliceObservers()
        self.onReferenceVolumeChanged(self.referenceSelector.currentNode())
        self.onOutputSegmentationChanged(
            self.segmentationSelector.currentNode()
        )
        self.ensureLocatorPlanes()
        self.installThreeDPointPickObservers()

    def cleanup(self):
        self.endClickPlacement(cancelIncomplete=True)
        self.removeInsertObservers()
        self.removeLocatorRotationObservers()
        self.removeLocatorPlanes()
        self.removeThreeDPointPickObservers()
        self.removeSliceObservers()
        self.removeObservers()
        if self._drawShortcut is not None:
            self._drawShortcut.setParent(None)
            self._drawShortcut = None

    def enter(self):
        self.enableSliceIntersections()
        self.installSliceObservers()
        self.ensureLocatorPlanes()
        self.installThreeDPointPickObservers()
        self.refreshContourList()
        self.updateContourVisibility()

    def exit(self):
        pass

    def onReloadModule(self):
        slicer.util.showStatusMessage("正在更新 Muscle Contours 模块…", 2000)
        slicer.util.reloadScriptedModule("SlicerMuscleContours")

    def installSliceObservers(self):
        self.removeSliceObservers()
        self._lastSliceOffsets = {}
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName)
            if not widget:
                continue
            node = widget.mrmlSliceNode()
            logic = widget.sliceLogic()
            if logic is not None:
                self._lastSliceOffsets[viewName] = logic.GetSliceOffset()
            tag = node.AddObserver(
                vtk.vtkCommand.ModifiedEvent,
                lambda caller, event, view=viewName, sliceLogic=logic: (
                    self.onSliceNodeModified(
                        view, sliceLogic, caller, event
                    )
                ),
            )
            self._sliceObservers.append((node, tag))

    def onSliceNodeModified(
        self, viewName, sliceLogic, caller=None, event=None
    ):
        if sliceLogic is not None:
            currentOffset = sliceLogic.GetSliceOffset()
            previousOffset = self._lastSliceOffsets.get(viewName)
            self._lastSliceOffsets[viewName] = currentOffset
            if (
                self._placingNode is not None
                and viewName == self._placingViewName
                and previousOffset is not None
                and abs(currentOffset - previousOffset) > 1e-4
            ):
                self.closeDrawingForSliceChange()
        self.clearContourSelectionIfSliceChanged(caller)
        self.updateContourVisibility()
        self.updateLocatorPlanes()

    def clearContourSelectionIfSliceChanged(self, modifiedSliceNode):
        """Deselect a contour after its source 2D view leaves that layer."""
        node = self.selectedContour()
        if node is None or modifiedSliceNode is None:
            return
        viewName = node.GetAttribute(self.logic.ATTR_VIEW)
        if viewName not in SLICE_VIEW_NAMES:
            return
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.sliceWidget(viewName) if layoutManager else None
        sourceSliceNode = widget.mrmlSliceNode() if widget else None
        if (
            sourceSliceNode is None
            or modifiedSliceNode.GetID() != sourceSliceNode.GetID()
        ):
            return
        originText = node.GetAttribute(self.logic.ATTR_PLANE_ORIGIN)
        normalText = node.GetAttribute(self.logic.ATTR_PLANE_NORMAL)
        if not originText or not normalText:
            return
        matrix = sourceSliceNode.GetSliceToRAS()
        currentOrigin = np.array(
            [matrix.GetElement(row, 3) for row in range(3)], dtype=float
        )
        currentNormal = np.array(
            [matrix.GetElement(row, 2) for row in range(3)], dtype=float
        )
        length = np.linalg.norm(currentNormal)
        if length < 1e-9:
            return
        currentNormal /= length
        contourOrigin = self.logic._parseVector(originText)
        contourNormal = self.logic._normalize(
            self.logic._parseVector(normalText)
        )
        parallel = abs(float(np.dot(contourNormal, currentNormal))) > 0.999
        distance = abs(
            float(np.dot(contourOrigin - currentOrigin, currentNormal))
        )
        if parallel and distance < 0.25:
            return
        blocked = self.contourList.blockSignals(True)
        self.contourList.setCurrentRow(-1)
        self.contourList.clearSelection()
        self.contourList.blockSignals(blocked)

    def removeSliceObservers(self):
        for node, tag in self._sliceObservers:
            if node:
                node.RemoveObserver(tag)
        self._sliceObservers = []

    def activeSlice(self):
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return None, None
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName)
            if widget and widget.sliceView().underMouse():
                self.drawingViewComboBox.setCurrentText(viewName)
                return viewName, widget.mrmlSliceNode()
        viewName = self.drawingViewComboBox.currentText
        if viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName)
            if widget:
                return viewName, widget.mrmlSliceNode()
        return None, None

    def onDrawingViewIndexChanged(self, index):
        if index < 0 or index >= self.refineViewComboBox.count:
            return
        blocked = self.refineViewComboBox.blockSignals(True)
        self.refineViewComboBox.setCurrentIndex(index)
        self.refineViewComboBox.blockSignals(blocked)

    def onRefineViewIndexChanged(self, index):
        if index < 0 or index >= self.drawingViewComboBox.count:
            return
        blocked = self.drawingViewComboBox.blockSignals(True)
        self.drawingViewComboBox.setCurrentIndex(index)
        self.drawingViewComboBox.blockSignals(blocked)

    @staticmethod
    def enableSliceIntersections():
        """Show the other two slice planes as colored locator lines in 2D."""
        for displayNode in slicer.util.getNodesByClass(
            "vtkMRMLSliceDisplayNode"
        ):
            displayNode.SetIntersectingSlicesVisibility(1)
        for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            sliceNode.Modified()
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            for viewName in SLICE_VIEW_NAMES:
                widget = layoutManager.sliceWidget(viewName)
                if widget:
                    widget.sliceView().scheduleRender()

    def onReferenceVolumeChanged(self, volume):
        """Make the selected reference volume visibly active in all 2D views."""
        if not hasattr(self, "referenceStatusLabel"):
            return
        if volume is None:
            self.referenceStatusLabel.text = "尚未选择参考影像"
            self.refreshContourList()
            return
        imageData = volume.GetImageData()
        if imageData is None:
            self.referenceStatusLabel.text = "所选节点没有可用的体素数据"
            self.refreshContourList()
            return
        dimensions = imageData.GetDimensions()
        spacing = volume.GetSpacing()
        self.referenceStatusLabel.text = (
            "已选择：{}　尺寸：{}×{}×{}　间距：{:.3g}×{:.3g}×{:.3g} mm"
        ).format(
            volume.GetName(),
            dimensions[0],
            dimensions[1],
            dimensions[2],
            spacing[0],
            spacing[1],
            spacing[2],
        )

        applicationLogic = slicer.app.applicationLogic()
        selectionNode = applicationLogic.GetSelectionNode()
        selectionNode.SetReferenceActiveVolumeID(volume.GetID())
        applicationLogic.PropagateVolumeSelection(0)

        layoutManager = slicer.app.layoutManager()
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName) if layoutManager else None
            if widget is None:
                continue
            compositeNode = widget.mrmlSliceCompositeNode()
            if compositeNode:
                compositeNode.SetBackgroundVolumeID(volume.GetID())
            try:
                widget.sliceLogic().FitSliceToAll()
            except Exception:
                pass
        self.alignViewsToAcquisitionPlane(showStatus=False)
        self.enableSliceIntersections()
        self.ensureLocatorPlanes()
        self.updateLocatorPlanes()
        self.refreshContourList()
        self.updateContourVisibility()
        slicer.util.showStatusMessage(
            "参考影像已切换为：{}".format(volume.GetName()), 3000
        )

    def ensureOutputSegmentation(self, volume):
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is not None:
            return segmentation
        segmentation = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "MuscleContoursSegmentation"
        )
        segmentation.CreateDefaultDisplayNodes()
        if volume is not None:
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(
                volume
            )
        self.segmentationSelector.setCurrentNode(segmentation)
        return segmentation

    def onOutputSegmentationChanged(self, segmentation):
        if segmentation is not None and not self._legacyContoursAssigned:
            for node in self.logic.contourNodes(
                self.referenceSelector.currentNode()
            ):
                if not node.GetNodeReferenceID(
                    self.logic.ROLE_OUTPUT_SEGMENTATION
                ):
                    node.SetNodeReferenceID(
                        self.logic.ROLE_OUTPUT_SEGMENTATION,
                        segmentation.GetID(),
                    )
            self._legacyContoursAssigned = True
        self.refreshContourList()
        self.updateContourVisibility()
        visible = True
        if segmentation is not None:
            segmentation.CreateDefaultDisplayNodes()
            displayNode = segmentation.GetDisplayNode()
            if displayNode is not None:
                visible = bool(displayNode.GetVisibility3D())
        blocked = self.showOutput3DCheckBox.blockSignals(True)
        self.showOutput3DCheckBox.checked = visible
        self.showOutput3DCheckBox.blockSignals(blocked)

    def onOutput3DVisibilityChanged(self, visible):
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is None:
            return
        segmentation.CreateDefaultDisplayNodes()
        displayNode = segmentation.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetVisibility3D(bool(visible))
        self.outputStatusLabel.text = (
            "当前输出 3D 模型已显示"
            if visible
            else "当前输出 3D 模型已隐藏"
        )

    def onDeleteAllOutputModels(self):
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is None:
            slicer.util.errorDisplay("请先选择要清理的输出 Segmentation。")
            return
        segmentIds = vtk.vtkStringArray()
        segmentation.GetSegmentation().GetSegmentIDs(segmentIds)
        if segmentIds.GetNumberOfValues() == 0:
            self.outputStatusLabel.text = "当前输出中没有可删除的 3D 结果。"
            return
        if not slicer.util.confirmYesNoDisplay(
            "确定删除当前输出 Segmentation 中的全部生成结果吗？\n"
            "绘制轮廓会保留，可稍后重新生成。"
        ):
            return
        idsToRemove = [
            segmentIds.GetValue(index)
            for index in range(segmentIds.GetNumberOfValues())
        ]
        for segmentId in idsToRemove:
            segmentation.GetSegmentation().RemoveSegment(segmentId)
        segmentation.Modified()
        self.outputStatusLabel.text = (
            "已删除当前输出的全部 3D 结果；绘制轮廓仍保留。"
        )

    @staticmethod
    def setStandardSliceOrientation(sliceNode, viewName):
        """Use typed setters, avoiding legacy string preset lookup."""
        if viewName == "Red":
            sliceNode.SetOrientationToAxial()
        elif viewName == "Yellow":
            sliceNode.SetOrientationToSagittal()
        elif viewName == "Green":
            sliceNode.SetOrientationToCoronal()

    @staticmethod
    def volumeIsOblique(volume):
        if volume is None:
            return False
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASDirectionMatrix(matrix)
        threshold = math.cos(math.radians(2.5))
        for column in range(3):
            if max(
                abs(matrix.GetElement(row, column)) for row in range(3)
            ) < threshold:
                return True
        return False

    def alignViewsToAcquisitionPlane(self, showStatus=True):
        volume = self.referenceSelector.currentNode()
        if volume is None:
            if showStatus:
                slicer.util.errorDisplay("请先选择参考影像。")
            return
        layoutManager = slicer.app.layoutManager()
        if self.volumeIsOblique(volume):
            for viewName in SLICE_VIEW_NAMES:
                widget = layoutManager.sliceWidget(viewName) if layoutManager else None
                if widget is None:
                    continue
                sliceNode = widget.mrmlSliceNode()
                self.setStandardSliceOrientation(sliceNode, viewName)
                sliceNode.RotateToVolumePlane(volume)
                try:
                    widget.sliceLogic().SnapSliceOffsetToIJK()
                    widget.sliceLogic().FitSliceToAll()
                except Exception:
                    pass
        else:
            self.resetViewsToStandardOrientation(showStatus=False)
        self.enableSliceIntersections()
        if showStatus:
            slicer.util.showStatusMessage("切片视图已对齐影像采集平面。", 3000)

    def resetViewsToStandardOrientation(self, showStatus=True):
        """Discard manual locator rotations and restore anatomical views."""
        self._locatorDrag = None
        layoutManager = slicer.app.layoutManager()
        volume = self.referenceSelector.currentNode()
        bounds = [0.0] * 6
        if volume is not None:
            volume.GetRASBounds(bounds)
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName) if layoutManager else None
            if widget is None:
                continue
            sliceNode = widget.mrmlSliceNode()
            try:
                self.setStandardSliceOrientation(sliceNode, viewName)
                if volume is not None:
                    sliceNode.JumpSliceByCentering(
                        0.5 * (bounds[0] + bounds[1]),
                        0.5 * (bounds[2] + bounds[3]),
                        0.5 * (bounds[4] + bounds[5]),
                    )
            except Exception:
                logging.exception("Failed to reset %s slice orientation", viewName)
            try:
                widget.sliceLogic().FitSliceToAll()
            except Exception:
                pass
        self.enableSliceIntersections()
        if showStatus:
            slicer.util.showStatusMessage("已恢复标准解剖方向。", 3000)

    def ensureLocatorPlanes(self):
        """Create one translucent 3D model for each standard slice plane."""
        for viewName in SLICE_VIEW_NAMES:
            model = self._locatorPlaneModels.get(viewName)
            if model is None or model.GetScene() is None:
                model = slicer.mrmlScene.GetFirstNodeByName(
                    LOCATOR_PLANE_MODEL_NAMES[viewName]
                )
            if model is None:
                model = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLModelNode", LOCATOR_PLANE_MODEL_NAMES[viewName]
                )
                model.CreateDefaultDisplayNodes()
                model.SetHideFromEditors(True)
            display = model.GetDisplayNode()
            color = LOCATOR_PLANE_COLORS[viewName]
            display.SetColor(*color)
            display.SetEdgeColor(*color)
            display.SetEdgeVisibility(True)
            display.SetLineWidth(3.0)
            display.SetVisibility2D(False)
            display.SetVisibility3D(self.showLocatorPlanesCheckBox.checked)
            display.SetOpacity(self.locatorPlaneBrightnessSlider.value / 100.0)
            display.SetBackfaceCulling(False)
            self._locatorPlaneModels[viewName] = model
        self.updateLocatorPlanes()

    def removeLocatorPlanes(self):
        for model in self._locatorPlaneModels.values():
            if model is not None and model.GetScene() is not None:
                slicer.mrmlScene.RemoveNode(model)
        self._locatorPlaneModels = {}

    def locatorPlaneGeometry(self, sliceNode, volume):
        """Return a quad covering the reference volume in the slice plane."""
        matrix = sliceNode.GetSliceToRAS()
        origin = np.array(
            [matrix.GetElement(row, 3) for row in range(3)], dtype=float
        )
        xAxis = np.array(
            [matrix.GetElement(row, 0) for row in range(3)], dtype=float
        )
        yAxis = np.array(
            [matrix.GetElement(row, 1) for row in range(3)], dtype=float
        )
        xAxis /= max(np.linalg.norm(xAxis), 1e-9)
        yAxis /= max(np.linalg.norm(yAxis), 1e-9)
        bounds = [0.0] * 6
        volume.GetRASBounds(bounds)
        corners = [
            np.array([x, y, z], dtype=float)
            for x in (bounds[0], bounds[1])
            for y in (bounds[2], bounds[3])
            for z in (bounds[4], bounds[5])
        ]
        xCoordinates = [float(np.dot(corner - origin, xAxis)) for corner in corners]
        yCoordinates = [float(np.dot(corner - origin, yAxis)) for corner in corners]
        margin = 2.0
        xMinimum, xMaximum = min(xCoordinates) - margin, max(xCoordinates) + margin
        yMinimum, yMaximum = min(yCoordinates) - margin, max(yCoordinates) + margin
        return [
            origin + xMinimum * xAxis + yMinimum * yAxis,
            origin + xMaximum * xAxis + yMinimum * yAxis,
            origin + xMaximum * xAxis + yMaximum * yAxis,
            origin + xMinimum * xAxis + yMaximum * yAxis,
        ]

    def updateLocatorPlanes(self):
        if not self._locatorPlaneModels:
            return
        volume = self.referenceSelector.currentNode()
        layoutManager = slicer.app.layoutManager()
        if volume is None or layoutManager is None:
            return
        for viewName, model in self._locatorPlaneModels.items():
            widget = layoutManager.sliceWidget(viewName)
            sliceNode = widget.mrmlSliceNode() if widget else None
            if sliceNode is None or model is None:
                continue
            corners = self.locatorPlaneGeometry(sliceNode, volume)
            points = vtk.vtkPoints()
            for corner in corners:
                points.InsertNextPoint(*corner)
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(4)
            for index in range(4):
                polygon.GetPointIds().SetId(index, index)
            polygons = vtk.vtkCellArray()
            polygons.InsertNextCell(polygon)
            lines = vtk.vtkCellArray()
            border = vtk.vtkPolyLine()
            border.GetPointIds().SetNumberOfIds(5)
            for index in range(4):
                border.GetPointIds().SetId(index, index)
            border.GetPointIds().SetId(4, 0)
            lines.InsertNextCell(border)
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetPolys(polygons)
            polyData.SetLines(lines)
            model.SetAndObservePolyData(polyData)
            model.Modified()

    def onLocatorPlanesVisibilityChanged(self, visible):
        self.ensureLocatorPlanes()
        for model in self._locatorPlaneModels.values():
            if model and model.GetDisplayNode():
                model.GetDisplayNode().SetVisibility3D(bool(visible))

    def onLocatorPlaneBrightnessChanged(self, value):
        self.ensureLocatorPlanes()
        opacity = max(0.0, min(1.0, float(value) / 100.0))
        for model in self._locatorPlaneModels.values():
            if model and model.GetDisplayNode():
                model.GetDisplayNode().SetOpacity(opacity)

    def onDrawToggled(self, checked):
        if not checked:
            self.endClickPlacement(cancelIncomplete=True)
            return
        if self.insertButton.checked:
            self.insertButton.checked = False
        if self.locatorRotateButton.checked:
            self.locatorRotateButton.checked = False
        volume = self.referenceSelector.currentNode()
        if not volume:
            slicer.util.errorDisplay("请先选择参考影像。")
            self.drawButton.checked = False
            return
        group = self.groupName.text.strip()
        if not group:
            slicer.util.errorDisplay("请输入肌肉名称。")
            self.drawButton.checked = False
            return
        segmentation = self.ensureOutputSegmentation(volume)
        viewName, sliceNode = self.activeSlice()
        if not sliceNode:
            slicer.util.errorDisplay("未找到可用的二维切片视图。")
            self.drawButton.checked = False
            return
        try:
            index, origin, normal = self.logic.sliceDescription(volume, sliceNode)
            node = self.logic.createContourNode(
                volume,
                group,
                viewName,
                index,
                origin,
                normal,
                segmentation,
            )
            self.beginClickPlacement(node, viewName)
            self.refreshContourList()
            self.selectNode(node, navigate=False)
            self.updateContourVisibility()
        except ValueError as exc:
            slicer.util.errorDisplay(str(exc))
            self.drawButton.checked = False

    def toggleDrawShortcut(self):
        focusWidget = qt.QApplication.focusWidget()
        if focusWidget is not None and (
            focusWidget.inherits("QLineEdit")
            or focusWidget.inherits("QTextEdit")
            or focusWidget.inherits("QPlainTextEdit")
            or focusWidget.inherits("QSpinBox")
            or focusWidget.inherits("QDoubleSpinBox")
        ):
            return
        self.drawButton.checked = not self.drawButton.checked

    def closeDrawingForSliceChange(self):
        self.endClickPlacement(cancelIncomplete=True)
        blocked = self.drawButton.blockSignals(True)
        self.drawButton.checked = False
        self.drawButton.blockSignals(blocked)
        slicer.util.showStatusMessage(
            "切片已翻页，闭合轮廓绘制模式已自动关闭。", 3000
        )

    def beginClickPlacement(self, node, viewName):
        """Place exactly one control point per left-button press."""
        self.endClickPlacement(cancelIncomplete=True)
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.sliceWidget(viewName) if layoutManager else None
        interactor = widget.sliceView().interactor() if widget else None
        if interactor is None:
            raise ValueError("当前二维视图没有可用的鼠标交互器。")
        self._placingNode = node
        self._placingViewName = viewName
        leftHolder, rightHolder = {}, {}
        leftTag = interactor.AddObserver(
            vtk.vtkCommand.LeftButtonPressEvent,
            lambda caller, event, holder=leftHolder: self.onPlacementClick(
                caller, holder.get("tag")
            ),
            10.0,
        )
        leftHolder["tag"] = leftTag
        rightTag = interactor.AddObserver(
            vtk.vtkCommand.RightButtonPressEvent,
            lambda caller, event, holder=rightHolder: self.onPlacementFinish(
                caller, holder.get("tag")
            ),
            10.0,
        )
        rightHolder["tag"] = rightTag
        self._placementObservers = [
            (interactor, leftTag),
            (interactor, rightTag),
        ]
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        slicer.util.showStatusMessage(
            "左键单击逐点绘制，右键结束闭合轮廓。", 5000
        )

    def onPlacementClick(self, caller, tag):
        node = self._placingNode
        layoutManager = slicer.app.layoutManager()
        widget = (
            layoutManager.sliceWidget(self._placingViewName)
            if layoutManager and self._placingViewName
            else None
        )
        sliceNode = widget.mrmlSliceNode() if widget else None
        if node is None or sliceNode is None:
            return
        x, y = caller.GetEventPosition()
        ras4 = sliceNode.GetXYToRAS().MultiplyPoint(
            [float(x), float(y), 0.0, 1.0]
        )
        node.AddControlPointWorld(vtk.vtkVector3d(ras4[0], ras4[1], ras4[2]))
        self.abortInteractorEvent(caller, tag)

    def onPlacementFinish(self, caller, tag):
        self.abortInteractorEvent(caller, tag)
        self.endClickPlacement(cancelIncomplete=True)
        blocked = self.drawButton.blockSignals(True)
        self.drawButton.checked = False
        self.drawButton.blockSignals(blocked)

    def endClickPlacement(self, cancelIncomplete=False):
        node = self._placingNode
        for interactor, tag in self._placementObservers:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._placementObservers = []
        self._placingNode = None
        self._placingViewName = None
        if (
            cancelIncomplete
            and node is not None
            and node.GetNumberOfControlPoints() < 3
            and node.GetScene() is not None
        ):
            slicer.mrmlScene.RemoveNode(node)
        self.refreshContourList()

    def onInsertToggled(self, checked):
        self.removeInsertObservers()
        if not checked:
            return
        self.endClickPlacement(cancelIncomplete=True)
        if self.drawButton.checked:
            blocked = self.drawButton.blockSignals(True)
            self.drawButton.checked = False
            self.drawButton.blockSignals(blocked)
        if self.locatorRotateButton.checked:
            self.locatorRotateButton.checked = False
        node = self.selectedContour()
        if node is None:
            slicer.util.errorDisplay("请先在列表中选择一个轮廓。")
            self.insertButton.checked = False
            return
        layoutManager = slicer.app.layoutManager()
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName) if layoutManager else None
            interactor = widget.sliceView().interactor() if widget else None
            if interactor is None:
                continue
            holder = {}
            tag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                lambda caller, event, view=viewName, h=holder: (
                    self.onInsertPointClick(view, caller, h.get("tag"))
                ),
                10.0,
            )
            holder["tag"] = tag
            self._insertObservers.append((interactor, tag))
        slicer.util.showStatusMessage("请在选中轮廓的曲线附近单击。", 4000)

    def removeInsertObservers(self):
        for interactor, tag in self._insertObservers:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._insertObservers = []

    def onInsertPointClick(self, viewName, caller, tag):
        node = self.selectedContour()
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.sliceWidget(viewName) if layoutManager else None
        sliceNode = widget.mrmlSliceNode() if widget else None
        if node is None or sliceNode is None:
            return
        x, y = caller.GetEventPosition()
        ras4 = sliceNode.GetXYToRAS().MultiplyPoint(
            [float(x), float(y), 0.0, 1.0]
        )
        try:
            self.logic.insertControlPointNearCurve(node, ras4[:3])
        except ValueError as exc:
            slicer.util.showStatusMessage(str(exc), 3000)
            return
        self.abortInteractorEvent(caller, tag)
        self.insertButton.checked = False

    def onCopy(self):
        volume = self.referenceSelector.currentNode()
        source = self.selectedContour()
        if not source:
            contours = self.logic.contourNodes(
                volume,
                self.groupName.text.strip(),
                self.segmentationSelector.currentNode(),
            )
            source = contours[-1] if contours else None
        if not volume or not source:
            slicer.util.errorDisplay("请先选择或绘制一个源轮廓。")
            return
        viewName, sliceNode = self.activeSlice()
        if not sliceNode:
            return
        try:
            index, origin, normal = self.logic.sliceDescription(volume, sliceNode)
            node = self.logic.copyContourToPlane(
                source, volume, viewName, index, origin, normal
            )
            self.refreshContourList()
            self.selectNode(node, navigate=False)
            self.updateContourVisibility()
        except ValueError as exc:
            slicer.util.errorDisplay(str(exc))

    def onDelete(self):
        node = self.selectedContour()
        if node:
            group = node.GetAttribute(self.logic.ATTR_GROUP)
            auxiliaryDisplays = []
            for index in range(node.GetNumberOfDisplayNodes()):
                displayNode = node.GetNthDisplayNode(index)
                if (
                    displayNode is not None
                    and displayNode.GetAttribute(self.logic.ATTR_3D_DISPLAY) == "1"
                ):
                    auxiliaryDisplays.append(displayNode)
            slicer.mrmlScene.RemoveNode(node)
            for displayNode in auxiliaryDisplays:
                if displayNode.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(displayNode)
            self.removeGeneratedSegment(group)
            self.refreshContourList()

    def removeGeneratedSegment(self, group):
        """Remove the stale generated surface after a source contour is deleted."""
        if not group:
            return
        segmentation = self.segmentationSelector.currentNode()
        if segmentation is None:
            return
        removed = False
        while True:
            segmentId = self.logic.findSegmentId(segmentation, group)
            if not segmentId:
                break
            segmentation.GetSegmentation().RemoveSegment(segmentId)
            removed = True
        if removed:
            segmentation.Modified()
            self.outputStatusLabel.text = (
                "已删除轮廓及旧的“{}”三维分割，请按需重新生成。".format(group)
            )

    @staticmethod
    def abortInteractorEvent(caller, tag):
        if tag is None:
            return
        try:
            command = caller.GetCommand(tag)
            if command:
                command.SetAbortFlag(1)
        except Exception:
            pass

    def onLocatorRotateToggled(self, checked):
        self.removeLocatorRotationObservers()
        if not checked:
            return
        self.enableSliceIntersections()
        self.endClickPlacement(cancelIncomplete=True)
        if self.drawButton.checked:
            blocked = self.drawButton.blockSignals(True)
            self.drawButton.checked = False
            self.drawButton.blockSignals(blocked)
        if self.insertButton.checked:
            self.insertButton.checked = False
        layoutManager = slicer.app.layoutManager()
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName) if layoutManager else None
            interactor = widget.sliceView().interactor() if widget else None
            if interactor is None:
                continue
            pressHolder, moveHolder, releaseHolder = {}, {}, {}
            pressTag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                lambda caller, event, view=viewName, h=pressHolder: (
                    self.onLocatorPress(view, caller, h.get("tag"))
                ),
                10.0,
            )
            pressHolder["tag"] = pressTag
            moveTag = interactor.AddObserver(
                vtk.vtkCommand.MouseMoveEvent,
                lambda caller, event, view=viewName, h=moveHolder: (
                    self.onLocatorDrag(view, caller, h.get("tag"))
                ),
                10.0,
            )
            moveHolder["tag"] = moveTag
            releaseTag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonReleaseEvent,
                lambda caller, event, view=viewName, h=releaseHolder: (
                    self.onLocatorRelease(view, caller, h.get("tag"))
                ),
                10.0,
            )
            releaseHolder["tag"] = releaseTag
            self._locatorRotationObservers.extend(
                [
                    (interactor, pressTag),
                    (interactor, moveTag),
                    (interactor, releaseTag),
                ]
            )
        slicer.util.showStatusMessage(
            "抓住二维视图中的彩色定位线并拖动旋转。", 5000
        )

    def removeLocatorRotationObservers(self):
        for interactor, tag in self._locatorRotationObservers:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._locatorRotationObservers = []
        self._locatorDrag = None

    @staticmethod
    def vectorDot(a, b):
        return sum(a[index] * b[index] for index in range(3))

    @staticmethod
    def vectorCross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def vectorNorm(vector):
        return math.sqrt(sum(value * value for value in vector))

    @classmethod
    def unitMatrixColumn(cls, matrix, column):
        vector = [matrix.GetElement(row, column) for row in range(3)]
        length = cls.vectorNorm(vector)
        return None if length < 1e-9 else [value / length for value in vector]

    @classmethod
    def slicePlane(cls, sliceNode):
        matrix = sliceNode.GetSliceToRAS()
        origin = [matrix.GetElement(row, 3) for row in range(3)]
        normal = cls.unitMatrixColumn(matrix, 2)
        return origin, normal

    def standardSliceNodes(self):
        layoutManager = slicer.app.layoutManager()
        for viewName in SLICE_VIEW_NAMES:
            widget = layoutManager.sliceWidget(viewName) if layoutManager else None
            if widget:
                yield viewName, widget.mrmlSliceNode()

    def threePlaneIntersection(self):
        planes = [self.slicePlane(node) for _name, node in self.standardSliceNodes()]
        if len(planes) != 3 or any(normal is None for _origin, normal in planes):
            return None
        matrix = np.asarray([normal for _origin, normal in planes], dtype=float)
        values = np.asarray(
            [self.vectorDot(normal, origin) for origin, normal in planes],
            dtype=float,
        )
        try:
            return np.linalg.solve(matrix, values).tolist()
        except np.linalg.LinAlgError:
            return None

    def onLocatorPress(self, operationView, caller, tag):
        self._locatorDrag = None
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.sliceWidget(operationView) if layoutManager else None
        operationNode = widget.mrmlSliceNode() if widget else None
        if operationNode is None:
            return
        x, y = caller.GetEventPosition()
        ras4 = operationNode.GetXYToRAS().MultiplyPoint(
            [float(x), float(y), 0.0, 1.0]
        )
        point = list(ras4[:3])
        _operationOrigin, operationNormal = self.slicePlane(operationNode)
        if operationNormal is None:
            return
        bestView, bestDistance = None, None
        for candidateView, candidateNode in self.standardSliceNodes():
            if candidateView == operationView:
                continue
            candidateOrigin, candidateNormal = self.slicePlane(candidateNode)
            if candidateNormal is None:
                continue
            distance = abs(
                self.vectorDot(
                    [point[i] - candidateOrigin[i] for i in range(3)],
                    candidateNormal,
                )
            )
            if bestDistance is None or distance < bestDistance:
                bestView, bestDistance = candidateView, distance
        if (
            bestView is None
            or bestDistance is None
            or bestDistance > LOCATOR_PICK_TOL_MM
        ):
            return
        targetWidget = layoutManager.sliceWidget(bestView)
        targetNode = targetWidget.mrmlSliceNode() if targetWidget else None
        if targetNode is None:
            return
        baseline = vtk.vtkMatrix4x4()
        baseline.DeepCopy(targetNode.GetSliceToRAS())
        anchor = self.threePlaneIntersection() or point
        operationMatrix = operationNode.GetSliceToRAS()
        xAxis = self.unitMatrixColumn(operationMatrix, 0)
        yAxis = self.unitMatrixColumn(operationMatrix, 1)
        if xAxis is None or yAxis is None:
            return
        delta = [point[i] - anchor[i] for i in range(3)]
        startAngle = (
            None
            if self.vectorNorm(delta) < LOCATOR_AZIMUTH_MIN_MM
            else math.atan2(self.vectorDot(delta, yAxis), self.vectorDot(delta, xAxis))
        )
        self._locatorDrag = {
            "operationView": operationView,
            "targetView": bestView,
            "axis": self.vectorCross(xAxis, yAxis),
            "baseline": baseline,
            "anchor": anchor,
            "xAxis": xAxis,
            "yAxis": yAxis,
            "startAngle": startAngle,
        }
        self.abortInteractorEvent(caller, tag)

    def onLocatorDrag(self, operationView, caller, tag):
        drag = self._locatorDrag
        if not drag:
            return
        layoutManager = slicer.app.layoutManager()
        operationWidget = layoutManager.sliceWidget(drag["operationView"])
        operationNode = operationWidget.mrmlSliceNode() if operationWidget else None
        targetWidget = layoutManager.sliceWidget(drag["targetView"])
        targetNode = targetWidget.mrmlSliceNode() if targetWidget else None
        if operationNode is None or targetNode is None:
            return
        x, y = caller.GetEventPosition()
        ras4 = operationNode.GetXYToRAS().MultiplyPoint(
            [float(x), float(y), 0.0, 1.0]
        )
        delta = [ras4[i] - drag["anchor"][i] for i in range(3)]
        if self.vectorNorm(delta) < LOCATOR_AZIMUTH_MIN_MM:
            return
        currentAngle = math.atan2(
            self.vectorDot(delta, drag["yAxis"]),
            self.vectorDot(delta, drag["xAxis"]),
        )
        if drag["startAngle"] is None:
            drag["startAngle"] = currentAngle
            return
        angle = currentAngle - drag["startAngle"]
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
        self.rotateSliceView(
            targetNode,
            drag["axis"],
            math.degrees(angle),
            drag["baseline"],
            drag["anchor"],
        )
        self.abortInteractorEvent(caller, tag)

    def onLocatorRelease(self, operationView, caller, tag):
        if self._locatorDrag:
            self._locatorDrag = None
            self.abortInteractorEvent(caller, tag)

    @staticmethod
    def rotateSliceView(sliceNode, axis, angleDegrees, baseline, center):
        transform = vtk.vtkTransform()
        transform.Translate(*center)
        transform.RotateWXYZ(angleDegrees, *axis)
        transform.Translate(*[-value for value in center])
        result = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(transform.GetMatrix(), baseline, result)
        sliceNode.GetSliceToRAS().DeepCopy(result)
        sliceNode.UpdateMatrices()

    def onGenerate(self):
        volume = self.referenceSelector.currentNode()
        group = self.groupName.text.strip()
        if not volume or not group:
            slicer.util.errorDisplay("请选择参考影像并输入肌肉名称。")
            return
        self.generateButton.enabled = False
        self.outputStatusLabel.text = "正在栅格化轮廓并生成分割…"
        slicer.app.processEvents()
        try:
            segmentation = self.ensureOutputSegmentation(volume)
            segmentation, segmentId = self.logic.generateSegmentation(
                volume, group, segmentation
            )
            self.segmentationSelector.setCurrentNode(segmentation)
            segmentation.GetDisplayNode().SetVisibility(True)
            segmentation.GetDisplayNode().SetVisibility3D(
                self.showOutput3DCheckBox.checked
            )
            segmentation.GetDisplayNode().SetSegmentVisibility(segmentId, True)
            if self.logic.lastFusionUsedFallback:
                self.outputStatusLabel.text = (
                    "生成完成：{}（方向交集为空，已使用并集融合）".format(group)
                )
            else:
                self.outputStatusLabel.text = "生成完成：{}".format(group)
            slicer.util.showStatusMessage(
                "已生成分割：{}".format(group), 4000
            )
        except (ValueError, RuntimeError) as exc:
            self.outputStatusLabel.text = "生成失败：{}".format(exc)
            slicer.util.errorDisplay(str(exc))
        finally:
            self.generateButton.enabled = True

    def onAddRefinementContour(self):
        """Convert the generated boundary on one slice into a new key contour."""
        volume = self.referenceSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        group = self.groupName.text.strip()
        viewName = self.refineViewComboBox.currentText
        if volume is None or segmentation is None or not group:
            slicer.util.errorDisplay(
                "请先选择参考影像、输出分割，并完成一次插值生成。"
            )
            return
        layoutManager = slicer.app.layoutManager()
        sliceWidget = layoutManager.sliceWidget(viewName) if layoutManager else None
        sliceNode = sliceWidget.mrmlSliceNode() if sliceWidget else None
        if sliceNode is None:
            slicer.util.errorDisplay("所选二维视图不可用。")
            return
        try:
            existing = self.logic.findContourOnPlane(
                volume, group, sliceNode, segmentation
            )
            if existing is not None:
                self.selectNode(existing, navigate=False)
                self.outputStatusLabel.text = "当前层已有关键轮廓，已选中。"
                return
            index, origin, normal = self.logic.sliceDescription(volume, sliceNode)
            node = self.logic.createRefinementContour(
                segmentation,
                group,
                volume,
                viewName,
                index,
                origin,
                normal,
            )
            self.refreshContourList()
            self.selectNode(node, navigate=False)
            self.updateContourVisibility()
            self.outputStatusLabel.text = (
                "调整轮廓已添加；拖动控制点后再次点击“插值并生成分割”。"
            )
        except (ValueError, RuntimeError) as exc:
            self.outputStatusLabel.text = "添加调整轮廓失败：{}".format(exc)
            slicer.util.errorDisplay(str(exc))

    def selectedContour(self):
        item = self.contourList.currentItem()
        if not item:
            return None
        return slicer.mrmlScene.GetNodeByID(item.data(qt.Qt.UserRole))

    def selectNode(self, node, navigate=True):
        for row in range(self.contourList.count):
            item = self.contourList.item(row)
            if item.data(qt.Qt.UserRole) == node.GetID():
                if navigate:
                    self.contourList.setCurrentRow(row)
                else:
                    blocked = self.contourList.blockSignals(True)
                    self.contourList.setCurrentRow(row)
                    self.contourList.blockSignals(blocked)
                break

    def onContourSelected(self, row):
        node = self.selectedContour()
        if node:
            self.jumpToContourPlane(node)

    def jumpToContourPlane(self, node):
        """Move the contour's source slice view to its stored plane origin."""
        if node is None:
            return
        viewName = node.GetAttribute(self.logic.ATTR_VIEW)
        originText = node.GetAttribute(self.logic.ATTR_PLANE_ORIGIN)
        if viewName not in SLICE_VIEW_NAMES or not originText:
            return
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.sliceWidget(viewName) if layoutManager else None
        sliceNode = widget.mrmlSliceNode() if widget else None
        if sliceNode is None:
            return
        origin = self.logic._parseVector(originText)
        try:
            sliceNode.JumpSliceByCentering(
                float(origin[0]), float(origin[1]), float(origin[2])
            )
        except Exception:
            # Older Slicer versions expose offset through slice logic only.
            matrix = sliceNode.GetSliceToRAS()
            normal = np.array(
                [matrix.GetElement(row, 2) for row in range(3)], dtype=float
            )
            normalLength = np.linalg.norm(normal)
            if normalLength > 1e-9:
                normal /= normalLength
                widget.sliceLogic().SetSliceOffset(float(np.dot(origin, normal)))
        self.updateContourVisibility()
        try:
            widget.sliceView().scheduleRender()
        except Exception:
            pass

    def installThreeDPointPickObservers(self):
        """Click visible read-only 3D control points without enabling dragging."""
        self.removeThreeDPointPickObservers()
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        for viewIndex in range(layoutManager.threeDViewCount):
            widget = layoutManager.threeDWidget(viewIndex)
            view = widget.threeDView() if widget else None
            interactor = view.interactor() if view else None
            if interactor is None:
                continue
            holder = {}
            tag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                lambda caller, event, currentView=view, h=holder: (
                    self.onThreeDPointClicked(
                        currentView, caller, h.get("tag")
                    )
                ),
                10.0,
            )
            holder["tag"] = tag
            self._threeDPointPickObservers.append((interactor, tag))

    def removeThreeDPointPickObservers(self):
        for interactor, tag in self._threeDPointPickObservers:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._threeDPointPickObservers = []

    def onThreeDPointClicked(self, threeDView, caller, tag):
        """Pick the nearest projected control point and jump, never drag."""
        rendererCollection = threeDView.renderWindow().GetRenderers()
        renderer = rendererCollection.GetFirstRenderer()
        if renderer is None:
            return
        clickX, clickY = caller.GetEventPosition()
        bestNode = None
        bestDistance = THREE_D_POINT_PICK_TOLERANCE_PIXELS
        point = [0.0, 0.0, 0.0]
        volume = self.referenceSelector.currentNode()
        for node in self.logic.contourNodes(
            volume, segmentation=self.segmentationSelector.currentNode()
        ):
            for pointIndex in range(node.GetNumberOfControlPoints()):
                node.GetNthControlPointPositionWorld(pointIndex, point)
                renderer.SetWorldPoint(point[0], point[1], point[2], 1.0)
                renderer.WorldToDisplay()
                displayPoint = renderer.GetDisplayPoint()
                if displayPoint[2] < 0.0 or displayPoint[2] > 1.0:
                    continue
                distance = math.hypot(
                    float(displayPoint[0]) - float(clickX),
                    float(displayPoint[1]) - float(clickY),
                )
                if distance <= bestDistance:
                    bestDistance = distance
                    bestNode = node
        if bestNode is None:
            return
        self.selectNode(bestNode)
        self.jumpToContourPlane(bestNode)
        self.abortInteractorEvent(caller, tag)

    def onSceneNodeChanged(self, caller=None, event=None, node=None):
        if node is None or node.IsA("vtkMRMLMarkupsClosedCurveNode"):
            qt.QTimer.singleShot(0, self.refreshContourList)

    def refreshContourList(self, *_args):
        if not hasattr(self, "contourList"):
            return
        selectedId = (
            self.selectedContour().GetID() if self.selectedContour() else None
        )
        self.contourList.clear()
        volume = self.referenceSelector.currentNode()
        contourNodes = self.logic.contourNodes(
            volume, segmentation=self.segmentationSelector.currentNode()
        )
        for node in contourNodes:
            item = qt.QListWidgetItem(
                "{} · {} · 第 {} 层".format(
                    node.GetAttribute(self.logic.ATTR_GROUP),
                    node.GetAttribute(self.logic.ATTR_VIEW),
                    node.GetAttribute(self.logic.ATTR_SLICE_INDEX),
                )
            )
            item.setData(qt.Qt.UserRole, node.GetID())
            self.contourList.addItem(item)
            if node.GetID() == selectedId:
                self.contourList.setCurrentItem(item)

    def updateContourVisibility(self):
        self.logic.updateContourVisibility(
            self.segmentationSelector.currentNode()
        )


class SlicerMuscleContoursLogic(
    ScriptedLoadableModuleLogic, VTKObservationMixin
):
    ATTR_MANAGED = "SlicerMuscleContours.Managed"
    ATTR_GROUP = "SlicerMuscleContours.Group"
    ATTR_VIEW = "SlicerMuscleContours.SourceView"
    ATTR_SLICE_INDEX = "SlicerMuscleContours.SliceIndex"
    ATTR_PLANE_ORIGIN = "SlicerMuscleContours.PlaneOriginRAS"
    ATTR_PLANE_NORMAL = "SlicerMuscleContours.PlaneNormalRAS"
    ATTR_AXIS = "SlicerMuscleContours.IJKAxis"
    ATTR_3D_DISPLAY = "SlicerMuscleContours.ThreeDDisplay"
    ATTR_COLOR_PREFIX = "SlicerMuscleContours.Color."
    ROLE_REFERENCE = "SlicerMuscleContours.ReferenceVolume"
    ROLE_OUTPUT_SEGMENTATION = "SlicerMuscleContours.OutputSegmentation"

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)
        self._pointObservers = {}
        self._projecting = set()
        self.lastFusionUsedFallback = False

    @staticmethod
    def _vectorText(values):
        return ",".join("{:.12g}".format(float(value)) for value in values)

    @staticmethod
    def _parseVector(text):
        return np.array([float(value) for value in text.split(",")], dtype=float)

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(vector, dtype=float)
        length = np.linalg.norm(vector)
        if length < 1e-9:
            raise ValueError("切片平面法向量无效。")
        return vector / length

    def volumeDirectionAxes(self, volume):
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASDirectionMatrix(matrix)
        return [
            self._normalize([matrix.GetElement(r, c) for r in range(3)])
            for c in range(3)
        ]

    def sliceDescription(self, volume, sliceNode):
        matrix = sliceNode.GetSliceToRAS()
        origin = np.array(
            [matrix.GetElement(row, 3) for row in range(3)], dtype=float
        )
        normal = self._normalize(
            [matrix.GetElement(row, 2) for row in range(3)]
        )
        axes = self.volumeDirectionAxes(volume)
        dots = [abs(float(np.dot(normal, axis))) for axis in axes]
        axis = int(np.argmax(dots))
        rasToIjk = vtk.vtkMatrix4x4()
        volume.GetRASToIJKMatrix(rasToIjk)
        ijk = [0.0, 0.0, 0.0, 1.0]
        rasToIjk.MultiplyPoint([*origin, 1.0], ijk)
        index = int(round(ijk[axis]))
        dims = volume.GetImageData().GetDimensions()
        if index < 0 or index >= dims[axis]:
            raise ValueError("当前切片位于参考影像范围之外。")
        return index, origin, normal

    def insertControlPointNearCurve(self, node, worldPosition):
        """Insert a control point into the nearest control-polygon segment."""
        count = node.GetNumberOfControlPoints()
        if count < 3:
            raise ValueError("轮廓尚未完成，不能插入控制点。")
        origin = self._parseVector(node.GetAttribute(self.ATTR_PLANE_ORIGIN))
        normal = self._normalize(
            self._parseVector(node.GetAttribute(self.ATTR_PLANE_NORMAL))
        )
        click = np.asarray(worldPosition, dtype=float)
        click = click - np.dot(click - origin, normal) * normal
        controlPoints = []
        point = [0.0, 0.0, 0.0]
        for index in range(count):
            node.GetNthControlPointPositionWorld(index, point)
            controlPoints.append(np.asarray(point, dtype=float).copy())

        bestDistance = None
        bestIndex = None
        bestPoint = None
        for index in range(count):
            first = controlPoints[index]
            second = controlPoints[(index + 1) % count]
            segment = second - first
            denominator = float(np.dot(segment, segment))
            fraction = (
                0.0
                if denominator < 1e-12
                else float(np.clip(np.dot(click - first, segment) / denominator, 0, 1))
            )
            candidate = first + fraction * segment
            distance = float(np.linalg.norm(click - candidate))
            if bestDistance is None or distance < bestDistance:
                bestDistance = distance
                bestIndex = index + 1
                bestPoint = click
        if bestDistance is None or bestDistance > INSERT_POINT_TOLERANCE_MM:
            raise ValueError("点击位置离选中曲线太远，请靠近曲线后重试。")

        vector = vtk.vtkVector3d(
            float(bestPoint[0]), float(bestPoint[1]), float(bestPoint[2])
        )
        if hasattr(node, "InsertControlPointWorld"):
            try:
                node.InsertControlPointWorld(bestIndex, vector)
                return bestIndex
            except TypeError:
                node.InsertControlPointWorld(vector, bestIndex)
                return bestIndex
        try:
            node.InsertControlPoint(bestIndex, vector)
        except TypeError:
            node.InsertControlPoint(vector, bestIndex)
        return bestIndex

    def createContourNode(
        self,
        volume,
        group,
        viewName,
        index,
        origin,
        normal,
        segmentation=None,
    ):
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsClosedCurveNode",
            "{}_{}_{}".format(group, viewName, index),
        )
        node.SetAttribute(self.ATTR_MANAGED, "1")
        node.SetAttribute(self.ATTR_GROUP, group)
        node.SetAttribute(self.ATTR_VIEW, viewName)
        node.SetAttribute(self.ATTR_SLICE_INDEX, str(int(index)))
        node.SetAttribute(self.ATTR_PLANE_ORIGIN, self._vectorText(origin))
        node.SetAttribute(
            self.ATTR_PLANE_NORMAL, self._vectorText(self._normalize(normal))
        )
        axes = self.volumeDirectionAxes(volume)
        axis = int(
            np.argmax([abs(float(np.dot(normal, item))) for item in axes])
        )
        node.SetAttribute(self.ATTR_AXIS, str(axis))
        node.SetNodeReferenceID(self.ROLE_REFERENCE, volume.GetID())
        if segmentation is not None:
            node.SetNodeReferenceID(
                self.ROLE_OUTPUT_SEGMENTATION, segmentation.GetID()
            )
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetSelectedColor(1.0, 0.65, 0.1)
        display.SetColor(0.2, 0.9, 0.35)
        display.SetGlyphScale(1.5)
        display.SetLineThickness(0.35)
        display.SetPropertiesLabelVisibility(False)
        try:
            node.SetCurveTypeToCardinalSpline()
        except AttributeError:
            logging.warning("Cardinal spline API unavailable; using default curve.")
        self.ensureSeparate2D3DDisplays(node)
        self.observeContour(node)
        return node

    def ensureSeparate2D3DDisplays(self, node):
        """Keep 2D control points editable while making 3D curve line-only."""
        if node is None:
            return None, None
        node.CreateDefaultDisplayNodes()
        display2D = None
        display3D = None
        for index in range(node.GetNumberOfDisplayNodes()):
            display = node.GetNthDisplayNode(index)
            if display is None:
                continue
            if display.GetAttribute(self.ATTR_3D_DISPLAY) == "1":
                display3D = display
            elif display2D is None:
                display2D = display
        if display2D is None:
            display2D = node.GetDisplayNode()
        if display3D is None and display2D is not None:
            display3D = slicer.mrmlScene.AddNewNodeByClass(
                display2D.GetClassName(),
                "{}_3DDisplay".format(node.GetName()),
            )
            display3D.SetAttribute(self.ATTR_3D_DISPLAY, "1")
            node.AddAndObserveDisplayNodeID(display3D.GetID())

        if display2D is not None:
            display2D.SetVisibility2D(True)
            display2D.SetVisibility3D(False)
            # Hide the center translation/rotation wheel. Individual contour
            # control points remain editable in the 2D slice view.
            display2D.SetHandlesInteractive(False)
            display2D.SetGlyphScale(1.5)
        if display3D is not None:
            display3D.SetColor(0.2, 0.9, 0.35)
            display3D.SetSelectedColor(0.2, 0.9, 0.35)
            display3D.SetActiveColor(0.2, 0.9, 0.35)
            display3D.SetLineThickness(0.35)
            display3D.SetTextScale(0)
            # Points remain visible for click-to-jump, but Markups interaction
            # handles stay disabled so they cannot be dragged in 3D.
            display3D.SetGlyphScale(1.5)
            display3D.SetHandlesInteractive(False)
            display3D.SetVisibility2D(False)
            display3D.SetVisibility3D(True)
        return display2D, display3D

    def observeContour(self, node):
        if not node or node.GetID() in self._pointObservers:
            return
        tag = node.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            self.onContourPointModified,
        )
        self._pointObservers[node.GetID()] = (node, tag)

    def onContourPointModified(self, node, event=None):
        nodeId = node.GetID()
        if nodeId in self._projecting:
            return
        originText = node.GetAttribute(self.ATTR_PLANE_ORIGIN)
        normalText = node.GetAttribute(self.ATTR_PLANE_NORMAL)
        if not originText or not normalText:
            return
        origin = self._parseVector(originText)
        normal = self._normalize(self._parseVector(normalText))
        self._projecting.add(nodeId)
        wasModifying = node.StartModify()
        try:
            point = [0.0, 0.0, 0.0]
            for pointIndex in range(node.GetNumberOfControlPoints()):
                node.GetNthControlPointPositionWorld(pointIndex, point)
                position = np.asarray(point, dtype=float)
                projected = position - np.dot(position - origin, normal) * normal
                node.SetNthControlPointPositionWorld(
                    pointIndex,
                    vtk.vtkVector3d(
                        float(projected[0]),
                        float(projected[1]),
                        float(projected[2]),
                    ),
                )
        finally:
            node.EndModify(wasModifying)
            self._projecting.discard(nodeId)

    def contourNodes(self, volume=None, group=None, segmentation=None):
        result = []
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsClosedCurveNode"):
            if node.GetAttribute(self.ATTR_MANAGED) != "1":
                continue
            if volume and node.GetNodeReferenceID(self.ROLE_REFERENCE) != volume.GetID():
                continue
            if group and node.GetAttribute(self.ATTR_GROUP) != group:
                continue
            if (
                segmentation is not None
                and node.GetNodeReferenceID(self.ROLE_OUTPUT_SEGMENTATION)
                != segmentation.GetID()
            ):
                continue
            self.ensureSeparate2D3DDisplays(node)
            self.observeContour(node)
            result.append(node)
        result.sort(
            key=lambda item: (
                int(item.GetAttribute(self.ATTR_AXIS) or 0),
                int(item.GetAttribute(self.ATTR_SLICE_INDEX) or 0),
            )
        )
        return result

    def findContourOnPlane(
        self,
        volume,
        group,
        sliceNode,
        segmentation=None,
        toleranceMm=0.25,
    ):
        matrix = sliceNode.GetSliceToRAS()
        currentOrigin = np.array(
            [matrix.GetElement(row, 3) for row in range(3)], dtype=float
        )
        currentNormal = self._normalize(
            [matrix.GetElement(row, 2) for row in range(3)]
        )
        for node in self.contourNodes(volume, group, segmentation):
            origin = self._parseVector(node.GetAttribute(self.ATTR_PLANE_ORIGIN))
            normal = self._normalize(
                self._parseVector(node.GetAttribute(self.ATTR_PLANE_NORMAL))
            )
            parallel = abs(float(np.dot(normal, currentNormal))) > 0.999
            distance = abs(float(np.dot(origin - currentOrigin, currentNormal)))
            if parallel and distance <= toleranceMm:
                return node
        return None

    @staticmethod
    def findSegmentId(segmentation, segmentName):
        segmentationData = segmentation.GetSegmentation()
        segmentIds = vtk.vtkStringArray()
        segmentationData.GetSegmentIDs(segmentIds)
        for index in range(segmentIds.GetNumberOfValues()):
            segmentId = segmentIds.GetValue(index)
            segment = segmentationData.GetSegment(segmentId)
            if segment and segment.GetName() == segmentName:
                return segmentId
        return None

    def colorForSegmentationGroup(self, segmentation, group):
        """Return a persistent color scoped to segmentation node and group."""
        attributeKey = self.ATTR_COLOR_PREFIX + hashlib.sha1(
            group.encode("utf-8")
        ).hexdigest()

        # Respect a color that the user changed on the existing segment.
        existingSegmentId = self.findSegmentId(segmentation, group)
        if existingSegmentId:
            existingSegment = segmentation.GetSegmentation().GetSegment(
                existingSegmentId
            )
            if existingSegment is not None:
                color = existingSegment.GetColor()
                if color is not None and len(color) >= 3:
                    color = tuple(float(color[index]) for index in range(3))
                    segmentation.SetAttribute(
                        attributeKey, self._vectorText(color)
                    )
                    return color

        storedColor = segmentation.GetAttribute(attributeKey)
        if storedColor:
            color = self._parseVector(storedColor)
            if len(color) == 3:
                return tuple(float(value) for value in color)

        # A different segmentation node ID yields a different stable hue.
        identity = "{}::{}".format(segmentation.GetID(), group)
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:4], byteorder="big") / float(2**32)
        saturation = 0.68 + 0.16 * (digest[4] / 255.0)
        value = 0.88 + 0.10 * (digest[5] / 255.0)
        color = colorsys.hsv_to_rgb(hue, saturation, value)
        segmentation.SetAttribute(attributeKey, self._vectorText(color))
        return color

    def createRefinementContour(
        self,
        segmentation,
        segmentName,
        volume,
        viewName,
        sliceIndex,
        planeOrigin,
        planeNormal,
    ):
        """Cut the generated surface without changing any slice-view state."""
        segmentId = self.findSegmentId(segmentation, segmentName)
        if not segmentId:
            raise ValueError(
                "输出分割中找不到名为“{}”的片段。".format(segmentName)
            )
        representationName = (
            slicer.vtkSegmentationConverter
            .GetSegmentationClosedSurfaceRepresentationName()
        )
        if not segmentation.GetSegmentation().ContainsRepresentation(
            representationName
        ):
            segmentation.CreateClosedSurfaceRepresentation()
        segment = segmentation.GetSegmentation().GetSegment(segmentId)
        surface = segment.GetRepresentation(representationName)
        if surface is None or surface.GetNumberOfPoints() == 0:
            raise RuntimeError("生成的分割没有可截取的闭合表面。")

        worldSurface = surface
        parentTransform = segmentation.GetParentTransformNode()
        if parentTransform is not None:
            generalTransform = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                parentTransform, None, generalTransform
            )
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(generalTransform)
            transformFilter.SetInputData(surface)
            transformFilter.Update()
            worldSurface = transformFilter.GetOutput()

        plane = vtk.vtkPlane()
        plane.SetOrigin(*planeOrigin)
        plane.SetNormal(*self._normalize(planeNormal))
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(worldSurface)
        cutter.Update()
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(cutter.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        intersection = stripper.GetOutput()
        if intersection.GetNumberOfCells() == 0:
            raise ValueError(
                "当前层没有插值分割边界，请将切片移动到分割体积内部。"
            )

        bestLoop = None
        bestLength = -1.0
        pointIds = vtk.vtkIdList()
        for cellIndex in range(intersection.GetNumberOfCells()):
            intersection.GetCellPoints(cellIndex, pointIds)
            if pointIds.GetNumberOfIds() < 3:
                continue
            loop = np.asarray(
                [
                    intersection.GetPoint(pointIds.GetId(pointIndex))
                    for pointIndex in range(pointIds.GetNumberOfIds())
                ],
                dtype=float,
            )
            closed = np.vstack([loop, loop[0]])
            length = float(
                np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
            )
            if length > bestLength:
                bestLength = length
                bestLoop = loop
        if bestLoop is None:
            raise RuntimeError("当前层交线无法组成闭合调整轮廓。")

        controlPoints = self._resampleClosedCurve(bestLoop, 24)
        node = self.createContourNode(
            volume,
            segmentName,
            viewName,
            sliceIndex,
            planeOrigin,
            planeNormal,
            segmentation,
        )
        node.SetName(
            "{}_{}_{}_Refinement".format(segmentName, viewName, sliceIndex)
        )
        for point in controlPoints:
            node.AddControlPointWorld(vtk.vtkVector3d(*point))
        return node

    def copyContourToPlane(
        self, source, volume, viewName, index, origin, normal
    ):
        target = self.createContourNode(
            volume,
            source.GetAttribute(self.ATTR_GROUP),
            viewName,
            index,
            origin,
            normal,
            slicer.mrmlScene.GetNodeByID(
                source.GetNodeReferenceID(self.ROLE_OUTPUT_SEGMENTATION)
            ),
        )
        sourceOrigin = self._parseVector(
            source.GetAttribute(self.ATTR_PLANE_ORIGIN)
        )
        delta = np.asarray(origin) - sourceOrigin
        point = [0.0, 0.0, 0.0]
        for pointIndex in range(source.GetNumberOfControlPoints()):
            source.GetNthControlPointPositionWorld(pointIndex, point)
            position = np.asarray(point) + delta
            target.AddControlPointWorld(vtk.vtkVector3d(*position))
        return target

    def updateContourVisibility(self, segmentation=None):
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return
        slicePlanes = {}
        for viewName in ("Red", "Yellow", "Green"):
            widget = layoutManager.sliceWidget(viewName)
            if not widget:
                continue
            sliceNode = widget.mrmlSliceNode()
            matrix = sliceNode.GetSliceToRAS()
            origin = np.array(
                [matrix.GetElement(row, 3) for row in range(3)], dtype=float
            )
            normal = self._normalize(
                [matrix.GetElement(row, 2) for row in range(3)]
            )
            slicePlanes[sliceNode.GetID()] = (viewName, origin, normal)

        threeDViewIds = []
        for viewNode in slicer.util.getNodesByClass("vtkMRMLViewNode"):
            threeDViewIds.append(viewNode.GetID())

        for contour in self.contourNodes():
            display2D, display3D = self.ensureSeparate2D3DDisplays(contour)
            if display2D is None:
                continue
            display2D.RemoveAllViewNodeIDs()
            if display3D is not None:
                display3D.RemoveAllViewNodeIDs()
            if (
                segmentation is not None
                and contour.GetNodeReferenceID(
                    self.ROLE_OUTPUT_SEGMENTATION
                )
                != segmentation.GetID()
            ):
                if not contour.GetLocked():
                    contour.SetLocked(True)
                continue
            for viewId in threeDViewIds:
                if display3D is not None:
                    display3D.AddViewNodeID(viewId)
            contourOrigin = self._parseVector(
                contour.GetAttribute(self.ATTR_PLANE_ORIGIN)
            )
            contourNormal = self._normalize(
                self._parseVector(contour.GetAttribute(self.ATTR_PLANE_NORMAL))
            )
            sourceViewName = contour.GetAttribute(self.ATTR_VIEW)
            editableOnSourcePlane = False
            for viewId, (viewName, origin, normal) in slicePlanes.items():
                if viewName != sourceViewName:
                    continue
                parallel = abs(float(np.dot(contourNormal, normal))) > 0.999
                distance = abs(float(np.dot(contourOrigin - origin, normal)))
                if parallel and distance < 0.25:
                    display2D.AddViewNodeID(viewId)
                    editableOnSourcePlane = True
            shouldLock = not editableOnSourcePlane
            if bool(contour.GetLocked()) != shouldLock:
                contour.SetLocked(shouldLock)

    def _contourSliceMask(self, node, volume, axis, index):
        dimensions = volume.GetImageData().GetDimensions()
        if node.GetNumberOfControlPoints() < 3:
            raise ValueError("轮廓至少需要三个控制点。")
        curve = node.GetCurveWorld()
        if not curve or curve.GetNumberOfPoints() < 3:
            raise ValueError("轮廓曲线尚未完成。")
        rasToIjk = vtk.vtkMatrix4x4()
        volume.GetRASToIJKMatrix(rasToIjk)
        points = []
        ras = [0.0, 0.0, 0.0]
        ijk = [0.0, 0.0, 0.0, 1.0]
        for pointIndex in range(curve.GetNumberOfPoints()):
            curve.GetPoint(pointIndex, ras)
            rasToIjk.MultiplyPoint([*ras, 1.0], ijk)
            points.append(np.array(ijk[:3], dtype=float))
        points = np.asarray(points)
        planeAxes = [item for item in range(3) if item != axis]
        width = dimensions[planeAxes[0]]
        height = dimensions[planeAxes[1]]
        polygon = vtk.vtkPoints()
        poly = vtk.vtkPolyData()
        lines = vtk.vtkCellArray()
        for point in points:
            polygon.InsertNextPoint(
                float(point[planeAxes[0]]),
                float(point[planeAxes[1]]),
                0.0,
            )
        count = polygon.GetNumberOfPoints()
        lines.InsertNextCell(count + 1)
        for pointIndex in range(count):
            lines.InsertCellPoint(pointIndex)
        lines.InsertCellPoint(0)
        poly.SetPoints(polygon)
        poly.SetLines(lines)

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(poly)
        stencil.SetOutputOrigin(0, 0, 0)
        stencil.SetOutputSpacing(1, 1, 1)
        stencil.SetOutputWholeExtent(0, width - 1, 0, height - 1, 0, 0)
        stencil.Update()
        stencilImage = vtk.vtkImageStencilToImage()
        stencilImage.SetInputConnection(stencil.GetOutputPort())
        stencilImage.SetInsideValue(1)
        stencilImage.SetOutsideValue(0)
        stencilImage.SetOutputScalarTypeToUnsignedChar()
        stencilImage.Update()
        output = stencilImage.GetOutput()
        scalars = output.GetPointData().GetScalars()
        return vtk_to_numpy(scalars).reshape(height, width).astype(bool)

    @staticmethod
    def _resampleClosedCurve(points, sampleCount):
        points = np.asarray(points, dtype=float)
        if len(points) < 3:
            raise ValueError("闭合曲线采样点不足。")
        if np.linalg.norm(points[0] - points[-1]) < 1e-6:
            points = points[:-1]
        closed = np.vstack([points, points[0]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = cumulative[-1]
        if total < 1e-6:
            raise ValueError("闭合曲线长度无效。")
        targets = np.linspace(0.0, total, sampleCount, endpoint=False)
        result = np.zeros((sampleCount, 3), dtype=float)
        segmentIndex = 0
        for outputIndex, target in enumerate(targets):
            while (
                segmentIndex + 1 < len(cumulative) - 1
                and cumulative[segmentIndex + 1] <= target
            ):
                segmentIndex += 1
            segmentLength = lengths[segmentIndex]
            fraction = (
                0.0
                if segmentLength < 1e-12
                else (target - cumulative[segmentIndex]) / segmentLength
            )
            result[outputIndex] = (
                closed[segmentIndex] * (1.0 - fraction)
                + closed[segmentIndex + 1] * fraction
            )
        return result

    @staticmethod
    def _ringNormal(points):
        normal = np.zeros(3, dtype=float)
        for index in range(len(points)):
            current = points[index]
            following = points[(index + 1) % len(points)]
            normal += np.cross(current, following)
        length = np.linalg.norm(normal)
        return normal / length if length > 1e-9 else normal

    def _alignedContourRings(self, contours):
        referenceNormal = self._normalize(
            self._parseVector(contours[0].GetAttribute(self.ATTR_PLANE_NORMAL))
        )
        ordered = []
        for node in contours:
            normal = self._normalize(
                self._parseVector(node.GetAttribute(self.ATTR_PLANE_NORMAL))
            )
            if abs(float(np.dot(normal, referenceNormal))) < math.cos(
                math.radians(2.5)
            ):
                raise ValueError(
                    "斜切体积生成要求所有关键层互相平行；请保持同一旋转角度后滚动切片。"
                )
            origin = self._parseVector(node.GetAttribute(self.ATTR_PLANE_ORIGIN))
            ordered.append((float(np.dot(origin, referenceNormal)), node))
        ordered.sort(key=lambda item: item[0])
        if ordered[-1][0] - ordered[0][0] < 1e-3:
            raise ValueError("至少需要两个空间位置不同的关键层轮廓。")

        sampleCount = 128
        rings = []
        for _position, node in ordered:
            curve = node.GetCurveWorld()
            if curve is None or curve.GetNumberOfPoints() < 3:
                raise ValueError("存在尚未完成的轮廓。")
            points = np.asarray(
                [curve.GetPoint(index) for index in range(curve.GetNumberOfPoints())],
                dtype=float,
            )
            ring = self._resampleClosedCurve(points, sampleCount)
            if float(np.dot(self._ringNormal(ring), referenceNormal)) < 0:
                ring = ring[::-1].copy()
            if rings:
                previous = rings[-1]
                costs = [
                    float(np.sum((previous - np.roll(ring, shift, axis=0)) ** 2))
                    for shift in range(sampleCount)
                ]
                ring = np.roll(ring, int(np.argmin(costs)), axis=0)
            rings.append(ring)
        return rings

    def _loftContoursToMask(self, contours, volume):
        """Build a watertight loft from parallel oblique contour rings."""
        rings = self._alignedContourRings(contours)
        rasToIjk = vtk.vtkMatrix4x4()
        volume.GetRASToIJKMatrix(rasToIjk)
        sampleCount = len(rings[0])
        points = vtk.vtkPoints()
        ijk4 = [0.0, 0.0, 0.0, 1.0]
        for ring in rings:
            for ras in ring:
                rasToIjk.MultiplyPoint([*ras, 1.0], ijk4)
                points.InsertNextPoint(ijk4[0], ijk4[1], ijk4[2])

        polygons = vtk.vtkCellArray()
        for ringIndex in range(len(rings) - 1):
            firstBase = ringIndex * sampleCount
            secondBase = (ringIndex + 1) * sampleCount
            for pointIndex in range(sampleCount):
                following = (pointIndex + 1) % sampleCount
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, firstBase + pointIndex)
                triangle.GetPointIds().SetId(1, secondBase + pointIndex)
                triangle.GetPointIds().SetId(2, secondBase + following)
                polygons.InsertNextCell(triangle)
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, firstBase + pointIndex)
                triangle.GetPointIds().SetId(1, secondBase + following)
                triangle.GetPointIds().SetId(2, firstBase + following)
                polygons.InsertNextCell(triangle)

        firstCenterId = points.InsertNextPoint(
            *np.mean(
                np.asarray(
                    [points.GetPoint(index) for index in range(sampleCount)]
                ),
                axis=0,
            )
        )
        lastBase = (len(rings) - 1) * sampleCount
        lastCenterId = points.InsertNextPoint(
            *np.mean(
                np.asarray(
                    [
                        points.GetPoint(lastBase + index)
                        for index in range(sampleCount)
                    ]
                ),
                axis=0,
            )
        )
        for pointIndex in range(sampleCount):
            following = (pointIndex + 1) % sampleCount
            firstCap = vtk.vtkTriangle()
            firstCap.GetPointIds().SetId(0, firstCenterId)
            firstCap.GetPointIds().SetId(1, following)
            firstCap.GetPointIds().SetId(2, pointIndex)
            polygons.InsertNextCell(firstCap)
            lastCap = vtk.vtkTriangle()
            lastCap.GetPointIds().SetId(0, lastCenterId)
            lastCap.GetPointIds().SetId(1, lastBase + pointIndex)
            lastCap.GetPointIds().SetId(2, lastBase + following)
            polygons.InsertNextCell(lastCap)

        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(polygons)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(surface)
        clean.Update()

        dimensions = volume.GetImageData().GetDimensions()
        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputConnection(clean.GetOutputPort())
        stencil.SetOutputOrigin(0.0, 0.0, 0.0)
        stencil.SetOutputSpacing(1.0, 1.0, 1.0)
        stencil.SetOutputWholeExtent(
            0,
            dimensions[0] - 1,
            0,
            dimensions[1] - 1,
            0,
            dimensions[2] - 1,
        )
        stencil.Update()
        stencilImage = vtk.vtkImageStencilToImage()
        stencilImage.SetInputConnection(stencil.GetOutputPort())
        stencilImage.SetInsideValue(1)
        stencilImage.SetOutsideValue(0)
        stencilImage.SetOutputScalarTypeToUnsignedChar()
        stencilImage.Update()
        scalars = stencilImage.GetOutput().GetPointData().GetScalars()
        mask = vtk_to_numpy(scalars).reshape(
            dimensions[2], dimensions[1], dimensions[0]
        )
        if not np.any(mask):
            raise RuntimeError("斜切轮廓放样后得到空体积，请检查轮廓是否闭合且位于影像范围内。")
        return np.asarray(mask, dtype=np.uint8)

    def _generateDirectionMask(self, contours, volume, axis):
        """Generate one candidate volume from a parallel contour direction."""
        volumeAxes = self.volumeDirectionAxes(volume)
        oblique = any(
            abs(
                float(
                    np.dot(
                        self._normalize(
                            self._parseVector(
                                node.GetAttribute(self.ATTR_PLANE_NORMAL)
                            )
                        ),
                        volumeAxes[axis],
                    )
                )
            )
            < math.cos(math.radians(2.5))
            for node in contours
        )
        dimensions = volume.GetImageData().GetDimensions()
        if len(contours) == 1:
            if oblique:
                raise ValueError(
                    "单个斜切面轮廓不能独立生成空间约束；"
                    "请在相同斜切方向再绘制一个层面。"
                )
            node = contours[0]
            index = int(node.GetAttribute(self.ATTR_SLICE_INDEX))
            sliceMask = self._contourSliceMask(node, volume, axis, index)
            mask = np.zeros(tuple(reversed(dimensions)), dtype=np.uint8)
            moved = np.moveaxis(mask, 2 - axis, 0)
            moved[:] = sliceMask.astype(np.uint8)
            return mask
        if oblique:
            mask = self._loftContoursToMask(contours, volume)
        else:
            keyMasks = {}
            for node in contours:
                index = int(node.GetAttribute(self.ATTR_SLICE_INDEX))
                sliceMask = self._contourSliceMask(node, volume, axis, index)
                keyMasks[index] = np.logical_or(
                    keyMasks.get(index, False), sliceMask
                )
            ordered = sorted(keyMasks)
            if len(ordered) < 2:
                raise ValueError("至少需要两个不同层面的轮廓。")

            volumeNumpyShape = tuple(reversed(dimensions))
            mask = np.zeros(volumeNumpyShape, dtype=np.uint8)
            numpyAxis = 2 - axis
            moved = np.moveaxis(mask, numpyAxis, 0)
            for index, sliceMask in keyMasks.items():
                moved[index] = np.maximum(moved[index], sliceMask)
            for first, second in zip(ordered[:-1], ordered[1:]):
                if second <= first + 1:
                    continue
                firstMask = moved[first].astype(bool)
                secondMask = moved[second].astype(bool)
                firstSdf = ndimage.distance_transform_edt(firstMask) - (
                    ndimage.distance_transform_edt(~firstMask)
                )
                secondSdf = ndimage.distance_transform_edt(secondMask) - (
                    ndimage.distance_transform_edt(~secondMask)
                )
                for index in range(first + 1, second):
                    alpha = (index - first) / float(second - first)
                    moved[index] = (
                        (1.0 - alpha) * firstSdf + alpha * secondSdf >= 0
                    )
        return np.asarray(mask, dtype=np.uint8)

    def generateSegmentation(self, volume, group, segmentation=None):
        self.lastFusionUsedFallback = False
        contours = self.contourNodes(volume, group, segmentation)
        if len(contours) < 2:
            raise ValueError("至少需要绘制两个关键层轮廓。")

        contoursByAxis = {}
        for node in contours:
            axis = int(node.GetAttribute(self.ATTR_AXIS))
            contoursByAxis.setdefault(axis, []).append(node)

        directionMasks = []
        for axis, directionContours in sorted(contoursByAxis.items()):
            directionMasks.append(
                self._generateDirectionMask(directionContours, volume, axis)
            )
        if len(directionMasks) == 1:
            mask = directionMasks[0]
        else:
            votes = np.sum(
                np.stack(directionMasks, axis=0).astype(np.uint8), axis=0
            )
            requiredVotes = len(directionMasks) // 2 + 1
            mask = (votes >= requiredVotes).astype(np.uint8)
            if not np.any(mask):
                logging.info(
                    "MuscleContours multi-direction consensus is empty; "
                    "falling back to union fusion for group '%s'.",
                    group,
                )
                self.lastFusionUsedFallback = True
                mask = (votes >= 1).astype(np.uint8)

        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "{}_InterpolatedLabelmap".format(group)
        )
        try:
            slicer.util.updateVolumeFromArray(labelmap, mask)
            ijkToRas = vtk.vtkMatrix4x4()
            volume.GetIJKToRASMatrix(ijkToRas)
            labelmap.SetIJKToRASMatrix(ijkToRas)
            if segmentation is None:
                segmentation = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", "MuscleContoursSegmentation"
                )
                segmentation.CreateDefaultDisplayNodes()
                segmentation.SetReferenceImageGeometryParameterFromVolumeNode(
                    volume
                )
            segmentColor = self.colorForSegmentationGroup(
                segmentation, group
            )
            existingIds = vtk.vtkStringArray()
            segmentation.GetSegmentation().GetSegmentIDs(existingIds)
            idsBefore = {
                existingIds.GetValue(index)
                for index in range(existingIds.GetNumberOfValues())
            }
            success = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap, segmentation
            )
            if not success:
                raise RuntimeError("无法将插值标签图导入分割节点。")
            segmentIds = vtk.vtkStringArray()
            segmentation.GetSegmentation().GetSegmentIDs(segmentIds)
            newIds = [
                segmentIds.GetValue(index)
                for index in range(segmentIds.GetNumberOfValues())
                if segmentIds.GetValue(index) not in idsBefore
            ]
            if not newIds:
                raise RuntimeError("标签图已导入，但没有创建新的分割片段。")
            segmentId = newIds[-1]
            segment = segmentation.GetSegmentation().GetSegment(segmentId)
            segment.SetName(group)
            segment.SetColor(*segmentColor)
            for oldId in idsBefore:
                oldSegment = segmentation.GetSegmentation().GetSegment(oldId)
                if oldSegment and oldSegment.GetName() == group:
                    segmentation.GetSegmentation().RemoveSegment(oldId)
            segmentation.CreateClosedSurfaceRepresentation()
            return segmentation, segmentId
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)


class SlicerMuscleContoursTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("SlicerMuscleContours smoke test")
