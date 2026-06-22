import os

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

# ---------------------------------------------------------------------------
# User-facing Chinese strings are stored as ASCII \u escapes so the source
# stays ASCII-only (CI check-utf8.yml rejects any non-ASCII byte in *.py),
# while the UI still renders Chinese at runtime.
# ---------------------------------------------------------------------------
STRINGS = {
    "module_title": "\u9010\u5c42\u7ed8\u5236\u5e76\u751f\u6210VR",
    "module_help": "\u5728\u5207\u7247\u4e0a\u9010\u5c42\u52fe\u753b\u95ed\u5408\u8f6e\u5ed3\uff0c\u81ea\u52a8\u5c42\u95f4\u63d2\u503c\u5e76\u751f\u6210\u5e73\u6ed1\u76843D\u8868\u9762\u6a21\u578b\u4e0e\u53ef\u9009\u4f53\u79ef\u6e32\u67d3\u3002",
    "grp_input": "\u8f93\u5165",
    "lbl_volume": "\u80cc\u666f\u4f53\u79ef",
    "lbl_seg": "\u5206\u5272\u8282\u70b9",
    "grp_draw": "\u9010\u5c42\u7ed8\u5236",
    "btn_start": "\u5f00\u59cb\u9010\u5c42\u7ed8\u5236",
    "btn_stop": "\u505c\u6b62\u7ed8\u5236",
    "btn_prev": "\u4e0a\u4e00\u5c42",
    "btn_next": "\u4e0b\u4e00\u5c42",
    "btn_undo": "\u64a4\u9500\u4e0a\u4e00\u6b21\u7ed8\u5236",
    "grp_gen": "\u751f\u6210\u4e0e\u5bfc\u51fa",
    "lbl_method": "\u5c42\u95f4\u63d2\u503c\u65b9\u6cd5",
    "method_builtin": "Slicer\u5185\u7f6e (Fill between slices)",
    "method_scipy": "\u5f62\u72b6\u63d2\u503c (scipy\u8ddd\u79bb\u573a)",
    "lbl_smooth": "\u8868\u9762\u5e73\u6ed1\u8fed\u4ee3\u6b21\u6570",
    "chk_vr": "\u540c\u65f6\u542f\u7528\u80cc\u666f\u4f53\u79ef\u6e32\u67d3",
    "btn_gen": "\u5b8c\u6210\u7ed8\u5236\u5e76\u751f\u6210VR",
    "btn_export": "\u5bfc\u51fa\u6a21\u578b\u4e3aSTL",
    "lbl_status": "\u72b6\u6001",
    "status_ready": "\u5c31\u7eea",
    "msg_need_volume": "\u8bf7\u5148\u9009\u62e9\u80cc\u666f\u4f53\u79ef\u3002",
    "msg_drawing_on": "\u7ed8\u5236\u6a21\u5f0f\u5df2\u5f00\u542f\uff1a\u5728\u5207\u7247\u89c6\u56fe\u7528\u5de6\u952e\u70b9\u51fb\u52fe\u753b\u95ed\u5408\u8f6e\u5ed3\uff0c\u53cc\u51fb\u6216\u56de\u8f66\u95ed\u5408\u5e76\u586b\u5145\uff1b\u6eda\u8f6e\u6216\u4e0a\u4e00\u5c42/\u4e0b\u4e00\u5c42\u6309\u94ae\u5207\u6362\u5c42\u9762\u3002",
    "msg_drawing_off": "\u7ed8\u5236\u6a21\u5f0f\u5df2\u5173\u95ed\u3002",
    "msg_no_slices": "\u5c1a\u672a\u7ed8\u5236\u4efb\u4f55\u5207\u7247\uff0c\u65e0\u6cd5\u751f\u6210\u3002\u8bf7\u5148\u81f3\u5c11\u7ed8\u5236\u4e00\u5c42\u3002",
    "title_warn": "\u63d0\u793a",
    "title_err": "\u9519\u8bef",
    "msg_large_gap": "\u68c0\u6d4b\u5230\u5c42\u95f4\u8ddd\u8f83\u5927(\u7ea6{0}mm)\uff0c\u63d2\u503c\u7ed3\u679c\u53ef\u80fd\u4e0d\u7cbe\u786e\uff0c\u5efa\u8bae\u7f29\u5c0f\u5c42\u8ddd\u6216\u591a\u7ed8\u5236\u51e0\u5c42\u3002\u662f\u5426\u7ee7\u7eed?",
    "step_interp": "\u6b63\u5728\u8fdb\u884c\u5c42\u95f4\u63d2\u503c...",
    "step_model": "\u6b63\u5728\u751f\u6210\u8868\u9762\u6a21\u578b...",
    "step_smooth": "\u6b63\u5728\u5e73\u6ed1\u6a21\u578b...",
    "step_vr": "\u6b63\u5728\u542f\u7528\u4f53\u79ef\u6e32\u67d3...",
    "step_done": "\u5b8c\u6210\uff0c\u5df2\u751f\u6210\u6a21\u578b: {0}",
    "msg_no_model": "\u6ca1\u6709\u53ef\u5bfc\u51fa\u7684\u6a21\u578b\uff0c\u8bf7\u5148\u751f\u6210VR\u3002",
    "msg_export_done": "\u5df2\u5bfc\u51fa: {0}",
    "msg_interp_fallback": "\u5185\u7f6e\u63d2\u503c\u4e0d\u53ef\u7528\uff0c\u5df2\u56de\u9000\u5230scipy\u5f62\u72b6\u63d2\u503c\u3002",
    "msg_no_scipy": "scipy\u4e0d\u53ef\u7528\uff0c\u65e0\u6cd5\u4f7f\u7528\u5f62\u72b6\u63d2\u503c\uff0c\u8bf7\u6539\u7528\u5185\u7f6e\u65b9\u6cd5\u3002",
    "msg_one_slice": "\u4ec5\u7ed8\u5236\u4e86\u4e00\u5c42\uff0c\u8df3\u8fc7\u5c42\u95f4\u63d2\u503c\uff0c\u76f4\u63a5\u751f\u6210\u8be5\u5c42\u6a21\u578b\u3002",
    "seg_default_name": "\u9010\u5c42\u8f6e\u5ed3",
    "msg_gen_fail": "\u751f\u6210\u5931\u8d25: {0}",
}

SETTINGS_NS = "SliceDrawVR/"
LARGE_GAP_MM = 10.0


def tr(key):
    return STRINGS.get(key, key)


class SliceDrawVR(ScriptedLoadableModule):
    """Module metadata."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = tr("module_title")
        parent.categories = ["Segmentation"]
        parent.dependencies = []
        parent.contributors = ["DicomProject (secondary development)"]
        parent.helpText = tr("module_help")
        parent.acknowledgementText = ""


class SliceDrawVRWidget(ScriptedLoadableModuleWidget):
    """The module panel."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        # Hidden background segment editor used to drive the Draw effect.
        self.segmentEditorNode = None
        self.segmentEditorWidget = None
        self.drawing = False
        # Generation step-machine state.
        self._genSteps = []
        self._genIndex = 0
        self._genContext = {}

    # -- lifecycle ----------------------------------------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SliceDrawVRLogic()
        self._buildGui()
        self._createSegmentEditor()
        self.updateButtons()

    def _buildGui(self):
        layout = self.layout

        # --- Input group ---
        inputBox = qt.QGroupBox(tr("grp_input"))
        form = qt.QFormLayout(inputBox)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.showHidden = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow(tr("lbl_volume"), self.volumeSelector)

        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segSelector.selectNodeUponCreation = True
        self.segSelector.addEnabled = True
        self.segSelector.removeEnabled = True
        self.segSelector.renameEnabled = True
        self.segSelector.noneEnabled = True
        self.segSelector.showHidden = False
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow(tr("lbl_seg"), self.segSelector)
        layout.addWidget(inputBox)

        # --- Draw group ---
        drawBox = qt.QGroupBox(tr("grp_draw"))
        drawLayout = qt.QVBoxLayout(drawBox)

        self.startButton = qt.QPushButton(tr("btn_start"))
        self.startButton.checkable = True
        self.startButton.toolTip = tr("msg_drawing_on")
        drawLayout.addWidget(self.startButton)

        navRow = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton(tr("btn_prev"))
        self.nextButton = qt.QPushButton(tr("btn_next"))
        navRow.addWidget(self.prevButton)
        navRow.addWidget(self.nextButton)
        drawLayout.addLayout(navRow)

        self.undoButton = qt.QPushButton(tr("btn_undo"))
        drawLayout.addWidget(self.undoButton)
        layout.addWidget(drawBox)

        # --- Generate group ---
        genBox = qt.QGroupBox(tr("grp_gen"))
        genForm = qt.QFormLayout(genBox)

        self.methodCombo = qt.QComboBox()
        self.methodCombo.addItem(tr("method_builtin"), "builtin")
        self.methodCombo.addItem(tr("method_scipy"), "scipy")
        genForm.addRow(tr("lbl_method"), self.methodCombo)

        self.smoothSpin = qt.QSpinBox()
        self.smoothSpin.minimum = 0
        self.smoothSpin.maximum = 60
        self.smoothSpin.value = int(self._setting("smooth_iters", 18))
        genForm.addRow(tr("lbl_smooth"), self.smoothSpin)

        self.vrCheck = qt.QCheckBox(tr("chk_vr"))
        self.vrCheck.checked = self._setting("enable_vr", "false") in (True, "true", "True", 1)
        genForm.addRow("", self.vrCheck)

        self.generateButton = qt.QPushButton(tr("btn_gen"))
        genForm.addRow(self.generateButton)

        self.progressBar = qt.QProgressBar()
        self.progressBar.minimum = 0
        self.progressBar.maximum = 100
        self.progressBar.value = 0
        genForm.addRow(self.progressBar)

        self.exportButton = qt.QPushButton(tr("btn_export"))
        genForm.addRow(self.exportButton)
        layout.addWidget(genBox)

        # --- Status ---
        self.statusLabel = qt.QLabel(tr("status_ready"))
        self.statusLabel.wordWrap = True
        layout.addWidget(self.statusLabel)
        layout.addStretch(1)

        # --- Connections ---
        self.startButton.connect("toggled(bool)", self.onStartToggled)
        self.prevButton.connect("clicked()", lambda: self.onStep(-1))
        self.nextButton.connect("clicked()", lambda: self.onStep(1))
        self.undoButton.connect("clicked()", self.onUndo)
        self.generateButton.connect("clicked()", self.onGenerate)
        self.exportButton.connect("clicked()", self.onExport)
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtons)
        self.segSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateButtons)

    def _createSegmentEditor(self):
        self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        try:
            self.segmentEditorWidget.setUndoEnabled(True)
        except Exception:
            pass
        self.segmentEditorWidget.hide()

    def cleanup(self):
        self._setActiveEffect(None)
        if self.segmentEditorWidget is not None:
            self.segmentEditorWidget.setMRMLScene(None)
            self.segmentEditorWidget = None
        if self.segmentEditorNode is not None and slicer.mrmlScene.IsNodePresent(self.segmentEditorNode):
            slicer.mrmlScene.RemoveNode(self.segmentEditorNode)
        self.segmentEditorNode = None

    # -- settings helpers ---------------------------------------------------
    def _setting(self, key, default):
        value = qt.QSettings().value(SETTINGS_NS + key)
        return default if value is None else value

    def _saveSetting(self, key, value):
        qt.QSettings().setValue(SETTINGS_NS + key, value)

    # -- state helpers ------------------------------------------------------
    def _currentVolume(self):
        return self.volumeSelector.currentNode()

    def _ensureSegmentation(self):
        """Return the segmentation node, creating one if needed."""
        segNode = self.segSelector.currentNode()
        if segNode is None:
            segNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", tr("seg_default_name"))
            segNode.CreateDefaultDisplayNodes()
            self.segSelector.setCurrentNode(segNode)
        volume = self._currentVolume()
        if volume is not None:
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        return segNode

    def _ensureSegment(self, segNode):
        """Return a selected segment id, creating a default segment if empty."""
        seg = segNode.GetSegmentation()
        selectedId = self.segmentEditorNode.GetSelectedSegmentID()
        if selectedId and seg.GetSegment(selectedId) is not None:
            return selectedId
        if seg.GetNumberOfSegments() == 0:
            selectedId = segNode.GetSegmentation().AddEmptySegment("", tr("seg_default_name"))
        else:
            selectedId = seg.GetNthSegmentID(0)
        self.segmentEditorNode.SetSelectedSegmentID(selectedId)
        return selectedId

    def _setMasterVolume(self, volume):
        # API renamed across Slicer versions (master -> source).
        if hasattr(self.segmentEditorWidget, "setSourceVolumeNode"):
            self.segmentEditorWidget.setSourceVolumeNode(volume)
        else:
            self.segmentEditorWidget.setMasterVolumeNode(volume)

    def _setActiveEffect(self, name):
        if self.segmentEditorWidget is None:
            return None
        if name is None:
            self.segmentEditorWidget.setActiveEffectByName("None")
            return None
        self.segmentEditorWidget.setActiveEffectByName(name)
        return self.segmentEditorWidget.activeEffect()

    # -- UI callbacks -------------------------------------------------------
    def updateButtons(self, *args):
        hasVolume = self._currentVolume() is not None
        self.startButton.enabled = hasVolume
        busy = bool(self._genSteps)
        self.generateButton.enabled = hasVolume and not busy
        self.exportButton.enabled = self.logic.lastModelNode() is not None
        for btn in (self.prevButton, self.nextButton, self.undoButton):
            btn.enabled = self.drawing

    def _setStatus(self, text):
        self.statusLabel.text = text
        slicer.app.processEvents()

    def onStartToggled(self, checked):
        if checked and self._currentVolume() is None:
            self.startButton.setChecked(False)
            slicer.util.warningDisplay(tr("msg_need_volume"), tr("title_warn"))
            return
        self.drawing = checked
        if checked:
            volume = self._currentVolume()
            segNode = self._ensureSegmentation()
            self._setMasterVolume(volume)
            self.segmentEditorWidget.setSegmentationNode(segNode)
            self._ensureSegment(segNode)
            self._setActiveEffect("Draw")
            self.startButton.text = tr("btn_stop")
            self._setStatus(tr("msg_drawing_on"))
        else:
            self._setActiveEffect(None)
            self.startButton.text = tr("btn_start")
            self._setStatus(tr("msg_drawing_off"))
        self.updateButtons()

    def onStep(self, direction):
        """Advance the Red slice view by one slice toward +/- direction."""
        volume = self._currentVolume()
        if volume is None:
            return
        try:
            sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
            sliceLogic = sliceWidget.sliceLogic()
            spacing = min(volume.GetSpacing())
            sliceLogic.SetSliceOffset(sliceLogic.GetSliceOffset() + direction * spacing)
        except Exception:
            pass

    def onUndo(self):
        try:
            self.segmentEditorWidget.undo()
        except Exception:
            pass

    # -- generation ---------------------------------------------------------
    def onGenerate(self):
        if self._genSteps:
            return
        volume = self._currentVolume()
        if volume is None:
            slicer.util.warningDisplay(tr("msg_need_volume"), tr("title_warn"))
            return
        # Make sure we have a segmentation/segment to read from.
        if self.drawing:
            self.startButton.setChecked(False)
        segNode = self.segSelector.currentNode()
        if segNode is None:
            slicer.util.warningDisplay(tr("msg_no_slices"), tr("title_warn"))
            return
        segmentId = self.segmentEditorNode.GetSelectedSegmentID()
        if not segmentId or segNode.GetSegmentation().GetSegment(segmentId) is None:
            if segNode.GetSegmentation().GetNumberOfSegments() == 0:
                slicer.util.warningDisplay(tr("msg_no_slices"), tr("title_warn"))
                return
            segmentId = segNode.GetSegmentation().GetNthSegmentID(0)

        # Analyze drawn slices (count, interpolation axis, max gap in mm).
        info = self.logic.analyzeDrawnSlices(segNode, segmentId, volume)
        if info["count"] == 0:
            slicer.util.warningDisplay(tr("msg_no_slices"), tr("title_warn"))
            return
        if info["count"] >= 2 and info["maxGapMm"] > LARGE_GAP_MM:
            msg = tr("msg_large_gap").format(round(info["maxGapMm"], 1))
            if not slicer.util.confirmYesNoDisplay(msg, tr("title_warn")):
                return

        method = self.methodCombo.itemData(self.methodCombo.currentIndex)
        self._saveSetting("smooth_iters", self.smoothSpin.value)
        self._saveSetting("enable_vr", "true" if self.vrCheck.checked else "false")

        self._genContext = {
            "volume": volume,
            "segNode": segNode,
            "segmentId": segmentId,
            "method": method,
            "needInterp": info["count"] >= 2,
            "smoothIters": self.smoothSpin.value,
            "enableVr": self.vrCheck.checked,
        }
        if info["count"] < 2:
            self._setStatus(tr("msg_one_slice"))

        self._genSteps = [
            (tr("step_interp"), 25, self._stepInterpolate),
            (tr("step_model"), 55, self._stepModel),
            (tr("step_smooth"), 80, self._stepSmooth),
            (tr("step_vr"), 95, self._stepVr),
            (None, 100, self._stepFinish),
        ]
        self._genIndex = 0
        self.progressBar.value = 0
        self.updateButtons()
        qt.QTimer.singleShot(0, self._runNextStep)

    def _runNextStep(self):
        if self._genIndex >= len(self._genSteps):
            self._genSteps = []
            self.updateButtons()
            return
        label, progress, fn = self._genSteps[self._genIndex]
        if label:
            self._setStatus(label)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            self._genSteps = []
            self.progressBar.value = 0
            self.updateButtons()
            slicer.util.errorDisplay(tr("msg_gen_fail").format(str(exc)), tr("title_err"))
            import traceback
            traceback.print_exc()
            return
        self.progressBar.value = progress
        self._genIndex += 1
        qt.QTimer.singleShot(10, self._runNextStep)

    def _stepInterpolate(self):
        ctx = self._genContext
        if not ctx["needInterp"]:
            return
        if ctx["method"] == "scipy":
            ok = self.logic.interpolateScipy(ctx["segNode"], ctx["segmentId"], ctx["volume"])
            if not ok:
                slicer.util.warningDisplay(tr("msg_no_scipy"), tr("title_warn"))
            return
        ok = self.logic.interpolateBuiltin(self.segmentEditorWidget, self.segmentEditorNode,
                                            ctx["segNode"], ctx["segmentId"], ctx["volume"],
                                            self._setMasterVolume)
        if not ok:
            self._setStatus(tr("msg_interp_fallback"))
            self.logic.interpolateScipy(ctx["segNode"], ctx["segmentId"], ctx["volume"])

    def _stepModel(self):
        ctx = self._genContext
        self.logic.exportModel(ctx["segNode"], ctx["segmentId"])

    def _stepSmooth(self):
        self.logic.smoothModel(self._genContext["smoothIters"])

    def _stepVr(self):
        ctx = self._genContext
        if ctx["enableVr"]:
            self.logic.enableVolumeRendering(ctx["volume"])

    def _stepFinish(self):
        model = self.logic.lastModelNode()
        name = model.GetName() if model is not None else "?"
        self._setStatus(tr("step_done").format(name))

    def onExport(self):
        model = self.logic.lastModelNode()
        if model is None:
            slicer.util.warningDisplay(tr("msg_no_model"), tr("title_warn"))
            return
        path = qt.QFileDialog.getSaveFileName(
            self.parent, tr("btn_export"), model.GetName() + ".stl", "STL (*.stl)")
        if not path:
            return
        slicer.util.saveNode(model, path)
        self._setStatus(tr("msg_export_done").format(os.path.basename(path)))


class SliceDrawVRLogic(ScriptedLoadableModuleLogic):
    """All MRML/VTK heavy lifting lives here."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._lastModelNodeId = None
        self._vrDisplayNodeId = None

    def lastModelNode(self):
        if self._lastModelNodeId is None:
            return None
        return slicer.mrmlScene.GetNodeByID(self._lastModelNodeId)

    # -- analysis -----------------------------------------------------------
    def _segmentArray(self, segNode, segmentId, volume):
        """Return the segment as a (z, y, x) uint8 array aligned to the volume.

        Returns (array, labelmapVolumeNode). Caller is responsible for
        removing the temporary labelmap volume node.
        """
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        segIds = vtk.vtkStringArray()
        segIds.InsertNextValue(segmentId)
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segNode, segIds, labelmapNode, volume)
        array = slicer.util.arrayFromVolume(labelmapNode)
        return array, labelmapNode

    def analyzeDrawnSlices(self, segNode, segmentId, volume):
        """Find which array axis the user drew along, how many slices and the
        largest physical gap between consecutive drawn slices."""
        array, labelmapNode = self._segmentArray(segNode, segmentId, volume)
        try:
            result = {"count": 0, "axis": 0, "maxGapMm": 0.0}
            if array is None or array.max() == 0:
                return result
            axis = self._drawingAxis(array)
            other = tuple(a for a in range(3) if a != axis)
            nonEmpty = [i for i in range(array.shape[axis])
                        if array.take(i, axis=axis).any()]
            result["count"] = len(nonEmpty)
            result["axis"] = axis
            # array axis order is (z, y, x); spacing is (x, y, z).
            spacing = volume.GetSpacing()
            axisSpacingMm = spacing[2 - axis]
            maxGap = 0
            for a, b in zip(nonEmpty, nonEmpty[1:]):
                maxGap = max(maxGap, b - a)
            result["maxGapMm"] = maxGap * axisSpacingMm
            return result
        finally:
            slicer.mrmlScene.RemoveNode(labelmapNode)

    def _drawingAxis(self, array):
        """The drawing axis is the one whose non-empty indices are sparsest
        (largest fraction of empty slices between first and last)."""
        bestAxis = 0
        bestSparsity = -1.0
        for axis in range(3):
            nonEmpty = [i for i in range(array.shape[axis])
                        if array.take(i, axis=axis).any()]
            if len(nonEmpty) < 1:
                continue
            span = nonEmpty[-1] - nonEmpty[0] + 1
            sparsity = 1.0 - (len(nonEmpty) / float(span)) if span > 0 else 0.0
            if sparsity > bestSparsity:
                bestSparsity = sparsity
                bestAxis = axis
        return bestAxis

    # -- interpolation ------------------------------------------------------
    def interpolateBuiltin(self, editorWidget, editorNode, segNode, segmentId,
                           volume, setMaster):
        """Run the core 'Fill between slices' effect. Return True on success."""
        try:
            setMaster(volume)
            editorWidget.setSegmentationNode(segNode)
            editorNode.SetSelectedSegmentID(segmentId)
            editorWidget.setActiveEffectByName("Fill between slices")
            effect = editorWidget.activeEffect()
            if effect is None:
                return False
            effect.self().onPreview()
            effect.self().onApply()
            editorWidget.setActiveEffectByName("None")
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            try:
                editorWidget.setActiveEffectByName("None")
            except Exception:
                pass
            return False

    def interpolateScipy(self, segNode, segmentId, volume):
        """Shape-based (signed distance transform) interpolation fallback.

        Returns True if scipy was available and interpolation ran."""
        try:
            import numpy as np
            from scipy import ndimage
        except Exception:
            return False
        array, labelmapNode = self._segmentArray(segNode, segmentId, volume)
        try:
            if array is None or array.max() == 0:
                return True
            axis = self._drawingAxis(array)
            work = np.moveaxis(array, axis, 0).astype(np.uint8)
            nonEmpty = [i for i in range(work.shape[0]) if work[i].any()]
            spacing = volume.GetSpacing()
            inPlaneSpacing = [spacing[2 - a] for a in range(3) if a != axis]
            for a, b in zip(nonEmpty, nonEmpty[1:]):
                if b - a <= 1:
                    continue
                self._interpolatePair(work, a, b, inPlaneSpacing, np, ndimage)
            result = np.moveaxis(work, 0, axis)
            slicer.util.updateVolumeFromArray(labelmapNode, result)
            # Write the interpolated labelmap back into the *existing* segment
            # (REPLACE), instead of importing it as a brand new segment.
            segLogic = slicer.vtkSlicerSegmentationsModuleLogic
            orientedImage = segLogic.CreateOrientedImageDataFromVolumeNode(labelmapNode)
            segLogic.SetBinaryLabelmapToSegment(
                orientedImage, segNode, segmentId, segLogic.MODE_REPLACE)
            return True
        finally:
            slicer.mrmlScene.RemoveNode(labelmapNode)

    def _interpolatePair(self, work, a, b, inPlaneSpacing, np, ndimage):
        """Fill slices strictly between a and b by interpolating the signed
        distance transforms of the two bounding slices."""
        sdtA = self._signedDistance(work[a], inPlaneSpacing, np, ndimage)
        sdtB = self._signedDistance(work[b], inPlaneSpacing, np, ndimage)
        for i in range(a + 1, b):
            t = (i - a) / float(b - a)
            blended = (1.0 - t) * sdtA + t * sdtB
            work[i] = (blended < 0).astype(np.uint8)

    def _signedDistance(self, mask, spacing, np, ndimage):
        mask = mask.astype(bool)
        if not mask.any():
            # No foreground: large positive distance everywhere (outside).
            return np.full(mask.shape, 1e6, dtype=np.float32)
        if mask.all():
            return np.full(mask.shape, -1e6, dtype=np.float32)
        outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
        inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        return (outside - inside).astype(np.float32)

    # -- model --------------------------------------------------------------
    def exportModel(self, segNode, segmentId):
        # Reduce the previous model so re-generation does not pile up nodes.
        old = self.lastModelNode()
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
            self._lastModelNodeId = None

        segNode.CreateClosedSurfaceRepresentation()
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        folderItemId = shNode.CreateFolderItem(
            shNode.GetSceneItemID(), tr("module_title"))
        segIds = vtk.vtkStringArray()
        segIds.InsertNextValue(segmentId)
        slicer.modules.segmentations.logic().ExportSegmentsToModels(
            segNode, segIds, folderItemId)

        modelNode = None
        childIds = vtk.vtkIdList()
        shNode.GetItemChildren(folderItemId, childIds)
        for i in range(childIds.GetNumberOfIds()):
            node = shNode.GetItemDataNode(childIds.GetId(i))
            if node is not None and node.IsA("vtkMRMLModelNode"):
                modelNode = node
        if modelNode is None:
            raise RuntimeError("ExportSegmentsToModels produced no model node")
        self._lastModelNodeId = modelNode.GetID()
        return modelNode

    def smoothModel(self, iterations):
        modelNode = self.lastModelNode()
        if modelNode is None or iterations <= 0:
            return
        polyData = modelNode.GetPolyData()
        if polyData is None:
            return
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(polyData)
        smoother.SetNumberOfIterations(int(iterations))
        smoother.SetPassBand(0.1)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(normals.GetOutput())
        modelNode.SetAndObservePolyData(result)
        if modelNode.GetDisplayNode() is None:
            modelNode.CreateDefaultDisplayNodes()

    # -- volume rendering ---------------------------------------------------
    def enableVolumeRendering(self, volume):
        vrLogic = slicer.modules.volumerendering.logic()
        displayNode = None
        if self._vrDisplayNodeId is not None:
            displayNode = slicer.mrmlScene.GetNodeByID(self._vrDisplayNodeId)
        if displayNode is None:
            displayNode = vrLogic.CreateVolumeRenderingDisplayNode()
            slicer.mrmlScene.AddNode(displayNode)
            displayNode.UnRegister(vrLogic)
            self._vrDisplayNodeId = displayNode.GetID()
        vrLogic.UpdateDisplayNodeFromVolumeNode(displayNode, volume)
        volume.AddAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetVisibility(True)
        # Make sure a 3D view is shown.
        try:
            layoutManager = slicer.app.layoutManager()
            threeDWidget = layoutManager.threeDWidget(0)
            threeDWidget.threeDView().resetFocalPoint()
        except Exception:
            pass


class SliceDrawVRTest(ScriptedLoadableModuleTest):
    """Minimal self-test so Slicer's 'Reload and Test' does not error.

    Real regression tests would run inside Slicer with a sample volume and a
    scripted drawing sequence; kept light here on purpose.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_logic_instantiates()

    def test_logic_instantiates(self):
        self.delayDisplay("Starting SliceDrawVR logic smoke test")
        logic = SliceDrawVRLogic()
        self.assertIsNone(logic.lastModelNode())
        self.delayDisplay("SliceDrawVR logic smoke test passed")
