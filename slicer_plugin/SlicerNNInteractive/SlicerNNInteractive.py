import io
import os
import gzip
import math
import requests
import copy
import threading
import time

import importlib.util

import numpy as np
from pathlib import Path

import slicer
import qt
import vtk
from qt import QApplication, QPalette

from vtkmodules.util.numpy_support import vtk_to_numpy

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox


###############################################################################
# Decorators and utility functions
###############################################################################


DEBUG_MODE = False

PLUGIN_VERSION = "1.2.5-triplanar-noreg"
print("[nni] SlicerNNInteractive build loaded:", PLUGIN_VERSION)


def _perf_log(msg, flush=True):
    """Append a flushed trace line to a fixed file next to this module so the
    step sequence survives a hard crash (Slicer process killed) and can be
    inspected directly; also echoes to the console. `flush` is accepted only for
    call-site compatibility with the print() it replaces."""
    try:
        line = str(msg)
    except Exception:  # noqa: BLE001
        line = "<unprintable trace line>"
    try:
        path = os.path.join(os.path.dirname(__file__), "nni_triplanar_trace.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("%.3f %s\n" % (time.time(), line))
    except Exception:  # noqa: BLE001 - tracing must never break the app
        pass
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001
        pass


_perf_log("=== nni perf trace start %s @%.3f ===" % (PLUGIN_VERSION, time.time()))

# A volume counts as oblique (and gets rotated to its own acquisition plane) when
# any IJK axis tilts more than ~2.5 degrees off the nearest RAS axis. The test
# compares each axis' largest direction-cosine against cos(2.5deg).
OBLIQUE_COS_THRESHOLD = 0.999


PLANE_DISPLAY_SELECTOR_NAMES = {
    "Red": "cbRedDisplayVolume",
    "Yellow": "cbYellowDisplayVolume",
    "Green": "cbGreenDisplayVolume",
}

# Tri-planar 3D locator frames: one hidden line-only model per standard view,
# drawn in the 3D view to mark where each slice plane currently sits (the 3D
# counterpart of the 2D slice intersection lines). Names end with the usual
# "(do not touch)" marker so they read as internal scaffolding.
TRIPLANAR_FRAME_NODE_NAMES = {
    "Red": "TriPlanarSliceFrameRed (do not touch)",
    "Yellow": "TriPlanarSliceFrameYellow (do not touch)",
    "Green": "TriPlanarSliceFrameGreen (do not touch)",
}
# Fallback RGB (0..1) per view when a slice node does not expose GetLayoutColor;
# values mirror Slicer's standard Red/Yellow/Green slice colors so the 3D frame
# matches the 2D intersection line color.
TRIPLANAR_FRAME_FALLBACK_COLORS = {
    "Red": (0.952, 0.297, 0.252),
    "Yellow": (0.952, 0.871, 0.255),
    "Green": (0.426, 0.748, 0.270),
}
# Opacity of the filled locator plane (the colored border is drawn on top via
# the same model's line cells). Low enough to see the segmentation through it.
TRIPLANAR_FRAME_OPACITY = 0.25
# Preferred camera side (RAS unit vector) per view for auto-rotation: which side
# the camera sits on when facing that series. Red=Superior(+S), Yellow=Anterior
# (+A), Green=Left(-R). The actual view direction is the series' slice-plane
# normal (oblique-aware); this only disambiguates the normal's sign so the
# viewpoint matches the conventional top/front/left expectation.
TRIPLANAR_CAMERA_PREFERRED_SIDE = {
    "Red": (0.0, 0.0, 1.0),
    "Yellow": (0.0, 1.0, 0.0),
    "Green": (-1.0, 0.0, 0.0),
}

# Hidden, geometry-only scalar volume that defines the canonical segmentation
# output grid when the high-resolution output feature is enabled.
OUTPUT_GEOMETRY_NODE_NAME = "NNInteractiveOutputGeometry (do not touch)"
# Voxel-count guardrails for the isotropic output grid (kept opt-in + bounded).
OUTPUT_GEOMETRY_SOFT_VOXEL_BUDGET = 50_000_000
OUTPUT_GEOMETRY_HARD_VOXEL_BUDGET = 150_000_000
# Tri-planar fusion runs SDF + resample + render on the output grid after EVERY
# interaction, so cap it well below the single-shot high-res budget to keep each
# operation light (load reduction, not RAM; tunable).
TRIPLANAR_MAX_OUTPUT_VOXELS = 32_000_000
OUTPUT_SPACING_MIN_MM = 0.3
OUTPUT_SPACING_MAX_MM = 10.0
# Lasso is a single-slice 2D prompt. Curve points are rounded to int voxels, so
# a slice sitting near a voxel boundary can scatter them across two adjacent
# slices. Snap the flattest axis to one slice when its spread is within this
# many voxels; reject (truly oblique / multi-slice) when it exceeds it.
LASSO_SLICE_AXIS_MAX_SPREAD = 2

# One-click three-series fusion: number of positive point seeds derived from the
# lasso interior (deepest distance-transform point + interior samples) that are
# sent to each orthogonal (non-source) series instead of the degenerate lasso.
FUSE3_NUM_SEEDS = 5
# Extra margin (mm) added around the lasso bounding box to form the total ROI in
# which the three-series fusion arithmetic is performed.
FUSE3_ROI_MARGIN_MM = 5.0

# Hidden linear transform that aligns a supplemental series to the source volume
# when their DICOM frames of reference differ (auto-registration).
SERIES_ALIGNMENT_TRANSFORM_NODE_NAME = "NNInteractiveSeriesAlignment (do not touch)"
# A registered translation larger than this (mm) is flagged for visual review.
REGISTRATION_OFFSET_WARN_MM = 30.0
# Below these magnitudes a registration result is treated as identity (series
# already aligned) and the transform is discarded to avoid needless resampling.
REGISTRATION_IDENTITY_TRANSLATION_MM = 1.0
REGISTRATION_IDENTITY_ROTATION_DEG = 1.0

# Operand sources for Selection Operations (cbOperandSource item order).
OPERAND_SOURCE_ROI = 0
OPERAND_SOURCE_WAND = 1
OPERAND_SOURCE_SEGMENT = 2
OPERAND_SOURCE_LASSO3D = 3

# ROI operand shapes (cbRoiShape item order).
ROI_SHAPE_BOX = 0
ROI_SHAPE_SPHERE = 1
ROI_SHAPE_ELLIPSOID = 2

# QSettings keys (all under the SlicerNNInteractive/ namespace).
SETTING_SERVER = "SlicerNNInteractive/server"
SETTING_SNAP_SLICES = "SlicerNNInteractive/snap_slices_to_grid"
SETTING_OUTPUT_SPACING = "SlicerNNInteractive/output_spacing"
SETTING_SEGMENT_OPACITY = "SlicerNNInteractive/segment_opacity"
SETTING_DISPLAY_SMOOTH_ENABLED = "SlicerNNInteractive/display_smooth_enabled"
SETTING_DISPLAY_SMOOTH_STRENGTH = "SlicerNNInteractive/display_smooth_strength"
SETTING_LASSO_CLIP_ENABLED = "SlicerNNInteractive/lasso_clip_enabled"
SETTING_LASSO_CLIP_N = "SlicerNNInteractive/lasso_clip_n"
SETTING_LASSO_MULTIVIEW_ENABLED = "SlicerNNInteractive/lasso_multiview_enabled"
SETTING_HIGH_RES_ENABLED = "SlicerNNInteractive/high_res_output_enabled"
SETTING_SMOOTH_INTERPOLATE_ENABLED = "SlicerNNInteractive/smooth_interpolate_enabled"
SETTING_TRIPLANAR_ENABLED = "SlicerNNInteractive/triplanar_mode_enabled"
# Per-view visibility of the 3D locator planes (one key per view, suffixed with
# the lower-cased view name); default visible.
SETTING_TRIPLANAR_FRAME_VISIBLE_PREFIX = (
    "SlicerNNInteractive/triplanar_frame_visible_"
)
# Opacity of the filled locator planes (0..1), adjustable via a slider.
SETTING_PLANE_OPACITY = "SlicerNNInteractive/plane_opacity"
# Auto-rotate the 3D camera to face the series of the view being interacted with.
SETTING_AUTO_CAMERA_ROTATION = "SlicerNNInteractive/auto_camera_rotation"
# Show the segmentation's 3D closed surface while in tri-planar mode.
SETTING_SHOW_3D_TRIPLANAR = "SlicerNNInteractive/show_3d_triplanar"
# Align the 3D camera to a series' acquisition frame (its direction matrix
# composed with any registration transform) instead of the 2D slice plane.
SETTING_OBLIQUE_CAMERA_ALIGN = "SlicerNNInteractive/oblique_camera_align"
# Debounce (ms) for rebuilding the tri-planar 3D surface so rapid interactions
# trigger a single (heavy) marching-cubes pass after the user pauses.
TRIPLANAR_3D_SURFACE_DEBOUNCE_MS = 600


def debug_print(*args):
    if DEBUG_MODE:
        print(*args)


def ensure_synched(func):
    """
    Decorator that ensures the image and segment are synced before calling
    the actual prompt function.
    """

    def inner(self, *args, **kwargs):
        if getattr(self, "_alignment_in_progress", False):
            slicer.util.showStatusMessage(
                "Series registration in progress; prompt was not sent.", 4000
            )
            return

        failed_to_sync = False
        uploaded_image = False

        if self.image_changed(do_prev_image_update=False):
            debug_print(
                "Inference image changed (or not previously set). "
                "Calling upload_image_to_server()"
            )
            result = self.upload_image_to_server()
            failed_to_sync = result is None
            uploaded_image = not failed_to_sync

        if (
            not failed_to_sync
            and (uploaded_image or self.selected_segment_changed())
        ):
            debug_print(
                "Segment changed (or not previously set). Calling upload_segment_to_server()"
            )
            self.remove_all_but_last_prompt()
            result = self.upload_segment_to_server()

            failed_to_sync = result is None
        else:
            debug_print("Segment did not change!")

        if not failed_to_sync:
            return func(self, *args, **kwargs)

        slicer.util.showStatusMessage(
            "Sync to server failed; prompt was not sent.", 4000
        )

    return inner


###############################################################################
# SlicerNNInteractive
###############################################################################


class SlicerNNInteractive(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("nnInteractive")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Segmentation")
        ]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente", "Kiran Vaidhya Venkadesh", "Bram van Ginneken", "Clara I. Sanchez"]
        self.parent.helpText = """
            This is an 3D Slicer extension for using nnInteractive.

            Build: %s

            Read more about this plugin here: https://github.com/coendevente/SlicerNNInteractive.
            """ % PLUGIN_VERSION
        self.parent.acknowledgementText = """When using SlicerNNInteractive, please cite as described here: https://github.com/coendevente/SlicerNNInteractive?tab=readme-ov-file#citation."""


###############################################################################
# SlicerNNInteractiveWidget
###############################################################################


class SlicerNNInteractiveWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    # Every historical name a magic wand seed node has had. Used to sweep
    # orphans on Clear Seeds / setup so reloads do not leak stale fiducials.
    _WAND_SEED_NODE_NAMES = (
        "SelectionOpWandSeeds",          # current
        "SelectionOpWandSeedsPositive",  # multi-seed v1 (positive)
        "SelectionOpWandSeedsNegative",  # multi-seed v1 (negative)
        "SelectionOpWandSeed",           # original single-point
    )

    ###############################################################################
    # Setup and initialization functions
    ###############################################################################

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        # Operation ROI used as an alternative operand for Selection Operations.
        self._sel_op_roi_node = None
        # Dedicated ROI for the "Crop Segment by Box" tool (independent of the
        # Selection Operations operand ROI above).
        self._crop_roi_node = None
        # Sphere/ellipsoid visualization for the operation ROI.
        self._sel_op_roi_preview_node = None
        self._sel_op_roi_preview_transform_node = None
        # Magic wand seeds: a multi-point Fiducial node feeding nnInteractive's
        # point prompt.
        self._sel_op_wand_seed_node = None
        # Live preview of the magic wand region (hidden segmentation node).
        self._sel_op_wand_preview_segment_node = None
        self._wand_preview_segment_id = None
        # Lasso (3D) operand: a hidden Segment Editor runs the native Scissors
        # effect to draw a 3D region (input node); the nnInteractive lasso AI
        # refines it into the preview node. Mirrors the Magic wand seed/preview.
        self._sel_op_lasso3d_input_segment_node = None
        self._lasso3d_input_segment_id = None
        self._sel_op_lasso3d_preview_segment_node = None
        self._lasso3d_preview_segment_id = None
        self._lasso3d_editor_widget = None
        self._lasso3d_editor_node = None
        self._lasso3d_input_observer_tag = None
        self._lasso3d_in_update = False
        # Native-series inference: keep supplemental-series AI results in a
        # preview until the user explicitly merges them into the source grid.
        self._inference_preview_segment_node = None
        self._inference_preview_segment_id = None
        self._inference_result_working_mask = None
        self._inference_result_source_mask = None
        self._inference_preview_target_segment_id = None
        self._inference_preview_source_volume_id = None
        # Multi-series fusion: latest per-series AI result on the OUTPUT grid,
        # keyed by inference-volume id. SDF-averaged into one smooth mask on Fuse.
        self._fusion_results = {}        # volume_id -> uint8 mask (output grid)
        self._fusion_grid_shape = None   # output (z,y,x) shape the masks belong to
        # FOV coverage of each series on the OUTPUT grid (True where that series
        # actually images the voxel), bit-packed and keyed by volume id. Used so
        # a series only votes inside its own field of view during fusion.
        self._fusion_coverage = {}       # volume_id -> (packed bits, (z,y,x) shape)
        # One-click three-series fusion (onFuseFromThreeSeriesWithROI). While the
        # capture flag is set, server results are stashed on the output grid keyed
        # by inference-volume id (per series) instead of being displayed, so the
        # orchestration can collect each series' result and fuse them at the end.
        self._fusion_capture_active = False
        self._fusion_capture_store = {}  # volume_id -> uint8 mask (output grid)
        # World-space (RAS) curve points of the most recently submitted lasso, so
        # the three-series fusion button can reuse it after the live lasso node was
        # consumed/cleared by the normal auto-submit flow.
        self._last_lasso_world_points = None
        # Tri-planar multi-series mode: each interaction is routed to the series
        # shown in the view it was made in, and the three directions are
        # direction-weighted-fused after every interaction. While a prompt is
        # routed, _active_inference_volume_override forces get_inference_volume_node
        # to return that view's series.
        self._triplanar_mode = False
        self._active_inference_volume_override = None
        # 3D locator planes for tri-planar mode: {view_name: vtkMRMLModelNode}
        # and the slice-node ModifiedEvent observers that keep them in sync,
        # stored as (slice_node, tag) like the snap observers.
        self._triplanar_frame_nodes = {}
        self._triplanar_frame_observers = []
        # Overlay R/Y/G toggle buttons (top-left of the 3D view) for showing or
        # hiding each locator plane individually, plus their container.
        self._triplanar_frame_buttons = {}
        self._triplanar_frame_button_bar = None
        # Applying per-plane display volumes enables a sticky view override.
        # Hidden Segment Editor widgets may reset slice backgrounds while
        # activating effects, so restore the user's selections afterward.
        self._plane_display_volumes_active = False
        self._plane_display_reapply_pending = False
        # Snapshot locked in when the user clicks Apply: {slice_view_name: nodeID
        # or None}. None means that view follows the current source volume. The
        # sticky reapply consumes this snapshot, not the live selectors, so that
        # selector edits made without re-clicking Apply are not silently applied.
        self._plane_display_snapshot = None
        # High-resolution output geometry: an optional isotropic grid that the
        # canonical segment is stored on, decoupled from the coarse source grid.
        self._output_geometry_node = None
        self._output_geometry_spacing = None
        self._output_geometry_source_id = None
        # Tri-planar: the output grid is built to cover the UNION of the three
        # assigned series (not the editor source, which may be a different/
        # smaller series and would clip anatomy). Cache signature so it only
        # rebuilds when the series assignment / spacing actually changes.
        self._output_geometry_triplanar_sig = None
        # Selection Operations-private undo stack: list of
        # (segment_id, packbits_mask, shape). Snapshots are bit-packed to keep
        # memory bounded on large (high-resolution) output grids.
        self._sel_op_undo_stack = []
        self._sel_op_undo_stack_limit = 10
        # Series alignment: auto-registration of a supplemental series to the
        # source volume. Maps (supplemental_id, source_id) -> linear transform
        # node. While a registration CLI runs, prompts and sync are blocked.
        self._alignment_transforms = {}
        self._alignment_cli_node = None
        self._alignment_cli_observer = None
        self._alignment_pending = None
        self._alignment_in_progress = False
        # Pending (moving_id, source_id) pairs waiting for a free CLI slot. The
        # three multi-plane display volumes can each need their own registration.
        self._alignment_queue = []

    def setup(self):
        """
        Overridden setup method. Initializes UI and setups up prompts.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.install_dependencies()

        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SlicerNNInteractive.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        for selector_name in PLANE_DISPLAY_SELECTOR_NAMES.values():
            getattr(self.ui, selector_name).setMRMLScene(slicer.mrmlScene)
        self.ui.cbInferenceWorkingVolume.setMRMLScene(slicer.mrmlScene)
        self.scribble_segment_node_name = "ScribbleSegmentNode (do not touch)"
        self.wand_preview_segment_node_name = "MagicWandPreviewSegmentNode (do not touch)"
        self.lasso3d_input_segment_node_name = "Lasso3dInputSegmentNode (do not touch)"
        self.lasso3d_preview_segment_node_name = "Lasso3dPreviewSegmentNode (do not touch)"
        self.inference_preview_segment_node_name = (
            "NativeSeriesInferencePreviewSegmentNode (do not touch)"
        )
        self.output_geometry_node_name = OUTPUT_GEOMETRY_NODE_NAME
        self.series_alignment_transform_node_name = (
            SERIES_ALIGNMENT_TRANSFORM_NODE_NAME
        )

        # Set up editor widget
        self.ui.editor_widget.setMaximumNumberOfUndoStates(10)
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        # Use the same segmentation parameter node as the Segment Editor core module
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode")
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        self.ui.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.ui.editor_widget.setSegmentationNode(self.get_segmentation_node())

        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "button": self.ui.pbInteractionPoint,
                "button_text": self.ui.pbInteractionPoint.text,
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
                "button_icon_filename": "bbox_icon.svg",
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "button": self.ui.pbInteractionLasso,
                "button_text": self.ui.pbInteractionLasso.text,
                "button_icon_filename": "lasso_icon.svg",
            },
        }

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        self.setup_prompts()

        self.enable_slice_intersections()
        # Snap slice-view scrolling to each view's own voxel grid so the mouse
        # wheel lands on real slices, not interpolated in-between frames. State is
        # installed here from the persisted setting; init_ui_functionality only
        # reflects it in the checkbox (with signals blocked).
        self._slice_snap_observers = []
        self._snapping_in_progress = False
        self._snap_last_offset = {}
        self._refresh_slice_snap()

        # (numpy_axis, center) describing the slice plane of the last lasso
        # prompt. Set only by lasso_points_to_mask and consumed (reset to None)
        # by show_segmentation, so only lasso results get slice-range clipped.
        self._last_lasso_slice = None

        # Multi-view lasso accumulation: one mask per drawn plane. On submit
        # each is sent as a separate lasso interaction (the session accumulates
        # them), which matches the model's single-plane lasso training better
        # than OR-ing them into one cross-shaped mask.
        self._multiview_lasso_masks = []
        self._multiview_lasso_count = 0
        self._multiview_lasso_nodes = []  # accumulated lasso nodes kept visible

        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

        # Sweep any orphaned magic wand seed nodes left behind by earlier
        # versions / earlier reloads so the scene is clean on module load.
        self._destroy_wand_seed()

    @staticmethod
    def enable_slice_intersections():
        """Show the other slice planes in each 2D slice view."""
        for slice_display_node in slicer.util.getNodesByClass(
            "vtkMRMLSliceDisplayNode"
        ):
            slice_display_node.SetIntersectingSlicesVisibility(1)

        # Force an immediate visual refresh after changing display node state.
        for slice_node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            slice_node.Modified()

    @staticmethod
    def _set_slice_view_background(slice_view_name, volume_node):
        """Set one standard slice view's background volume."""
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return False

        slice_widget = layout_manager.sliceWidget(slice_view_name)
        if slice_widget is None:
            return False

        composite_node = slice_widget.mrmlSliceCompositeNode()
        if composite_node is None:
            return False

        composite_node.SetBackgroundVolumeID(
            volume_node.GetID() if volume_node is not None else None
        )
        return True

    @staticmethod
    def _get_qsetting(key, default, cast=None):
        """Read a QSettings value with an optional typed cast.

        cast=bool reproduces the str->bool handling Qt needs on platforms that
        store booleans as the strings "true"/"false". cast=int/float fall back
        to ``default`` when the stored value is not numeric. cast=None returns
        the raw stored value.
        """
        v = qt.QSettings().value(key, default)
        if cast is bool:
            if isinstance(v, str):
                return v.lower() == "true"
            return bool(v)
        if cast is int or cast is float:
            try:
                return cast(v)
            except (TypeError, ValueError):
                return default
        return v

    @staticmethod
    def _set_qsetting(key, value):
        """Persist a QSettings value under the SlicerNNInteractive/ namespace."""
        qt.QSettings().setValue(key, value)

    def _safe_add_observer(self, node, event, callback):
        """Add a VTKObservationMixin observer once (no-op if already present)."""
        if node is not None and not self.hasObserver(node, event, callback):
            self.addObserver(node, event, callback)

    def _safe_remove_observer(self, node, event, callback):
        """Remove a VTKObservationMixin observer if present (no-op otherwise)."""
        if node is not None and self.hasObserver(node, event, callback):
            self.removeObserver(node, event, callback)

    def _get_snap_slices_setting(self):
        """Read whether scrolling snaps to the original voxel grid. Default True."""
        return self._get_qsetting(SETTING_SNAP_SLICES, True, cast=bool)

    def _iter_standard_slice_logics(self):
        """Yield (view_name, sliceLogic, sliceNode) for the three standard views.

        Views not currently realized in the layout are skipped.
        """
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return
        for view_name in PLANE_DISPLAY_SELECTOR_NAMES:
            slice_widget = layout_manager.sliceWidget(view_name)
            if slice_widget is None:
                continue
            slice_logic = slice_widget.sliceLogic()
            slice_node = slice_widget.mrmlSliceNode()
            if slice_logic is None or slice_node is None:
                continue
            yield view_name, slice_logic, slice_node

    def _view_background_volume(self, view_name):
        """Return the background volume node for one slice view, or None."""
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return None
        slice_widget = layout_manager.sliceWidget(view_name)
        if slice_widget is None:
            return None
        composite_node = slice_widget.mrmlSliceCompositeNode()
        if composite_node is None:
            return None
        volume_id = composite_node.GetBackgroundVolumeID()
        if not volume_id:
            return None
        return slicer.mrmlScene.GetNodeByID(volume_id)

    def _volume_is_oblique(self, volume):
        """True when the volume's axes are not aligned with the RAS axes.

        Each column of the IJK-to-RAS direction matrix is a unit IJK axis. An
        axis-aligned volume has one component at magnitude 1 and the rest near 0
        in every column. We flag the volume oblique when any column's largest
        component drops below cos(threshold), i.e. that axis is tilted off RAS.
        """
        if volume is None:
            return False
        get_dirs = getattr(volume, "GetIJKToRASDirectionMatrix", None)
        if get_dirs is None:
            return False
        matrix = vtk.vtkMatrix4x4()
        get_dirs(matrix)
        for col in range(3):
            largest = max(abs(matrix.GetElement(row, col)) for row in range(3))
            if largest < OBLIQUE_COS_THRESHOLD:
                return True
        return False

    def _align_views_to_volume_planes(self):
        """Rotate oblique-series views to their acquisition plane, then snap.

        For an obliquely acquired series the standard RAS views reslice it at an
        angle, so scrolling never lands on the real acquired slices. Rotating each
        view to the volume plane straightens the image, aligns the slice-
        intersection cross to the real series axes, and makes one view step
        through the original slices. Axis-aligned volumes are left untouched
        (rotating would be a no-op and could override a manual orientation). After
        rotating, snap the offset onto the nearest slice center.
        """
        if getattr(self, "_snapping_in_progress", False):
            return
        _perf_log("[DEBUG triplanar.perf] align_views start")
        self._snapping_in_progress = True
        try:
            for view_name, slice_logic, slice_node in (
                self._iter_standard_slice_logics()
            ):
                volume = self._view_background_volume(view_name)
                rotated = False
                if volume is not None and self._volume_is_oblique(volume):
                    _perf_log("[DEBUG triplanar.perf] align_views: RotateToVolumePlane"
                              " view=%s vol=%s" % (view_name, volume.GetName()))
                    slice_node.RotateToVolumePlane(volume)
                    rotated = True
                _perf_log("[DEBUG triplanar.perf] align_views: snap view=%s rotated=%s"
                          % (view_name, rotated))
                slice_logic.SnapSliceOffsetToIJK()
                self._snap_last_offset[view_name] = slice_logic.GetSliceOffset()
        finally:
            self._snapping_in_progress = False
        _perf_log("[DEBUG triplanar.perf] align_views done")

    def _on_slice_node_modified(
        self, view_name, slice_logic, caller=None, event=None
    ):
        """Re-snap one slice view after its offset changed (guards recursion).

        SnapSliceOffsetToIJK modifies the slice node, which re-fires this same
        observer; the in-progress flag short-circuits that recursion. ModifiedEvent
        also fires for non-offset changes (field of view, orientation), so we only
        act when the offset actually moved since we last saw this view.

        Slicer's default scroll step (~1-2 mm) is often smaller than the voxel
        spacing (e.g. 5 mm), so a naive nearest-voxel snap bounces back to the
        same slice every tick. When that happens we detect the scroll direction and
        force exactly one voxel step so every wheel event advances one real slice.
        """
        if self._snapping_in_progress:
            return
        before = slice_logic.GetSliceOffset()
        last = self._snap_last_offset.get(view_name)
        if last is not None and abs(before - last) < 1e-4:
            # Offset unchanged; this Modified came from something else. Ignore.
            return
        direction = 1.0 if (last is None or before > last) else -1.0
        self._snapping_in_progress = True
        try:
            slice_logic.SnapSliceOffsetToIJK()
            snapped = slice_logic.GetSliceOffset()
            # If the snap landed back on the same slice (scroll delta was too
            # small to cross the half-voxel threshold), nudge one full step in
            # the scroll direction so every wheel tick advances exactly one layer.
            if last is not None and abs(snapped - last) < 1e-4:
                spacing = slice_logic.GetLowestVolumeSliceSpacing()
                step = spacing[2] if spacing and len(spacing) > 2 else 0.0
                if step > 1e-4:
                    slice_logic.SetSliceOffset(last + direction * step)
                    slice_logic.SnapSliceOffsetToIJK()
        finally:
            self._snapping_in_progress = False
        self._snap_last_offset[view_name] = slice_logic.GetSliceOffset()

    def _install_slice_snap_observers(self):
        """Observe the three standard slice nodes and snap offsets to the grid."""
        self._remove_slice_snap_observers()
        self._snap_last_offset = {}
        installed = []
        for view_name, slice_logic, slice_node in self._iter_standard_slice_logics():
            # Bind this view per-iteration so each callback snaps its own view
            # (default-arg capture avoids late binding in the loop).
            callback = (
                lambda caller, event, name=view_name, logic=slice_logic: (
                    self._on_slice_node_modified(name, logic, caller, event)
                )
            )
            tag = slice_node.AddObserver(vtk.vtkCommand.ModifiedEvent, callback)
            self._slice_snap_observers.append((slice_node, tag))
        # Align the current views at once so the effect is visible immediately.
        self._align_views_to_volume_planes()

    def _remove_slice_snap_observers(self):
        """Drop any installed slice-node snap observers."""
        for slice_node, tag in getattr(self, "_slice_snap_observers", []):
            if slice_node is not None:
                slice_node.RemoveObserver(tag)
        self._slice_snap_observers = []

    def _refresh_slice_snap(self):
        """Install or remove the snap observers to match the persisted setting."""
        if not hasattr(self, "_slice_snap_observers"):
            self._slice_snap_observers = []
        if not hasattr(self, "_snapping_in_progress"):
            self._snapping_in_progress = False
        if not hasattr(self, "_snap_last_offset"):
            self._snap_last_offset = {}
        enabled = self._get_snap_slices_setting()
        if enabled:
            self._install_slice_snap_observers()
        else:
            self._remove_slice_snap_observers()

    def _on_snap_slices_changed(self, checked):
        """Persist the snap-to-grid toggle and (un)install the observers."""
        self._set_qsetting(SETTING_SNAP_SLICES, bool(checked))
        self._refresh_slice_snap()
        if checked:
            slicer.util.showStatusMessage(
                "Slice scrolling now snaps to the original voxel grid.", 4000
            )
        else:
            slicer.util.showStatusMessage(
                "Slice scrolling restored to Slicer default.", 4000
            )

    def _apply_plane_display_volumes(self, show_status=False):
        """Apply the configured supplemental backgrounds to standard slice views."""
        source_volume = self.get_volume_node()
        if source_volume is None:
            if show_status:
                slicer.util.showStatusMessage(
                    "Load a source volume before configuring slice-view volumes.",
                    4000,
                )
            return False

        snapshot = self._plane_display_snapshot or {}
        missing_views = []
        for slice_view_name in PLANE_DISPLAY_SELECTOR_NAMES:
            volume_id = snapshot.get(slice_view_name)
            display_volume = source_volume
            if volume_id:
                # A snapshotted volume may have been removed from the scene; fall
                # back to the source volume instead of failing.
                display_volume = (
                    slicer.mrmlScene.GetNodeByID(volume_id) or source_volume
                )
            _perf_log("[DEBUG triplanar.perf] apply: set bg view=%s vol=%s" % (
                slice_view_name,
                display_volume.GetName() if display_volume is not None else None))
            if not self._set_slice_view_background(slice_view_name, display_volume):
                missing_views.append(slice_view_name)

        if missing_views:
            if show_status:
                slicer.util.showStatusMessage(
                    "Slice views unavailable: " + ", ".join(missing_views),
                    4000,
                )
            return False

        # A new background may be an oblique series; re-align views to it.
        # Changing the composite node does not fire the slice-node observer.
        if self._get_snap_slices_setting():
            _perf_log("[DEBUG triplanar.perf] apply: before align_views")
            self._align_views_to_volume_planes()
            _perf_log("[DEBUG triplanar.perf] apply: after align_views")

        if show_status:
            slicer.util.showStatusMessage(
                "Applied registered per-plane display volumes. "
                "Segmentation source volume is unchanged.",
                4000,
            )
        # In tri-planar mode the locator planes may not exist yet (e.g. the mode
        # was restored as CHECKED at startup, so the toggle->setup path that
        # builds them never ran). Now that series are applied, build them.
        self._ensure_triplanar_slice_frames("apply")
        return True

    def on_apply_plane_display_volumes_clicked(self, checked=False):
        """
        Show registered supplemental volumes in individual slice views.

        This only changes view backgrounds. Segmentation geometry, prompts, and
        server synchronization continue to use the Segment Editor source volume.
        """
        # Lock in the current selections so later sticky reapplies follow what was
        # applied, not whatever the selectors happen to show at reapply time.
        self._plane_display_snapshot = {}
        for slice_view_name, selector_name in PLANE_DISPLAY_SELECTOR_NAMES.items():
            selected = getattr(self.ui, selector_name).currentNode()
            self._plane_display_snapshot[slice_view_name] = (
                selected.GetID() if selected is not None else None
            )
        # Sticky mode is on only when the apply fully succeeded.
        self._plane_display_volumes_active = self._apply_plane_display_volumes(
            show_status=True
        )
        self._align_display_volumes()

    def _align_display_volumes(self):
        """Auto-register each per-plane display volume to the source volume.

        Display volumes only sharpen the 2D views, but a background drawn from a
        series with a different DICOM frame of reference would still be shown at
        the wrong physical location. Aligning them keeps the slice intersections
        honest. Registration is asynchronous; the slice view follows the
        volume's parent transform live, so backgrounds re-place themselves once
        each registration completes.
        """
        _perf_log("[DEBUG triplanar.perf] align_display start")
        # Tri-planar's precondition is co-registered series; and the user can
        # confirm alignment explicitly. In either case skip auto-registration
        # entirely -- it is unnecessary for aligned series and the async BRAINSFit
        # CLI on large oblique series can crash Slicer. Backgrounds are still
        # applied by _apply_plane_display_volumes; only registration is skipped.
        if self._triplanar_mode or (
            hasattr(self, "ui") and self.ui.cbConfirmSeriesAligned.isChecked()
        ):
            _perf_log("[DEBUG triplanar.perf] align_display: SKIP registration "
                      "(triplanar/confirmed; series assumed pre-aligned)")
            return
        source_volume = self.get_volume_node()
        if source_volume is None or not self._plane_display_snapshot:
            _perf_log("[DEBUG triplanar.perf] align_display: nothing to do")
            return
        # Drop transforms left over from a previous (now stale) source volume so
        # they are not orphaned when a new alignment overwrites the parent.
        self._prune_alignment_for_source(source_volume.GetID())
        seen = set()
        for volume_id in self._plane_display_snapshot.values():
            if not volume_id or volume_id == source_volume.GetID():
                continue
            if volume_id in seen:
                continue
            seen.add(volume_id)
            display_volume = slicer.mrmlScene.GetNodeByID(volume_id)
            if display_volume is not None:
                _perf_log("[DEBUG triplanar.perf] align_display: ensure_alignment "
                          "vol=%s" % display_volume.GetName())
                self._ensure_alignment(display_volume, source_volume)
        _perf_log("[DEBUG triplanar.perf] align_display done")

    def _reapply_plane_display_volumes_if_active(self):
        """Restore sticky per-plane backgrounds after Segment Editor activity."""
        self._plane_display_reapply_pending = False
        if self._plane_display_volumes_active:
            _perf_log("[DEBUG triplanar.perf] reapply fired")
            self._apply_plane_display_volumes(show_status=False)

    def _schedule_plane_display_reapply(self):
        """Restore per-plane backgrounds after the current Qt event completes."""
        if (
            not self._plane_display_volumes_active
            or self._plane_display_reapply_pending
        ):
            return
        self._plane_display_reapply_pending = True
        qt.QTimer.singleShot(0, self._reapply_plane_display_volumes_if_active)

    def on_reset_plane_display_volumes_clicked(self, checked=False):
        """Restore the Segment Editor source volume in all standard slice views."""
        source_volume = self.get_volume_node()
        if source_volume is None:
            slicer.util.showStatusMessage(
                "Load a source volume before resetting slice-view volumes.",
                4000,
            )
            return

        self._plane_display_volumes_active = False
        self._plane_display_snapshot = None
        for selector_name in PLANE_DISPLAY_SELECTOR_NAMES.values():
            getattr(self.ui, selector_name).setCurrentNode(None)

        missing_views = [
            slice_view_name
            for slice_view_name in PLANE_DISPLAY_SELECTOR_NAMES
            if not self._set_slice_view_background(slice_view_name, source_volume)
        ]
        if missing_views:
            slicer.util.showStatusMessage(
                "Slice views unavailable: " + ", ".join(missing_views),
                4000,
            )
            return

        # Backgrounds changed; re-align views (and straighten any oblique series).
        if self._get_snap_slices_setting():
            self._align_views_to_volume_planes()

        slicer.util.showStatusMessage(
            "Restored the Segment Editor source volume in all slice views.",
            4000,
        )

    def get_inference_volume_node(self):
        """
        Return the image grid currently used by nnInteractive.

        The embedded Segment Editor source volume remains the canonical output
        grid. A supplemental working volume is used only while native-series
        inference is explicitly enabled. In tri-planar mode a per-prompt override
        (set while routing an interaction to the clicked view's series) takes
        precedence so the prompt runs against that series.
        """
        override = getattr(self, "_active_inference_volume_override", None)
        if override is not None and slicer.mrmlScene.IsNodePresent(override):
            return override
        source_volume = self.get_volume_node()
        if source_volume is None:
            return None
        if not hasattr(self, "ui"):
            return source_volume
        if not self.ui.cbEnableNativeSeriesInference.isChecked():
            return source_volume
        return self.ui.cbInferenceWorkingVolume.currentNode() or source_volume

    def _is_native_series_inference_active(self):
        """True when nnInteractive is analyzing a supplemental volume."""
        source_volume = self.get_volume_node()
        working_volume = self.get_inference_volume_node()
        return (
            source_volume is not None
            and working_volume is not None
            and source_volume.GetID() != working_volume.GetID()
        )

    # -- High-resolution output geometry -------------------------------------

    def _high_res_output_enabled(self):
        """True when the user opted into a high-resolution output grid."""
        if not hasattr(self, "ui"):
            return False
        return self.ui.cbEnableHighResOutput.isChecked()

    def _get_output_spacing(self, source_volume=None):
        """Isotropic output spacing in mm (0/empty -> finest source spacing),
        coarsened so the resulting isotropic grid stays within the voxel budget
        (tighter in tri-planar mode, where SDF fusion + resample + render run on
        the output grid after every interaction). Returns the effective (capped)
        spacing so the geometry-cache comparison in _ensure_output_geometry_node
        stays stable (otherwise the grid would rebuild on every call)."""
        value = 0.0
        if hasattr(self, "ui"):
            try:
                value = float(self.ui.sbOutputSpacing.value)
            except Exception:
                value = 0.0
        else:
            value = float(
                slicer.util.settingsValue(
                    SETTING_OUTPUT_SPACING, 0.0, converter=float
                )
            )
        if source_volume is None:
            source_volume = self.get_volume_node()
        if value <= 0.0:
            value = min(source_volume.GetSpacing()) if source_volume else 1.0
        value = max(OUTPUT_SPACING_MIN_MM, min(OUTPUT_SPACING_MAX_MM, value))
        # Coarsen to fit the voxel budget so the isotropic grid stays tractable.
        budget = (
            TRIPLANAR_MAX_OUTPUT_VOXELS
            if getattr(self, "_triplanar_mode", False)
            else OUTPUT_GEOMETRY_HARD_VOXEL_BUDGET
        )
        image = source_volume.GetImageData() if source_volume is not None else None
        if image is not None:
            spacing = source_volume.GetSpacing()
            dims = image.GetDimensions()
            extents = [dims[a] * spacing[a] for a in range(3)]

            def voxel_count(iso):
                return float(
                    np.prod([max(1, int(np.ceil(e / iso))) for e in extents])
                )

            if voxel_count(value) > budget:
                factor = (voxel_count(value) / budget) ** (1.0 / 3.0)
                value = min(OUTPUT_SPACING_MAX_MM, value * factor)
        return value

    def get_output_volume_node(self):
        """
        Return the scalar volume whose grid the canonical segment is stored on.

        Defaults to the Segment Editor source volume (fully backward compatible).
        When the high-resolution output feature is on, returns a hidden isotropic
        geometry volume so masks are stored at fine resolution in every plane.
        """
        source_volume = self.get_volume_node()
        if source_volume is None:
            return None
        if not self._high_res_output_enabled():
            return source_volume
        node = self._ensure_output_geometry_node(source_volume)
        if node is None:
            slicer.util.showStatusMessage(
                "Could not build the high-resolution output grid; "
                "using the source grid.",
                4000,
            )
            return source_volume
        return node

    def _output_geometry_active(self):
        """True when the canonical output grid differs from the source grid."""
        source_volume = self.get_volume_node()
        output_volume = self.get_output_volume_node()
        return (
            source_volume is not None
            and output_volume is not None
            and source_volume.GetID() != output_volume.GetID()
        )

    def _output_grid_shape(self):
        """numpy (z, y, x) shape of the canonical output grid, or None."""
        output_volume = self.get_output_volume_node()
        if output_volume is None:
            return None
        return slicer.util.arrayFromVolume(output_volume).shape

    def _to_output_grid(self, mask):
        """Resample a source-grid mask onto the output grid (no-op if equal)."""
        return self._resample_mask_between_volumes(
            mask, self.get_volume_node(), self.get_output_volume_node()
        )

    def _smoothing_active(self):
        """True when smooth (interpolated) results are enabled and possible.

        Smoothing needs a fine output grid to interpolate onto, so it only
        counts as active when the high-resolution output grid is also active.
        """
        return (
            hasattr(self, "ui")
            and self.ui.cbSmoothInterpolate.isChecked()
            and self._output_geometry_active()
        )

    def _interpolate_mask_to_output_grid(self, coarse_mask, mask_volume):
        """Interpolate a coarse mask onto the fine output grid for smoothness.

        The coarse mask is blocky in the through-plane direction because the
        source series is thick-sliced. When the mask sits on the same grid the
        output geometry was built from (the normal main-series path), use
        signed-distance (shape-based) interpolation: this reconstructs a smooth
        surface between the thick slices instead of replicating blocky steps.
        For a mask on a non-coplanar grid (e.g. a supplemental working volume),
        fall back to nearest-neighbor resampling followed by Gaussian smoothing.

        Returns a uint8 mask on the output grid, or None so callers can degrade.
        """
        if not self._output_geometry_active():
            return None
        try:
            from scipy import ndimage

            coarse = np.asarray(coarse_mask).astype(bool)
            out_shape = self._output_grid_shape()
            if out_shape is None:
                return None
            if not coarse.any():
                return np.zeros(out_shape, dtype=np.uint8)

            iso = self._output_geometry_spacing
            spacing = mask_volume.GetSpacing()  # (x, y, z) in IJK order
            # numpy axis order is (z, y, x).
            samp = (spacing[2], spacing[1], spacing[0])

            coplanar = mask_volume.GetID() == self._output_geometry_source_id
            if coplanar:
                # Signed distance field in mm: positive inside, negative outside.
                sdf = (
                    ndimage.distance_transform_edt(coarse, sampling=samp)
                    - ndimage.distance_transform_edt(~coarse, sampling=samp)
                ).astype(np.float32)
                # Output/input voxel ratio per numpy axis (z, y, x).
                zoom = (
                    spacing[2] / iso,
                    spacing[1] / iso,
                    spacing[0] / iso,
                )
                scaled = ndimage.zoom(sdf, zoom, order=1)
                # zoom rounds dims while the grid uses ceil; fit exact shape.
                out = np.full(out_shape, -1.0, dtype=np.float32)
                clip = tuple(
                    slice(0, min(out_shape[a], scaled.shape[a])) for a in range(3)
                )
                out[clip] = scaled[clip]
                return (out >= 0).astype(np.uint8)

            # Non-coplanar grid: nearest resample, then Gaussian-smooth on output.
            resampled = self._resample_mask_between_volumes(
                coarse_mask, mask_volume, self.get_output_volume_node()
            )
            if resampled is None:
                return None
            source_spacing = self.get_volume_node().GetSpacing()
            sigma = float(np.clip(0.5 * max(source_spacing) / iso, 0.5, 3.0))
            smoothed = ndimage.gaussian_filter(
                resampled.astype(np.float32), sigma=sigma
            )
            return (smoothed >= 0.5).astype(np.uint8)
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            print(f"[nni] smooth interpolate failed: {e}")
            return None

    def _ensure_output_geometry_node(self, source_volume):
        """Build or reuse the isotropic output-geometry volume for source_volume."""
        # Tri-planar: cover the UNION of the three assigned series, not the editor
        # source (which may be a different/smaller series and would clip anatomy
        # that is visible in the views but outside the source FOV).
        if getattr(self, "_triplanar_mode", False):
            node = self._ensure_triplanar_output_geometry_node()
            if node is not None:
                return node
            # Fall through to the source-based grid if the union is unavailable
            # (e.g. fewer than two assigned series): keeps the old behavior.
        iso = self._get_output_spacing(source_volume)
        node = self._output_geometry_node
        if (
            node is not None
            and slicer.mrmlScene.IsNodePresent(node)
            and self._output_geometry_spacing is not None
            and abs(self._output_geometry_spacing - iso) < 1e-6
            and self._output_geometry_source_id == source_volume.GetID()
        ):
            return node
        return self._build_output_geometry_node(source_volume, iso)

    def _triplanar_coverage_volumes(self):
        """The distinct assigned tri-planar display series (Red/Yellow/Green
        backgrounds), in selector order, de-duplicated by node id."""
        vols, seen = [], set()
        for v in PLANE_DISPLAY_SELECTOR_NAMES:
            n = self._view_background_volume(v)
            if n is not None and n.GetID() not in seen:
                seen.add(n.GetID())
                vols.append(n)
        return vols

    def _triplanar_union_ras_bounds(self, vols):
        """RAS axis-aligned union [xmin,xmax,ymin,ymax,zmin,zmax] of `vols`, or
        None. GetRASBounds already bounds each (possibly oblique) volume in world
        RAS, so the union box is guaranteed to contain all of their anatomy."""
        union = None
        for n in vols:
            b = [0.0] * 6
            n.GetRASBounds(b)
            if union is None:
                union = list(b)
            else:
                for i in (0, 2, 4):
                    union[i] = min(union[i], b[i])
                for i in (1, 3, 5):
                    union[i] = max(union[i], b[i])
        return union

    def _ensure_triplanar_output_geometry_node(self):
        """Build or reuse a RAS-axis-aligned isotropic grid that covers the union
        of the assigned tri-planar series. Returns None when fewer than two
        series are assigned (caller then falls back to the source grid)."""
        vols = self._triplanar_coverage_volumes()
        if len(vols) < 2:
            return None
        bounds = self._triplanar_union_ras_bounds(vols)
        if bounds is None:
            return None
        # Base spacing: the finest of the assigned series, coarsened to the
        # tri-planar voxel budget against the (larger) union extents.
        base_iso = min(min(v.GetSpacing()) for v in vols)
        base_iso = max(OUTPUT_SPACING_MIN_MM, min(OUTPUT_SPACING_MAX_MM, base_iso))
        extents = [bounds[1] - bounds[0], bounds[3] - bounds[2],
                   bounds[5] - bounds[4]]

        def voxel_count(s):
            return float(np.prod([max(1, int(np.ceil(e / s))) for e in extents]))

        iso = base_iso
        if voxel_count(iso) > TRIPLANAR_MAX_OUTPUT_VOXELS:
            factor = (voxel_count(iso) / float(TRIPLANAR_MAX_OUTPUT_VOXELS)) ** (
                1.0 / 3.0)
            iso = min(OUTPUT_SPACING_MAX_MM, iso * factor)
        sig = (
            round(iso, 4),
            tuple(sorted(v.GetID() for v in vols)),
            tuple(round(b, 2) for b in bounds),
        )
        node = self._output_geometry_node
        if (
            node is not None
            and slicer.mrmlScene.IsNodePresent(node)
            and self._output_geometry_triplanar_sig == sig
        ):
            return node
        return self._build_triplanar_output_geometry_node(bounds, extents, iso, sig)

    def _build_triplanar_output_geometry_node(self, bounds, extents, iso, sig):
        """Create the hidden RAS-axis-aligned isotropic geometry volume covering
        the union box `bounds` at spacing `iso`."""
        new_dims = [max(1, int(np.ceil(extents[a] / iso))) for a in range(3)]
        total = new_dims[0] * new_dims[1] * new_dims[2]
        new_matrix = vtk.vtkMatrix4x4()
        new_matrix.Identity()
        for a in range(3):
            new_matrix.SetElement(a, a, iso)
        # RAS origin at the min corner of the union box (no parent transform:
        # GetRASBounds is already world RAS).
        new_matrix.SetElement(0, 3, bounds[0])
        new_matrix.SetElement(1, 3, bounds[2])
        new_matrix.SetElement(2, 3, bounds[4])

        new_image = vtk.vtkImageData()
        new_image.SetDimensions(new_dims[0], new_dims[1], new_dims[2])
        new_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        new_image.GetPointData().GetScalars().Fill(0)

        node = self._output_geometry_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", self.output_geometry_node_name
            )
            node.HideFromEditorsOn()
        node.SetAndObserveImageData(new_image)
        node.SetIJKToRASMatrix(new_matrix)
        node.SetAndObserveTransformNodeID(None)
        self._output_geometry_node = node
        self._output_geometry_spacing = iso
        # Sentinel so the non-tri-planar cache comparison never accidentally
        # reuses this union grid as if it were a source-aligned one.
        self._output_geometry_source_id = "__triplanar_union__"
        self._output_geometry_triplanar_sig = sig
        print("[DEBUG triplanar.grid] union bounds={} extents(mm)={} iso={} "
              "dims={} voxels={}".format(
                  [round(b, 1) for b in bounds],
                  [round(e, 1) for e in extents], round(iso, 3),
                  tuple(new_dims), total))
        if total > TRIPLANAR_MAX_OUTPUT_VOXELS:
            slicer.util.showStatusMessage(
                "Tri-planar output grid coarsened to %.2f mm to fit the budget."
                % iso, 5000)
        return node

    def _build_output_geometry_node(self, source_volume, iso):
        """Create a hidden isotropic, source-aligned geometry-only scalar volume."""
        image = source_volume.GetImageData()
        if image is None:
            return None
        spacing = source_volume.GetSpacing()
        dims = image.GetDimensions()
        extents = [dims[a] * spacing[a] for a in range(3)]

        def voxel_dims(spacing_iso):
            return [max(1, int(np.ceil(extents[a] / spacing_iso))) for a in range(3)]

        new_dims = voxel_dims(iso)
        total = new_dims[0] * new_dims[1] * new_dims[2]
        if total > OUTPUT_GEOMETRY_HARD_VOXEL_BUDGET:
            factor = (total / float(OUTPUT_GEOMETRY_HARD_VOXEL_BUDGET)) ** (1.0 / 3.0)
            iso = iso * factor
            new_dims = voxel_dims(iso)
            total = new_dims[0] * new_dims[1] * new_dims[2]
            slicer.util.showStatusMessage(
                "Output spacing coarsened to %.2f mm to fit the memory budget."
                % iso,
                5000,
            )
        elif total > OUTPUT_GEOMETRY_SOFT_VOXEL_BUDGET:
            slicer.util.showStatusMessage(
                "High-resolution output grid is large: %d x %d x %d voxels."
                % (new_dims[0], new_dims[1], new_dims[2]),
                5000,
            )

        src_ijk_to_ras = vtk.vtkMatrix4x4()
        source_volume.GetIJKToRASMatrix(src_ijk_to_ras)
        new_matrix = vtk.vtkMatrix4x4()
        new_matrix.DeepCopy(src_ijk_to_ras)
        for axis in range(3):
            scale = iso / spacing[axis]
            for row in range(3):
                new_matrix.SetElement(
                    row, axis, src_ijk_to_ras.GetElement(row, axis) * scale
                )

        new_image = vtk.vtkImageData()
        new_image.SetDimensions(new_dims[0], new_dims[1], new_dims[2])
        new_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        new_image.GetPointData().GetScalars().Fill(0)

        node = self._output_geometry_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", self.output_geometry_node_name
            )
            node.HideFromEditorsOn()
        node.SetAndObserveImageData(new_image)
        node.SetIJKToRASMatrix(new_matrix)
        node.SetAndObserveTransformNodeID(source_volume.GetTransformNodeID())
        self._output_geometry_node = node
        self._output_geometry_spacing = iso
        self._output_geometry_source_id = source_volume.GetID()
        return node

    def _refresh_native_series_inference_ui(self, *args):
        """Update controls after enabling, disabling, or changing the working volume."""
        enabled = self.ui.cbEnableNativeSeriesInference.isChecked()
        active = self._is_native_series_inference_active()
        has_preview = self._inference_result_source_mask is not None
        registering = self._alignment_in_progress
        self.ui.cbInferenceWorkingVolume.setEnabled(enabled and not registering)
        self.ui.cbInferenceSyncMode.setEnabled(active)
        # Block sync while a registration runs so a stale preview built on the
        # old geometry is not merged with the freshly aligned grid.
        self.ui.pbSyncInferenceResult.setEnabled(
            active and has_preview and not registering
        )
        self.ui.pbClearInferencePreview.setEnabled(has_preview)
        # Alignment preferences also serve the multi-plane display volumes, so
        # they stay available even when native-series inference is off.
        self.ui.cbAutoRegisterSupplemental.setEnabled(not registering)
        self.ui.cbConfirmSeriesAligned.setEnabled(not registering)
        self.ui.pbRegisterSupplemental.setEnabled(active and not registering)
        self.ui.pbClearAlignment.setEnabled(
            bool(self._alignment_transforms) and not registering
        )
        # Multi-series fusion controls.
        fusion_on = self._get_fusion_enabled()
        n = len(self._fusion_results)
        self.ui.pbFuseSeries.setVisible(fusion_on)
        self.ui.pbFuseSeries.setEnabled(fusion_on and n >= 1)
        self.ui.pbFuseSeries.setText("Fuse & apply ({})".format(n))

    def _clear_inference_cache(self):
        """Drop the inference preview and force the next prompt to re-upload."""
        self._destroy_inference_preview()
        if hasattr(self, "previous_states"):
            self.previous_states.pop("image_data", None)
            self.previous_states.pop("image_volume_id", None)
            self.previous_states.pop("segment_data", None)

    def _on_native_series_inference_settings_changed(self, *args):
        """
        Discard stale previews and force a server resync after changing the
        inference image. Switching volumes resets the server interaction chain.
        """
        self._clear_inference_cache()
        self._reset_multiview_lasso_accumulation()
        self._sync_supplemental_alignment()
        self._refresh_native_series_inference_ui()

    # -- Supplemental-series alignment (auto-registration) -------------------

    def _set_registration_status(self, text, warn=False):
        """Show alignment status both in the panel label and the status bar."""
        if not hasattr(self, "ui"):
            return
        label = self.ui.lblRegistrationStatus
        label.setText(text)
        label.setStyleSheet("color: #c0392b;" if warn else "")
        if text:
            slicer.util.showStatusMessage(text, 5000)

    def _frame_of_reference_uid(self, volume_node):
        """
        Return the DICOM FrameOfReferenceUID (0020,0052) for a volume, or None.

        Two volumes are physically comparable in RAS only when they share this
        UID. A new mid-exam localizer or repositioning yields a different UID,
        so equal UIDs mean aligned and different UIDs mean registration is
        needed. Non-DICOM volumes have no UID and return None (unknown).
        """
        if volume_node is None:
            return None
        try:
            db = slicer.dicomDatabase
            sh = slicer.mrmlScene.GetSubjectHierarchyNode()
            item = sh.GetItemByDataNode(volume_node) if sh is not None else 0
            series_uid = sh.GetItemUID(item, "DICOM") if item else ""
            if series_uid:
                files = db.filesForSeries(series_uid)
                if files:
                    value = db.fileValue(files[0], "0020,0052")
                    if value:
                        return value
            instance_uids = volume_node.GetAttribute("DICOM.instanceUIDs")
            if instance_uids:
                first = instance_uids.split()[0]
                value = db.fileValueForInstance(first, "0020,0052")
                if value:
                    return value
        except Exception as exc:
            print("[nni] frame-of-reference lookup failed: %s" % exc)
        return None

    def _series_aligned(self, supplemental, source):
        """True/False if both frames of reference are known, else None."""
        supp_for = self._frame_of_reference_uid(supplemental)
        src_for = self._frame_of_reference_uid(source)
        if not supp_for or not src_for:
            return None
        return supp_for == src_for

    def _attach_alignment_transform(self, supplemental, transform):
        """Parent the supplemental volume under its alignment transform."""
        if supplemental is not None and transform is not None:
            supplemental.SetAndObserveTransformNodeID(transform.GetID())

    def _drop_alignment_entry(self, key):
        """Detach and delete the transform cached for one (supp, source) pair."""
        transform = self._alignment_transforms.pop(key, None)
        supp = slicer.mrmlScene.GetNodeByID(key[0])
        if (
            supp is not None
            and transform is not None
            and supp.GetTransformNodeID() == transform.GetID()
        ):
            supp.SetAndObserveTransformNodeID(None)
        if transform is not None and slicer.mrmlScene.IsNodePresent(transform):
            slicer.mrmlScene.RemoveNode(transform)

    def _remove_alignment_transforms(self):
        """Detach and delete every cached alignment transform."""
        self._alignment_queue = []
        for key in list(self._alignment_transforms.keys()):
            self._drop_alignment_entry(key)

    def _cancel_active_registration(self):
        """Tear down an in-flight registration so cleanup cannot dangle.

        The CLI observer is added directly on the node (not via
        VTKObservationMixin), so removeObservers() would not catch it, and the
        pending transform is not yet cached. Remove both explicitly.
        """
        cli = self._alignment_cli_node
        if cli is not None:
            if self._alignment_cli_observer is not None:
                try:
                    cli.RemoveObserver(self._alignment_cli_observer)
                except Exception:
                    pass
            try:
                cli.Cancel()
            except Exception:
                pass
        if self._alignment_pending is not None:
            node = slicer.mrmlScene.GetNodeByID(self._alignment_pending[2])
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._alignment_cli_observer = None
        self._alignment_cli_node = None
        self._alignment_pending = None
        self._alignment_in_progress = False
        self._alignment_queue = []

    def _prune_alignment_for_source(self, source_id):
        """Drop cached transforms that target a stale (no longer current) source."""
        for key in list(self._alignment_transforms.keys()):
            if key[1] != source_id:
                self._drop_alignment_entry(key)

    def _on_alignment_geometry_changed(self):
        """Force the next prompt to re-upload after the working geometry moved."""
        self._clear_inference_cache()
        # A series moved relative to the output grid, so its cached FOV coverage
        # is geometrically stale and must be rebuilt on the next fuse.
        self._invalidate_fusion_coverage("alignment geometry changed")

    def _sync_supplemental_alignment(self):
        """Align the native-series working volume after a settings change."""
        if not hasattr(self, "ui"):
            return
        source = self.get_volume_node()
        if source is not None:
            # A stale source invalidates transforms that targeted it; drop them.
            # Transforms for the current source stay attached and cached.
            self._prune_alignment_for_source(source.GetID())
        if not self._is_native_series_inference_active():
            self._set_registration_status("", False)
            return
        self._ensure_alignment(self.get_inference_volume_node(), source)

    def _ensure_alignment(self, supplemental, source, force=False):
        """
        Make sure the supplemental volume is aligned to the source volume.

        Reuses a cached transform when present; otherwise inspects the DICOM
        frame of reference and, when it differs (or force is set), starts a
        rigid registration. Unknown frames of reference never auto-register.
        """
        if supplemental is None or source is None:
            return
        if supplemental.GetID() == source.GetID():
            return
        key = (supplemental.GetID(), source.GetID())

        if force:
            # Re-registering: discard the previous transform so it is not leaked
            # when the cache entry is overwritten on completion.
            self._drop_alignment_entry(key)

        cached = self._alignment_transforms.get(key)
        if (
            not force
            and cached is not None
            and slicer.mrmlScene.IsNodePresent(cached)
        ):
            self._attach_alignment_transform(supplemental, cached)
            self._set_registration_status(
                "Reusing the existing registration for the supplemental series.",
                False,
            )
            return

        if not force:
            # If the user has confirmed the series are already aligned, never
            # auto-register -- regardless of whether the frames of reference are
            # readable. (Previously this was only honored when the FoR was
            # unreadable; a readable-but-different FoR still queued BRAINSFit,
            # which on large oblique series can crash Slicer.)
            if hasattr(self, "ui") and self.ui.cbConfirmSeriesAligned.isChecked():
                _perf_log("[DEBUG triplanar.perf] ensure_alignment: skip "
                          "(user confirmed aligned) vol=%s"
                          % supplemental.GetName())
                self._set_registration_status(
                    "Series confirmed aligned; auto-registration skipped.", False
                )
                return
            aligned = self._series_aligned(supplemental, source)
            if aligned is True:
                self._set_registration_status(
                    "Supplemental series shares the source frame of reference; "
                    "no registration needed.",
                    False,
                )
                return
            if aligned is None:
                if self.ui.cbConfirmSeriesAligned.isChecked():
                    self._set_registration_status(
                        "Alignment confirmed manually; registration skipped.",
                        False,
                    )
                    return
                self._set_registration_status(
                    "Could not read the DICOM frame of reference. Verify "
                    "alignment visually or click Register now.",
                    True,
                )
                return
            if not self.ui.cbAutoRegisterSupplemental.isChecked():
                self._set_registration_status(
                    "Supplemental series uses a different frame of reference. "
                    "Click Register now to align it to the source volume.",
                    True,
                )
                return

        _perf_log("[DEBUG triplanar.perf] ensure_alignment: queue registration "
                  "(BRAINSFit) vol=%s" % supplemental.GetName())
        self._enqueue_alignment(supplemental, source)

    def _enqueue_alignment(self, moving, fixed):
        """Queue a (moving -> fixed) registration and start it when a slot frees."""
        pair = (moving.GetID(), fixed.GetID())
        pending_pair = (
            self._alignment_pending[:2] if self._alignment_pending else None
        )
        if pair == pending_pair or pair in self._alignment_queue:
            return
        self._alignment_queue.append(pair)
        self._pump_alignment_queue()

    def _pump_alignment_queue(self):
        """Start the next queued registration if none is currently running."""
        if self._alignment_in_progress or not self._alignment_queue:
            return
        moving_id, fixed_id = self._alignment_queue.pop(0)
        moving = slicer.mrmlScene.GetNodeByID(moving_id)
        fixed = slicer.mrmlScene.GetNodeByID(fixed_id)
        if moving is None or fixed is None:
            # A queued volume disappeared; skip it and try the next one.
            self._pump_alignment_queue()
            return
        self._start_registration(moving, fixed)

    def _start_registration(self, moving, fixed):
        """Launch an asynchronous BRAINSFit registration (moving -> fixed)."""
        _perf_log("[DEBUG triplanar.perf] start_registration moving=%s fixed=%s"
                  % (moving.GetName() if moving else None,
                     fixed.GetName() if fixed else None))
        if self._alignment_in_progress:
            return
        try:
            brainsfit = slicer.modules.brainsfit
        except AttributeError:
            self._set_registration_status(
                "Registration module (BRAINSFit) is unavailable in this Slicer "
                "build; cannot auto-align the supplemental series.",
                True,
            )
            return

        transform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode",
            self.series_alignment_transform_node_name,
        )
        transform.HideFromEditorsOn()
        transform.SetAttribute(
            "NNInteractive.AlignmentPair",
            "%s|%s" % (moving.GetID(), fixed.GetID()),
        )
        # Rigid handles patient motion / repositioning; Affine is an escape hatch
        # when a rigid fit leaves a large residual.
        transform_type = "Rigid"
        if (
            hasattr(self.ui, "cbRegistrationMode")
            and self.ui.cbRegistrationMode.currentText == "Affine"
        ):
            transform_type = "Rigid,Affine"
        params = {
            "fixedVolume": fixed.GetID(),
            "movingVolume": moving.GetID(),
            "linearTransform": transform.GetID(),
            "transformType": transform_type,
            "initializeTransformMode": "useMomentsAlign",
            "costMetric": "MMI",
            "samplingPercentage": 0.02,
        }
        self._alignment_in_progress = True
        self._alignment_pending = (
            moving.GetID(),
            fixed.GetID(),
            transform.GetID(),
        )
        self._set_registration_status(
            "Registering the supplemental series to the source volume...",
            False,
        )
        self._refresh_native_series_inference_ui()
        try:
            cli_node = slicer.cli.run(
                brainsfit, None, params, wait_for_completion=False
            )
        except Exception as exc:
            slicer.mrmlScene.RemoveNode(transform)
            self._alignment_in_progress = False
            self._alignment_pending = None
            self._set_registration_status(
                "Could not start registration: %s" % exc, True
            )
            self._refresh_native_series_inference_ui()
            return
        self._alignment_cli_node = cli_node
        self._alignment_cli_observer = cli_node.AddObserver(
            slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent,
            self._on_registration_status_modified,
        )

    def _on_registration_status_modified(self, cli_node, event):
        """Finish once the registration CLI reaches a terminal state."""
        if cli_node.IsBusy():
            return
        self._finish_registration(cli_node, cli_node.GetStatusString())

    def _finish_registration(self, cli_node, status):
        """Attach the transform on success or clean up and warn on failure."""
        _perf_log("[DEBUG triplanar.perf] finish_registration status=%s" % status)
        if not self._alignment_in_progress:
            return
        self._alignment_in_progress = False
        pending = self._alignment_pending
        self._alignment_pending = None
        if self._alignment_cli_observer is not None:
            cli_node.RemoveObserver(self._alignment_cli_observer)
        self._alignment_cli_observer = None
        self._alignment_cli_node = None

        moving_id, fixed_id, transform_id = pending
        moving = slicer.mrmlScene.GetNodeByID(moving_id)
        transform = slicer.mrmlScene.GetNodeByID(transform_id)
        if slicer.mrmlScene.IsNodePresent(cli_node):
            slicer.mrmlScene.RemoveNode(cli_node)

        if status == "Completed" and moving is not None and transform is not None:
            offset = self._transform_translation_mm(transform)
            rotation = self._transform_rotation_deg(transform)
            if (
                offset <= REGISTRATION_IDENTITY_TRANSLATION_MM
                and rotation <= REGISTRATION_IDENTITY_ROTATION_DEG
            ):
                # Frames of reference differ in name only; the series are already
                # aligned. Drop the transform to avoid needless resampling.
                slicer.mrmlScene.RemoveNode(transform)
                self._set_registration_status(
                    "Supplemental series is already aligned with the source "
                    "volume (registration was near-identity).",
                    False,
                )
            else:
                self._attach_alignment_transform(moving, transform)
                self._alignment_transforms[(moving_id, fixed_id)] = transform
                self._enable_slice_intersections()
                self._set_registration_status(
                    "Registered the supplemental series to the source volume "
                    "(translation %.1f mm, rotation %.1f deg). Verify alignment "
                    "with slice intersections." % (offset, rotation),
                    offset > REGISTRATION_OFFSET_WARN_MM,
                )
        else:
            if transform is not None and slicer.mrmlScene.IsNodePresent(transform):
                slicer.mrmlScene.RemoveNode(transform)
            self._set_registration_status(
                "Registration failed (%s). The supplemental series is NOT "
                "aligned; results may be misplaced." % status,
                True,
            )
        # Any outcome (attach, near-identity discard, or failure) can change the
        # working grid relative to what the server last received -- a forced
        # re-registration already detached the previous transform. Force the next
        # prompt to re-upload whenever the active working volume was involved.
        if self._is_active_working_volume(moving_id):
            self._on_alignment_geometry_changed()
        self._refresh_native_series_inference_ui()
        # Hand off to the next queued registration, if any.
        self._pump_alignment_queue()

    def _is_active_working_volume(self, volume_id):
        """True when volume_id is the volume nnInteractive currently analyzes."""
        if not self._is_native_series_inference_active():
            return False
        working = self.get_inference_volume_node()
        return working is not None and working.GetID() == volume_id

    @staticmethod
    def _enable_slice_intersections():
        """Turn on slice intersections so the user can verify alignment."""
        try:
            view_nodes = slicer.util.getNodesByClass("vtkMRMLSliceDisplayNode")
            for node in view_nodes:
                node.SetIntersectingSlicesVisibility(True)
        except Exception:
            # Older Slicer builds expose this differently; verification is a
            # convenience, so a failure here is non-fatal.
            pass

    @staticmethod
    def _transform_translation_mm(transform):
        """Magnitude of the translation column of a linear transform, in mm."""
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(matrix)
        return (
            matrix.GetElement(0, 3) ** 2
            + matrix.GetElement(1, 3) ** 2
            + matrix.GetElement(2, 3) ** 2
        ) ** 0.5

    @staticmethod
    def _transform_rotation_deg(transform):
        """Rotation angle of a linear transform's 3x3 block, in degrees."""
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(matrix)
        trace = (
            matrix.GetElement(0, 0)
            + matrix.GetElement(1, 1)
            + matrix.GetElement(2, 2)
        )
        # Clamp to acos's domain to absorb floating-point and scale/shear noise.
        cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.degrees(math.acos(cos_angle))

    def on_register_supplemental_clicked(self, checked=False):
        """Manually register the current working volume to the source volume."""
        source = self.get_volume_node()
        working = self.get_inference_volume_node()
        if source is None or working is None or source.GetID() == working.GetID():
            self._set_registration_status(
                "Select a supplemental working volume different from the source "
                "volume before registering.",
                True,
            )
            return
        self._ensure_alignment(working, source, force=True)

    def on_clear_alignment_clicked(self, checked=False):
        """Remove all alignment transforms and restore original positions."""
        self._remove_alignment_transforms()
        self._on_alignment_geometry_changed()
        self._set_registration_status(
            "Cleared series alignment; the supplemental series is back to its "
            "original position.",
            False,
        )
        self._refresh_native_series_inference_ui()

    @staticmethod
    def _set_segmentation_reference_volume(segmentation_node, volume_node):
        """Align a staging segmentation node with a scalar volume."""
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
            volume_node
        )
        segmentation_node.SetAndObserveTransformNodeID(
            volume_node.GetTransformNodeID()
        )

    def _resample_mask_between_volumes(
        self, mask, source_volume, target_volume
    ):
        """
        Resample a binary numpy mask between registered scalar-volume grids.

        A temporary segmentation node delegates the conversion to Slicer's
        segmentation infrastructure, which preserves label semantics.

        Returns a uint8 mask on the target grid, or None if Slicer could not
        perform the resample (callers must degrade gracefully).
        """
        mask = np.asarray(mask).astype(np.uint8)
        if source_volume.GetID() == target_volume.GetID():
            return mask.copy()

        # An empty segment cannot be exported to a reference geometry (Slicer's
        # GenerateSharedLabelmap fails), so short-circuit to a target-grid zero
        # mask. This also speeds up clear/empty paths.
        if int(mask.sum()) == 0:
            return np.zeros(
                slicer.util.arrayFromVolume(target_volume).shape, dtype=np.uint8
            )

        staging_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        staging_node.SetName("NativeSeriesInferenceResample (temporary)")
        staging_node.HideFromEditorsOn()
        self._set_segmentation_reference_volume(staging_node, source_volume)
        staging_segment_id = staging_node.GetSegmentation().AddEmptySegment(
            "Resample", "Resample"
        )
        try:
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask,
                staging_node,
                staging_segment_id,
                source_volume,
            )
            resampled = slicer.util.arrayFromSegmentBinaryLabelmap(
                staging_node,
                staging_segment_id,
                target_volume,
            )
            if resampled is None:
                raise RuntimeError("Slicer returned no resampled labelmap.")
            return resampled.astype(np.uint8)
        except Exception as exc:
            # Most likely the target grid is too large to resample into. Report
            # the geometry so the failure can be diagnosed, and let callers fall
            # back rather than crashing the UI.
            try:
                target_dims = slicer.util.arrayFromVolume(target_volume).shape
            except Exception:
                target_dims = "unknown"
            print(
                "[nni] mask resample failed (target dims=%s): %s"
                % (target_dims, exc)
            )
            return None
        finally:
            slicer.mrmlScene.RemoveNode(staging_node)

    @staticmethod
    def compute_inference_sync_mask(source_mask, preview_mask, mode):
        """
        Merge a supplemental-series preview into the canonical source mask.

        mode: 0=Add, 1=Replace, 2=Subtract, 3=Intersect.
        """
        source_mask = np.asarray(source_mask).astype(bool)
        preview_mask = np.asarray(preview_mask).astype(bool)
        if source_mask.shape != preview_mask.shape:
            raise ValueError(
                "Source and preview masks have different shapes: "
                f"{source_mask.shape} vs {preview_mask.shape}."
            )
        if mode == 0:
            result = source_mask | preview_mask
        elif mode == 1:
            result = preview_mask
        elif mode == 2:
            result = source_mask & ~preview_mask
        elif mode == 3:
            result = source_mask & preview_mask
        else:
            raise ValueError(f"Unknown inference sync mode index: {mode}")
        return result.astype(np.uint8)

    def _get_or_create_inference_preview_segmentation(self):
        """Create the output-grid overlay used for supplemental inference previews."""
        node = self._inference_preview_segment_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            node.SetName(self.inference_preview_segment_node_name)
            node.HideFromEditorsOn()
            output_volume = self.get_output_volume_node()
            if output_volume is not None:
                self._set_segmentation_reference_volume(node, output_volume)
            node.CreateDefaultDisplayNodes()
            self._inference_preview_segment_node = node

        segmentation = node.GetSegmentation()
        segment_id = self._inference_preview_segment_id
        if not segment_id or not segmentation.GetSegment(segment_id):
            segment_id = segmentation.AddEmptySegment(
                "NativeSeriesInferencePreview",
                "NativeSeriesInferencePreview",
                [1.0, 0.6, 0.1],
            )
            self._inference_preview_segment_id = segment_id

        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentOpacity2DFill(segment_id, 0.35)
            display_node.SetSegmentOpacity2DOutline(segment_id, 0.9)
            display_node.SetSegmentVisibility(segment_id, True)
        return node, segment_id

    def _update_inference_preview(self, working_mask):
        """Resample an nnInteractive result onto the output grid and display it.

        Resampling working->output (instead of working->source) keeps the full
        high-resolution detail of a supplemental-series result on the canonical
        grid, which is the whole point of the high-resolution output feature.
        """
        output_volume = self.get_output_volume_node()
        working_volume = self.get_inference_volume_node()
        if self._smoothing_active():
            output_mask = self._interpolate_mask_to_output_grid(
                working_mask, working_volume
            )
        else:
            output_mask = self._resample_mask_between_volumes(
                working_mask, working_volume, output_volume
            )
        if output_mask is None:
            # High-res output resample failed; fall back to the source grid so
            # the preview is still produced.
            self._disable_high_res_output(
                "High-resolution output resample failed; "
                "reverted to the source grid."
            )
            output_volume = self.get_output_volume_node()
            output_mask = self._resample_mask_between_volumes(
                working_mask, working_volume, output_volume
            )
        self._inference_result_working_mask = np.asarray(working_mask).astype(
            np.uint8
        )
        # Stored on the canonical output grid (named _source for historical reasons).
        self._inference_result_source_mask = output_mask.astype(np.uint8)
        self._maybe_collect_fusion_result(output_mask)
        self._inference_preview_target_segment_id = self.get_current_segment_id()
        # Track the source volume identity so we can invalidate if it changes.
        self._inference_preview_source_volume_id = self.get_volume_node().GetID()
        node, segment_id = self._get_or_create_inference_preview_segmentation()
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self._inference_result_source_mask,
            node,
            segment_id,
            output_volume,
        )
        self._refresh_native_series_inference_ui()
        slicer.util.showStatusMessage(
            "Supplemental-series result updated as a preview. "
            "Sync it to the source volume when ready.",
            4000,
        )

    def _destroy_inference_preview(self, checked=False):
        """Discard the supplemental-series preview without editing the source segment."""
        node = self._inference_preview_segment_node
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
        self._inference_preview_segment_node = None
        self._inference_preview_segment_id = None
        self._inference_result_working_mask = None
        self._inference_result_source_mask = None
        self._inference_preview_target_segment_id = None
        self._inference_preview_source_volume_id = None
        if hasattr(self, "ui"):
            self._refresh_native_series_inference_ui()

    def on_clear_inference_preview_clicked(self, checked=False):
        """Discard the preview and restore the source mask as the server target."""
        had_preview = self._inference_result_source_mask is not None
        self._destroy_inference_preview()
        if not had_preview:
            return
        result = self.upload_segment_to_server()
        if result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Native-series inference",
                "The preview was cleared locally, but restoring the source "
                "mask on the server failed.",
            )
            return
        slicer.util.showStatusMessage(
            "Supplemental-series preview cleared; source mask restored on server.",
            4000,
        )

    # -- Multi-series fusion (SDF average) --

    def _get_fusion_enabled(self):
        return hasattr(self, "ui") and self.ui.cbEnableSeriesFusion.isChecked()

    def _maybe_collect_fusion_result(self, output_grid_mask):
        """Store the latest result for the active inference series, keyed by its
        volume id, so the Fuse button can SDF-average all series together.
        Masks must live on the current output grid; stale ones are dropped."""
        # Tri-planar mode always collects per-series results for fusion, even if
        # the cbEnableSeriesFusion checkbox got reset (it is not persisted across
        # a module reload, unlike the tri-planar mode flag).
        enabled = self._get_fusion_enabled() or self._triplanar_mode
        print("[DEBUG fusion.collect] called, fusion_enabled={}".format(enabled))
        if not enabled:
            return
        out_shape = self._output_grid_shape()
        if out_shape is None:
            print("[DEBUG fusion.collect] SKIP: output grid shape is None")
            return
        m = np.asarray(output_grid_mask).astype(np.uint8)
        print("[DEBUG fusion.collect] mask shape={}, out_shape={}, mask sum={}".format(
            tuple(m.shape), tuple(out_shape), int(m.sum())))
        if tuple(m.shape) != tuple(out_shape):
            print("[DEBUG fusion.collect] SKIP: shape mismatch (stale grid)")
            return  # stale grid; skip
        if self._fusion_grid_shape != tuple(out_shape):
            # Output grid changed (source / high-res switch); drop stale store.
            print("[DEBUG fusion.collect] output grid changed -> clearing store")
            self._fusion_results = {}
            self._fusion_grid_shape = tuple(out_shape)
            self._invalidate_fusion_coverage("output grid changed")
        inf_node = self.get_inference_volume_node()
        vid = inf_node.GetID()
        vname = inf_node.GetName()
        is_native = self._is_native_series_inference_active()
        # Store the raw per-series result. The live tri-planar display no longer
        # SDF-averages these (that voting erased fresh edits); they are kept only
        # for the OPTIONAL manual Fuse button, so no cross-series diff sync.
        self._fusion_results[vid] = m.copy()
        print("[DEBUG fusion.collect] stored under id={} name='{}' "
              "native_active={}; store now has {} series: {}".format(
                  vid, vname, is_native, len(self._fusion_results),
                  [slicer.mrmlScene.GetNodeByID(k).GetName()
                   if slicer.mrmlScene.GetNodeByID(k) else k
                   for k in self._fusion_results]))
        if len(self._fusion_results) == 1 and not is_native:
            print("[DEBUG fusion.collect] WARNING: only 1 series and "
                  "native-series inference is OFF -- every result is keyed by "
                  "the same source volume, so fusing will not change anything. "
                  "Enable native-series inference and switch the working volume "
                  "to a different registered series for each plane.")
        self._refresh_native_series_inference_ui()

    def _invalidate_fusion_coverage(self, reason=None):
        """Drop cached per-series FOV coverage masks. Called whenever the output
        grid or a series' geometry (registration transform) changes, since the
        cached coverage is rasterized onto a now-stale grid."""
        if not self._fusion_coverage:
            return
        self._fusion_coverage = {}
        print("[DEBUG fusion.cover] coverage cache invalidated{}".format(
            "" if not reason else " ({})".format(reason)))

    def _series_coverage_mask(self, series_volume_id):
        """Bool mask on the current output grid: True where this series' FOV
        covers the voxel. Built once per (series id, output grid shape) by
        resampling an all-ones array of the series' shape onto the output grid
        via _resample_mask_between_volumes, then bit-packed in the cache (an
        unpacked uint8 trio would be hundreds of MB on large grids). Returns
        None when the node is gone or the resample fails; callers MUST treat
        None as FULL coverage so behavior degrades to the pre-fix semantics."""
        out_shape = self._output_grid_shape()
        if out_shape is None:
            return None
        cached = self._fusion_coverage.get(series_volume_id)
        if cached is not None and tuple(cached[1]) == tuple(out_shape):
            packed, shape = cached
            return (
                np.unpackbits(packed)[: int(np.prod(shape))]
                .reshape(shape)
                .astype(bool)
            )
        series_volume = slicer.mrmlScene.GetNodeByID(series_volume_id)
        output_volume = self.get_output_volume_node()
        if series_volume is None or output_volume is None:
            return None
        try:
            series_shape = slicer.util.arrayFromVolume(series_volume).shape
        except Exception:  # noqa: BLE001 - defensive around VTK accessors
            return None
        ones = np.ones(series_shape, dtype=np.uint8)
        cov = self._resample_mask_between_volumes(
            ones, series_volume, output_volume
        )
        if cov is None or tuple(cov.shape) != tuple(out_shape):
            print("[DEBUG fusion.cover] id={} name='{}' coverage build FAILED "
                  "-> treated as full coverage".format(
                      series_volume_id, series_volume.GetName()))
            return None
        cov_bool = cov.astype(bool)
        self._fusion_coverage[series_volume_id] = (
            np.packbits(cov_bool), tuple(out_shape)
        )
        total = int(np.prod(out_shape))
        nvox = int(cov_bool.sum())
        print("[DEBUG fusion.cover] id={} name='{}' coverage voxels={} / grid={} "
              "(frac={})".format(
                  series_volume_id, series_volume.GetName(), nvox, total,
                  round(nvox / float(total), 3) if total else 0.0))
        return cov_bool

    def _series_anisotropic_sigma(self, series_volume):
        """Per-output-axis Gaussian sigma (numpy z,y,x voxels) that blurs
        `series_volume` ONLY along the output-grid axis closest to its thick
        (slice-stacking) acquisition direction, leaving its sharp in-plane axes
        untouched. Oblique-tolerant: the slice direction is approximated to the
        nearest output axis (no axis-alignment requirement). Returns the
        (sz, sy, sx) sigma tuple, or None when geometry is unavailable. Prints the
        obliquity angle so the approximation quality can be judged."""
        output_volume = self.get_output_volume_node()
        if output_volume is None or series_volume is None:
            return None
        get_s = getattr(series_volume, "GetIJKToRASDirectionMatrix", None)
        get_o = getattr(output_volume, "GetIJKToRASDirectionMatrix", None)
        if get_s is None or get_o is None:
            return None
        sm = vtk.vtkMatrix4x4()
        get_s(sm)
        om = vtk.vtkMatrix4x4()
        get_o(om)
        s_spacing = tuple(series_volume.GetSpacing())  # (sI, sJ, sK)
        o_spacing = tuple(output_volume.GetSpacing())
        # Series slice axis = thickest IJK axis; take its RAS unit direction.
        slice_ijk = int(np.argmax(s_spacing))
        slice_dir = np.array(
            [sm.GetElement(r, slice_ijk) for r in range(3)], dtype=float
        )
        norm = float(np.linalg.norm(slice_dir))
        if norm == 0:
            return None
        slice_dir /= norm
        # Output numpy axes -> IJK columns: z=K(col2), y=J(col1), x=I(col0).
        out_col_for_np_axis = (2, 1, 0)
        best_np_axis, best_dot = 0, -1.0
        for np_axis, col in enumerate(out_col_for_np_axis):
            d = np.array([om.GetElement(r, col) for r in range(3)], dtype=float)
            dn = float(np.linalg.norm(d))
            if dn == 0:
                continue
            dot = abs(float(np.dot(slice_dir, d / dn)))
            if dot > best_dot:
                best_dot, best_np_axis = dot, np_axis
        out_axis_spacing = o_spacing[out_col_for_np_axis[best_np_axis]]
        ratio = (
            s_spacing[slice_ijk] / out_axis_spacing if out_axis_spacing > 0 else 1.0
        )
        # No smoothing where the series is as fine as (or finer than) the output
        # grid along that axis; grow with coarseness, capped to stay local.
        sigma_val = float(np.clip(0.5 * (ratio - 1.0), 0.0, 3.0))
        sigma = [0.0, 0.0, 0.0]  # numpy z, y, x
        sigma[best_np_axis] = sigma_val
        angle_deg = float(np.degrees(np.arccos(max(0.0, min(1.0, best_dot)))))
        print("[DEBUG triplanar.obliq] '{}' slice_ijk={} slice_dir_ras={} -> "
              "out_np_axis={} angle={}deg ratio={} sigma(z,y,x)={}".format(
                  series_volume.GetName(), slice_ijk,
                  [round(float(c), 3) for c in slice_dir], best_np_axis,
                  round(angle_deg, 1), round(ratio, 2),
                  tuple(round(s, 2) for s in sigma)))
        return tuple(sigma)

    def _fuse_series_results(self):
        """Direction-weighted SDF fusion of the collected per-series masks
        (output grid) into one smooth uint8 mask on the output grid, or None if
        nothing to fuse. Each series' signed distance field is blurred only along
        its own thick (stair-stepped) acquisition axis before averaging, so the
        series that is sharpest in a given plane dominates the fused boundary
        there and through-plane jaggies are suppressed."""
        try:
            from scipy import ndimage
        except Exception:
            print("[DEBUG fusion.fuse] SKIP: scipy unavailable")
            return None
        out_shape = self._output_grid_shape()
        if out_shape is None:
            print("[DEBUG fusion.fuse] SKIP: output grid shape is None")
            return None
        print("[DEBUG fusion.fuse] store has {} series, out_shape={}".format(
            len(self._fusion_results), tuple(out_shape)))
        for k, m in self._fusion_results.items():
            nm = slicer.mrmlScene.GetNodeByID(k)
            print("[DEBUG fusion.fuse]   series id={} name='{}' shape={} sum={}".format(
                k, nm.GetName() if nm else "<gone>",
                None if m is None else tuple(m.shape),
                None if m is None else int(m.sum())))
        output_volume = self.get_output_volume_node()
        spacing = output_volume.GetSpacing()  # (x, y, z) mm
        samp = (spacing[2], spacing[1], spacing[0])  # numpy (z, y, x)
        # Pre-interaction segment, used as the fallback wherever NO series covers
        # a voxel (so manual edits outside every FOV survive autofuse). Shape-
        # guarded against a stale previous_states from before a grid rebuild.
        baseline = self.previous_states.get("segment_data")
        if baseline is None or tuple(np.asarray(baseline).shape) != tuple(out_shape):
            baseline = self.get_segment_data(output_volume)
        if baseline is not None and tuple(np.asarray(baseline).shape) != tuple(out_shape):
            baseline = None
        sdf_sum = np.zeros(out_shape, dtype=np.float32)
        # Per-voxel vote count: how many series' FOV cover each voxel. A series
        # contributes its SDF only inside its own coverage, so a region imaged by
        # one series only is not voted away by the others' (out-of-FOV) zeros.
        cnt = np.zeros(out_shape, dtype=np.uint8)
        n = 0
        _perf_log("[DEBUG triplanar.perf] fuse: SDF start, out_voxels={}, series={}"
              .format(int(np.prod(out_shape)), len(self._fusion_results)),
              flush=True)
        for vid, m in list(self._fusion_results.items()):
            if slicer.mrmlScene.GetNodeByID(vid) is None:
                print("[DEBUG fusion.fuse] drop id={} (node deleted)".format(vid))
                del self._fusion_results[vid]
                continue
            if m is None or tuple(m.shape) != tuple(out_shape) or not m.any():
                continue
            mb = m.astype(bool)
            _perf_log("[DEBUG triplanar.perf] fuse: edt id={} start".format(vid),
                  flush=True)
            # Positive inside, negative outside (matches the convention used by
            # _interpolate_mask_to_output_grid).
            sdf = (
                ndimage.distance_transform_edt(mb, sampling=samp)
                - ndimage.distance_transform_edt(~mb, sampling=samp)
            ).astype(np.float32)
            _perf_log("[DEBUG triplanar.perf] fuse: edt id={} done".format(vid),
                  flush=True)
            # Direction weighting: blur this series' level set only along its own
            # thick (coarse) axis so its through-plane stair-steps drop out, while
            # its sharp in-plane edges survive and govern the fused surface there.
            sigma = self._series_anisotropic_sigma(slicer.mrmlScene.GetNodeByID(vid))
            if sigma is not None and any(s > 0 for s in sigma):
                sdf = ndimage.gaussian_filter(sdf, sigma=sigma).astype(np.float32)
                print("[DEBUG fusion.fuse]   id={} thick-axis blur sigma(z,y,x)="
                      "{}".format(vid, tuple(round(float(s), 2) for s in sigma)))
            else:
                print("[DEBUG fusion.fuse]   id={} no thick-axis blur "
                      "(isotropic / oblique)".format(vid))
            # Restrict this series' vote to its own FOV: blur first (so the hard
            # out-of-FOV zeros do not smear into valid in-plane boundaries), then
            # mask. None coverage == full coverage (pre-fix fallback).
            cov = self._series_coverage_mask(vid)
            if cov is not None:
                sdf *= cov
                cnt += cov.astype(np.uint8)
            else:
                cnt += 1
            sdf_sum += sdf
            n += 1
        if n == 0:
            print("[DEBUG fusion.fuse] SKIP: no usable masks after filtering")
            return None
        # Threshold the per-voxel mean level set at zero. Only the SIGN matters,
        # so dividing by the per-voxel count is unnecessary; the `& covered`
        # guard is mandatory because sdf_sum == 0 where nothing voted and a bare
        # `>= 0` would mark the whole uncovered region as inside.
        covered = cnt > 0
        fused = (sdf_sum >= 0) & covered
        kept = 0
        if baseline is not None:
            base_bool = np.asarray(baseline).astype(bool)
            uncovered_keep = base_bool & ~covered
            kept = int(uncovered_keep.sum())
            fused |= uncovered_keep
        fused = fused.astype(np.uint8)
        hist = np.bincount(cnt.ravel(), minlength=n + 1)
        print("[DEBUG fusion.fuse] {} usable series (direction-weighted); "
              "coverage histogram (votes->voxels)={}; uncovered kept from "
              "segment={}; fused voxels={}".format(
                  n, {i: int(c) for i, c in enumerate(hist) if c},
                  kept, int(fused.sum())))
        return fused

    def on_fuse_series_clicked(self, checked=False):
        """SDF-average all collected per-series results and write the smooth
        combined mask into the current segment (undoable)."""
        print("[DEBUG fusion.apply] Fuse clicked")
        fused = self._fuse_series_results()
        if fused is None:
            print("[DEBUG fusion.apply] RETURN: nothing to fuse")
            slicer.util.showStatusMessage(
                "Collect at least one series result first (run a prompt with "
                "Fuse enabled).",
                4000,
            )
            return
        seg_id = self.get_current_segment_id()
        pre = self.get_segment_data(self.get_output_volume_node())
        if pre is None:
            print("[DEBUG fusion.apply] RETURN: segment data unavailable")
            slicer.util.showStatusMessage(
                "Could not read the current segment; fuse was not applied.",
                4000,
            )
            return
        pre = pre.astype(np.uint8).copy()
        print("[DEBUG fusion.apply] segment before sum={}, fused sum={}, "
              "diff voxels={}".format(
                  int(pre.sum()), int(fused.sum()),
                  int(np.sum(pre.astype(bool) != fused.astype(bool)))))
        self._record_selection_op_undo(seg_id, pre)
        self.show_segmentation(fused)  # fused already on output grid (replace)
        self._destroy_inference_preview()
        self._fusion_results = {}
        self._refresh_native_series_inference_ui()
        self.upload_segment_to_server()
        print("[DEBUG fusion.apply] applied; segment written and uploaded")
        slicer.util.showStatusMessage("Fused series result applied.", 3000)

    def _on_series_fusion_toggled(self, checked):
        self._fusion_results = {}
        self._fusion_grid_shape = None
        self._invalidate_fusion_coverage("series fusion toggled")
        self._refresh_native_series_inference_ui()

    # -- One-click three-series fusion (lasso -> 3 series -> ROI fusion) ------

    def _get_active_lasso(self):
        """Return the current lasso polygon as an (N, 3) array of RAS world
        points, or None. Prefer the live lasso node; fall back to the cached
        points of the last submitted lasso (the live node is consumed/cleared by
        the normal auto-submit flow)."""
        node = self.prompt_types.get("lasso", {}).get("node")
        if node is not None:
            try:
                vtk_pts = node.GetCurvePointsWorld()
                if vtk_pts is not None:
                    pts = vtk_to_numpy(vtk_pts.GetData())
                    if pts is not None and len(pts) >= 3:
                        return np.array(pts)
            except Exception:  # noqa: BLE001 - defensive around VTK accessors
                pass
        cached = self._last_lasso_world_points
        if cached is not None and len(cached) >= 3:
            return np.array(cached)
        return None

    def _lasso_interior_seeds(self, lasso_ras, ref_grid_node,
                              n_seeds=FUSE3_NUM_SEEDS):
        """Derive positive 3D point seeds (RAS) from the lasso interior. The
        lasso is filled on ref_grid_node; the distance transform of the filled
        region yields the deepest interior point (first seed, safe even for
        concave / ring shapes) plus a few interior samples (depth above half the
        maximum, picked at an even stride for spread) so one seed cannot
        under-segment. Returns a list of [R, A, S] points (always >= 1)."""
        centroid = list(np.asarray(lasso_ras, dtype=float).mean(axis=0))
        try:
            from scipy import ndimage
        except Exception:
            return [centroid]
        try:
            xyzs = [
                self.ras_to_xyz(list(p), volume_node=ref_grid_node)
                for p in lasso_ras
            ]
            filled = self.lasso_points_to_mask(
                xyzs, ras_points=np.asarray(lasso_ras), volume_node=ref_grid_node
            )
        except Exception as e:  # noqa: BLE001
            print("[DEBUG fuse3] seed rasterization failed: {}".format(e))
            return [centroid]
        filled = np.asarray(filled).astype(bool)
        if not filled.any():
            return [centroid]
        dt = ndimage.distance_transform_edt(filled)
        kji = np.unravel_index(int(np.argmax(dt)), dt.shape)  # (k, j, i)
        seeds_kji = [tuple(int(c) for c in kji)]
        thr = 0.5 * float(dt.max())
        cand = np.argwhere(dt > thr)
        extra = max(0, int(n_seeds) - 1)
        if extra > 0 and len(cand) > 0:
            step = max(1, len(cand) // (extra + 1))
            for s in range(1, extra + 1):
                idx = min(len(cand) - 1, s * step)
                seeds_kji.append(tuple(int(c) for c in cand[idx]))
        ijk_to_ras = vtk.vtkMatrix4x4()
        ref_grid_node.GetIJKToRASMatrix(ijk_to_ras)
        tnode = ref_grid_node.GetParentTransformNode()
        local_to_world = None
        if tnode is not None:
            local_to_world = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                tnode, None, local_to_world
            )
        seeds_ras, seen = [], set()
        for (k, j, i) in seeds_kji:
            if (k, j, i) in seen:
                continue
            seen.add((k, j, i))
            ras = ijk_to_ras.MultiplyPoint([float(i), float(j), float(k), 1.0])
            p = [ras[0], ras[1], ras[2]]
            if local_to_world is not None:
                p = list(local_to_world.TransformPoint(p))
            seeds_ras.append([p[0], p[1], p[2]])
        print("[DEBUG fuse3] derived {} interior seed(s) from lasso".format(
            len(seeds_ras)))
        return seeds_ras if seeds_ras else [centroid]

    def _send_point_seeds_for_series(self, series, seeds_ras):
        """Send each in-bounds positive point seed to `series` via point_prompt.
        A seed mapping outside the series' voxel grid (its field of view) is
        skipped, so a series that does not image the lasso region simply gets no
        prompt. Returns True if at least one seed was sent."""
        try:
            shape = slicer.util.arrayFromVolume(series).shape  # (z, y, x)
        except Exception:  # noqa: BLE001
            return False
        depth, height, width = int(shape[0]), int(shape[1]), int(shape[2])
        sent = 0
        for ras in seeds_ras:
            xyz = self.ras_to_xyz(list(ras), volume_node=series)  # [i, j, k]
            if not (0 <= xyz[0] < width and 0 <= xyz[1] < height
                    and 0 <= xyz[2] < depth):
                continue
            self.point_prompt(xyz=xyz, positive_click=True)
            sent += 1
        print("[DEBUG fuse3] series '{}' received {} of {} seed(s)".format(
            series.GetName(), sent, len(seeds_ras)))
        return sent > 0

    def _get_lasso_roi_world(self, lassoPoints, margin_mm=FUSE3_ROI_MARGIN_MM):
        """RAS axis-aligned bounding box of the lasso, expanded by margin_mm.
        Returns (min_xyz, max_xyz) as length-3 lists."""
        pts = np.asarray(lassoPoints, dtype=float)
        mn = (pts.min(axis=0) - float(margin_mm)).tolist()
        mx = (pts.max(axis=0) + float(margin_mm)).tolist()
        return mn, mx

    def _ras_box_to_output_index_box(self, mn, mx):
        """Convert an RAS bounding box to a clamped numpy index sub-box
        (z0, z1, y0, y1, x0, x1) on the current output grid. Returns the full
        grid on any failure or a degenerate box, or None when no output grid."""
        out_shape = self._output_grid_shape()
        output_volume = self.get_output_volume_node()
        if out_shape is None or output_volume is None:
            return None
        D, H, W = int(out_shape[0]), int(out_shape[1]), int(out_shape[2])
        full = (0, D, 0, H, 0, W)
        try:
            ras_to_ijk = vtk.vtkMatrix4x4()
            output_volume.GetRASToIJKMatrix(ras_to_ijk)
            tnode = output_volume.GetParentTransformNode()
            world_to_local = None
            if tnode is not None:
                world_to_local = vtk.vtkGeneralTransform()
                slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                    None, tnode, world_to_local
                )
            iis, jjs, kks = [], [], []
            for rx in (mn[0], mx[0]):
                for ry in (mn[1], mx[1]):
                    for rz in (mn[2], mx[2]):
                        p = [rx, ry, rz]
                        if world_to_local is not None:
                            p = list(world_to_local.TransformPoint(p))
                        ijk = ras_to_ijk.MultiplyPoint([p[0], p[1], p[2], 1.0])
                        iis.append(ijk[0])
                        jjs.append(ijk[1])
                        kks.append(ijk[2])
            x0 = max(0, int(np.floor(min(iis))))
            x1 = min(W, int(np.ceil(max(iis))) + 1)
            y0 = max(0, int(np.floor(min(jjs))))
            y1 = min(H, int(np.ceil(max(jjs))) + 1)
            z0 = max(0, int(np.floor(min(kks))))
            z1 = min(D, int(np.ceil(max(kks))) + 1)
            if x1 <= x0 or y1 <= y0 or z1 <= z0:
                return full
            return (z0, z1, y0, y1, x0, x1)
        except Exception as e:  # noqa: BLE001
            print("[DEBUG fuse3] ROI index box failed: {}".format(e))
            return full

    def _get_series_valid_region(self, seriesNode, refGridNode):
        """FOV validity mask of seriesNode on the current output grid (True where
        the series images the voxel). Delegates to the cached coverage builder;
        refGridNode is expected to be the current output grid. Returns a bool
        array on the output grid, or None when coverage is unavailable (callers
        then treat the series as fully covering)."""
        if seriesNode is None:
            return None
        return self._series_coverage_mask(seriesNode.GetID())

    def _fuse_masks_in_roi(self, valid, roi_box, output_volume):
        """FOV-weighted, direction-weighted SDF fusion of per-series output-grid
        masks, computed only inside roi_box for speed. `valid` is a list of
        (series_node, output_grid_uint8_mask). Each series' signed distance field
        is blurred only along its own thick acquisition axis, then masked to its
        own field of view, so a region imaged by one series only is not voted away
        by the others' out-of-FOV zeros. Returns a full output-grid uint8 mask
        (fused inside the ROI, existing segment preserved outside), or None."""
        try:
            from scipy import ndimage
        except Exception:
            return None
        out_shape = self._output_grid_shape()
        if out_shape is None or roi_box is None:
            return None
        z0, z1, y0, y1, x0, x1 = roi_box
        spacing = output_volume.GetSpacing()  # (x, y, z) mm
        samp = (spacing[2], spacing[1], spacing[0])  # numpy (z, y, x)
        sub_shape = (z1 - z0, y1 - y0, x1 - x0)
        sdf_sum = np.zeros(sub_shape, dtype=np.float32)
        cnt = np.zeros(sub_shape, dtype=np.uint8)
        n = 0
        for series, m in valid:
            if m is None or tuple(np.asarray(m).shape) != tuple(out_shape):
                continue
            mb = np.asarray(m)[z0:z1, y0:y1, x0:x1].astype(bool)
            if not mb.any():
                continue
            sdf = (
                ndimage.distance_transform_edt(mb, sampling=samp)
                - ndimage.distance_transform_edt(~mb, sampling=samp)
            ).astype(np.float32)
            sigma = self._series_anisotropic_sigma(series)
            if sigma is not None and any(s > 0 for s in sigma):
                sdf = ndimage.gaussian_filter(sdf, sigma=sigma).astype(np.float32)
            cov = self._get_series_valid_region(series, output_volume)
            if cov is not None:
                covc = np.asarray(cov)[z0:z1, y0:y1, x0:x1]
                sdf *= covc
                cnt += covc.astype(np.uint8)
            else:
                cnt += 1
            sdf_sum += sdf
            n += 1
        if n == 0:
            return None
        fused_sub = ((sdf_sum >= 0) & (cnt > 0)).astype(np.uint8)
        baseline = self.get_segment_data(output_volume)
        if (baseline is None
                or tuple(np.asarray(baseline).shape) != tuple(out_shape)):
            final = np.zeros(out_shape, dtype=np.uint8)
        else:
            final = np.asarray(baseline).astype(np.uint8).copy()
        final[z0:z1, y0:y1, x0:x1] = fused_sub
        print("[DEBUG fuse3] fused {} series in ROI {} -> voxels={}".format(
            n, roi_box, int(final.sum())))
        return final

    def _export_to_vr(self, segmentId, smoothing=0.5):
        """Build a smoothed closed-surface (3D model) representation for the
        current segmentation and make it visible in 3D. `smoothing` sets the
        marching-cubes smoothing factor (0..1)."""
        seg_node, sel_id = self.get_selected_segmentation_node_and_segment_id()
        if seg_node is None:
            return
        if not segmentId:
            segmentId = sel_id
        seg = seg_node.GetSegmentation()
        try:
            seg.SetConversionParameter("Smoothing factor", str(smoothing))
        except Exception:  # noqa: BLE001
            pass
        try:
            seg.RemoveRepresentation(self._closed_surface_name())
        except Exception:  # noqa: BLE001
            pass
        seg_node.CreateClosedSurfaceRepresentation()
        dn = seg_node.GetDisplayNode()
        if dn is not None:
            dn.SetVisibility3D(True)
            if segmentId:
                try:
                    dn.SetSegmentVisibility3D(segmentId, True)
                except Exception:  # noqa: BLE001
                    pass
        print("[DEBUG fuse3] exported closed-surface VR model "
              "(smoothing={})".format(smoothing))

    def onFuseFromThreeSeriesWithROI(self, checked=False):
        """One-click: drive three co-registered series from a single lasso, fuse
        their results by field-of-view-weighted SDF averaging inside the lasso
        ROI, write the combined mask into the current segment, sync, and build a
        3D model. The source series (the lasso's own plane) uses the real lasso;
        the two orthogonal series use positive point seeds derived from the lasso
        interior (a lasso is degenerate off its own plane)."""
        print("[DEBUG fuse3] onFuseFromThreeSeriesWithROI clicked")
        # 1) Require a series assigned to each of Red/Yellow/Green.
        missing = [
            v for v, sel in PLANE_DISPLAY_SELECTOR_NAMES.items()
            if getattr(self.ui, sel).currentNode() is None
        ]
        if missing:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Three-series fusion",
                "Assign a series to each of the Red/Yellow/Green views in the "
                "Multi-plane display panel first.",
            )
            return
        # 2) Require a lasso (live or last-submitted).
        lasso = self._get_active_lasso()
        if lasso is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Three-series fusion",
                "Draw a region with the lasso (shortcut L) first.",
            )
            return
        # 3) Ensure tri-planar prerequisites (union output grid covering all 3
        #    series + high-resolution output + fusion). Enabling the mode builds
        #    the grid, so we never clip anatomy outside the editor source FOV.
        if not self._triplanar_mode:
            self.ui.cbTriPlanarMode.setChecked(True)
        output_volume = self.get_output_volume_node()
        if output_volume is None or self._output_grid_shape() is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Three-series fusion",
                "Could not prepare the high-resolution output grid. Make sure "
                "the three series are loaded and assigned.",
            )
            return
        # 4) Total ROI = lasso bounding box + margin, as an output-grid sub-box.
        mn, mx = self._get_lasso_roi_world(lasso, FUSE3_ROI_MARGIN_MM)
        roi_box = self._ras_box_to_output_index_box(mn, mx)
        if roi_box is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Three-series fusion",
                "Could not map the lasso ROI onto the output grid.",
            )
            return
        # 5) Source series (the lasso plane's view) and derived point seeds.
        centroid = list(np.asarray(lasso).mean(axis=0))
        src_view = self._view_for_ras_point(centroid)
        src_series = (
            self._view_background_volume(src_view)
            if src_view is not None else None
        )
        seeds = self._lasso_interior_seeds(lasso, output_volume)
        print("[DEBUG fuse3] source view={}, src_series={}, seeds={}".format(
            src_view, src_series.GetName() if src_series else None, len(seeds)))
        # 6) Hybrid inference in capture mode: source -> lasso, others -> seeds.
        self._fusion_capture_active = True
        self._fusion_capture_store = {}
        valid = []
        seen = set()
        try:
            for v in ("Red", "Yellow", "Green"):
                series = self._view_background_volume(v)
                if series is None or series.GetID() in seen:
                    continue
                seen.add(series.GetID())
                self._active_inference_volume_override = series
                try:
                    is_source = (
                        src_series is not None
                        and series.GetID() == src_series.GetID()
                    )
                    if is_source:
                        xyzs = [
                            self.ras_to_xyz(list(p), volume_node=series)
                            for p in lasso
                        ]
                        mask = self.lasso_points_to_mask(
                            xyzs, ras_points=np.asarray(lasso),
                            volume_node=series,
                        )
                        self.lasso_or_scribble_prompt(
                            mask=mask, positive_click=True, tp="lasso",
                            mask_volume_node=series,
                        )
                    else:
                        if not self._send_point_seeds_for_series(series, seeds):
                            print("[DEBUG fuse3] '{}' got no in-FOV seeds; "
                                  "skipping".format(series.GetName()))
                            continue
                    res = self._fusion_capture_store.get(series.GetID())
                    if res is not None and res.any():
                        valid.append((series, res))
                except Exception as e:  # noqa: BLE001
                    print("[DEBUG fuse3] skip series '{}': {}".format(
                        series.GetName(), e))
        finally:
            self._fusion_capture_active = False
            self._active_inference_volume_override = None
        # 7) Need at least two contributing series to fuse.
        if len(valid) < 2:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Three-series fusion",
                "Fewer than 2 series produced a result; fusion aborted.",
            )
            return
        # 8) Fuse inside the ROI, apply (undoable), sync, build the 3D model.
        final = self._fuse_masks_in_roi(valid, roi_box, output_volume)
        if final is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Three-series fusion",
                "Fusion produced no result.",
            )
            return
        seg_id = self.get_current_segment_id()
        pre = self.get_segment_data(output_volume)
        if pre is not None:
            self._record_selection_op_undo(
                seg_id, np.asarray(pre).astype(np.uint8).copy()
            )
        # The fused mask is a full 3D result; never let a stale lasso slice
        # marker clip it to a single slice in show_segmentation.
        self._last_lasso_slice = None
        self.show_segmentation(final)
        self._fusion_results = {}
        self.upload_segment_to_server()
        self._export_to_vr(seg_id, smoothing=0.5)
        slicer.util.showStatusMessage(
            "Fused {} series and exported a 3D model.".format(len(valid)), 4000
        )
        print("[DEBUG fuse3] done; fused {} series".format(len(valid)))

    # -- Tri-planar multi-series mode ----------------------------------------

    def _get_triplanar_enabled(self):
        """Read the persisted tri-planar mode preference. Default False."""
        return self._get_qsetting(SETTING_TRIPLANAR_ENABLED, False, cast=bool)

    # -- Tri-planar 3D slice locator frames ----------------------------------

    def _slice_frame_geometry(self, slice_node, volume):
        """Return (center, u, v, n, hu, hv) for a view's locator plane, or None.

        center: RAS center of the plane on the current slice offset; u/v/n: the
        normalized in-plane axes and normal from SliceToRAS; hu/hv: half-widths
        sized to `volume`'s RAS bounding box (falls back to the view field of
        view when the volume is missing/degenerate). Shared by the polydata
        builder and the camera-rotation feature so both stay consistent."""
        m = slice_node.GetSliceToRAS()
        u = [m.GetElement(i, 0) for i in range(3)]
        v = [m.GetElement(i, 1) for i in range(3)]
        n = [m.GetElement(i, 2) for i in range(3)]
        origin = [m.GetElement(i, 3) for i in range(3)]

        def _normalize(a):
            length = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
            if length < 1e-9:
                return None
            return [a[0] / length, a[1] / length, a[2] / length]

        u = _normalize(u)
        v = _normalize(v)
        n = _normalize(n)
        if u is None or v is None or n is None:
            return None

        hu = hv = 0.0
        center = list(origin)
        have_bounds = False
        if volume is not None:
            bounds = [0.0] * 6
            volume.GetRASBounds(bounds)
            ext = [bounds[1] - bounds[0], bounds[3] - bounds[2],
                   bounds[5] - bounds[4]]
            if min(ext) > 1e-6:
                have_bounds = True
                # Half span of an axis-aligned box projected onto a unit
                # direction d is 0.5 * sum(|d_axis| * extent_axis).
                hu = 0.5 * (abs(u[0]) * ext[0] + abs(u[1]) * ext[1]
                            + abs(u[2]) * ext[2])
                hv = 0.5 * (abs(v[0]) * ext[0] + abs(v[1]) * ext[1]
                            + abs(v[2]) * ext[2])
                # Project the box center onto the live slice plane (through
                # origin with normal n) so the frame rides the slice offset.
                box_c = [0.5 * (bounds[0] + bounds[1]),
                         0.5 * (bounds[2] + bounds[3]),
                         0.5 * (bounds[4] + bounds[5])]
                d = [box_c[i] - origin[i] for i in range(3)]
                dist = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]
                center = [box_c[i] - dist * n[i] for i in range(3)]
        if not have_bounds:
            # Fall back to the (zoom-dependent) FOV about the view center.
            fov = slice_node.GetFieldOfView()
            hu = 0.5 * float(fov[0])
            hv = 0.5 * float(fov[1])
            center = list(origin)
        if hu < 1e-6 or hv < 1e-6:
            return None
        return center, u, v, n, hu, hv

    def _make_slice_frame_polydata(self, slice_node, volume):
        """Build a filled translucent quad + closed border (vtkPolyData) lying
        on the current slice plane of `slice_node`, sized to the RAS bounding
        box of `volume` (its own series). The plane slides along the slice
        normal as the user scrolls and rotates with oblique re-orientation
        because orientation/offset are read live from SliceToRAS. Falls back to
        the view's field of view when `volume` is missing or degenerate.
        Returns None when the plane axes or extents are degenerate."""
        geometry = self._slice_frame_geometry(slice_node, volume)
        if geometry is None:
            return None
        center, u, v, _n, hu, hv = geometry

        points = vtk.vtkPoints()
        for su, sv in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
            points.InsertNextPoint(
                center[0] + su * hu * u[0] + sv * hv * v[0],
                center[1] + su * hu * u[1] + sv * hv * v[1],
                center[2] + su * hu * u[2] + sv * hv * v[2],
            )
        # Closed border line (drawn crisply regardless of surface representation).
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(5)
        for idx, pid in enumerate((0, 1, 2, 3, 0)):
            line.GetPointIds().SetId(idx, pid)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)
        # Filled quad (the locator "plane"); rendered as a translucent surface.
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        for idx in range(4):
            polygon.GetPointIds().SetId(idx, idx)
        polys = vtk.vtkCellArray()
        polys.InsertNextCell(polygon)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.SetPolys(polys)
        return poly

    def _get_frame_color(self, view_name, slice_node):
        """RGB (0..1) for a view's 3D locator frame: the slice node's layout
        color when available (matching the 2D intersection line), else a fixed
        per-view fallback."""
        try:
            color = slice_node.GetLayoutColor()
            if color is not None and len(color) == 3:
                return (float(color[0]), float(color[1]), float(color[2]))
        except Exception:  # noqa: BLE001 - defensive around optional accessor
            pass
        return TRIPLANAR_FRAME_FALLBACK_COLORS.get(view_name, (1.0, 1.0, 1.0))

    def _get_frame_visible(self, view_name):
        """Persisted per-view visibility of the locator plane (default True)."""
        return self._get_qsetting(
            SETTING_TRIPLANAR_FRAME_VISIBLE_PREFIX + view_name.lower(),
            True, cast=bool)

    def _set_frame_visible(self, view_name, visible):
        """Persist a view's locator-plane visibility preference."""
        self._set_qsetting(
            SETTING_TRIPLANAR_FRAME_VISIBLE_PREFIX + view_name.lower(),
            bool(visible))

    def _get_plane_opacity(self):
        """Persisted locator-plane fill opacity (0..1). Default from constant."""
        v = self._get_qsetting(
            SETTING_PLANE_OPACITY, TRIPLANAR_FRAME_OPACITY, cast=float)
        return max(0.0, min(1.0, v))

    def _on_plane_opacity_changed(self, value):
        """Slider drag -- set every locator plane's opacity and persist it."""
        pct = int(value)
        if hasattr(self, "ui") and hasattr(self.ui, "lblPlaneOpacityValue"):
            self.ui.lblPlaneOpacityValue.setText("%d %%" % pct)
        fraction = float(pct) / 100.0
        self._set_qsetting(SETTING_PLANE_OPACITY, fraction)
        for model_node in self._triplanar_frame_nodes.values():
            if model_node is None or not slicer.mrmlScene.IsNodePresent(
                    model_node):
                continue
            display_node = model_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetOpacity(fraction)
        self._force_render_3d_views()

    # -- Auto-rotate the 3D camera to face a series ---------------------------

    def _get_auto_camera_rotation(self):
        """Persisted auto-rotate-camera-on-interaction preference (default On)."""
        return self._get_qsetting(
            SETTING_AUTO_CAMERA_ROTATION, True, cast=bool)

    def _on_auto_camera_rotation_toggled(self, checked):
        """Persist the auto-rotate toggle."""
        self._set_qsetting(SETTING_AUTO_CAMERA_ROTATION, bool(checked))

    def _get_oblique_camera_align(self):
        """Persisted 'align camera to series acquisition frame' pref (default On)."""
        return self._get_qsetting(
            SETTING_OBLIQUE_CAMERA_ALIGN, True, cast=bool)

    def _on_oblique_camera_align_toggled(self, checked):
        """Persist the oblique-camera-align toggle."""
        self._set_qsetting(SETTING_OBLIQUE_CAMERA_ALIGN, bool(checked))

    def _volume_acquisition_frame(self, volume):
        """Return (center, i_axis, j_axis, k_axis) in world/RAS for a volume's
        acquisition frame, or None.

        i/j/k are the unit IJK column directions of GetIJKToRASDirectionMatrix
        rotated into world space by the volume's parent transform (so a series
        made oblique purely by a registration transform is handled). j is the
        image column direction (view-up); k is the slice-stacking direction
        (the plane normal). center is the world bounding-box center
        (GetRASBounds applies the parent transform). None on a missing or
        degenerate direction matrix."""
        if volume is None:
            return None
        get_dirs = getattr(volume, "GetIJKToRASDirectionMatrix", None)
        if get_dirs is None:
            return None
        dm = vtk.vtkMatrix4x4()
        get_dirs(dm)
        cols = [[dm.GetElement(r, c) for r in range(3)] for c in range(3)]
        # Rotate the direction vectors into world space via the parent
        # transform. Subtracting the transformed origin keeps only rotation
        # (no translation) -- the correct operation for a direction vector.
        tnode = volume.GetParentTransformNode()
        if tnode is not None:
            local_to_world = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                tnode, None, local_to_world)
            origin_w = list(local_to_world.TransformPoint([0.0, 0.0, 0.0]))
            rotated = []
            for vec in cols:
                pw = list(local_to_world.TransformPoint(vec))
                rotated.append([pw[a] - origin_w[a] for a in range(3)])
            cols = rotated

        def _norm(a):
            length = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
            if length < 1e-9:
                return None
            return [a[0] / length, a[1] / length, a[2] / length]

        i_axis, j_axis, k_axis = _norm(cols[0]), _norm(cols[1]), _norm(cols[2])
        if i_axis is None or j_axis is None or k_axis is None:
            return None
        bounds = [0.0] * 6  # GetRASBounds applies the parent transform (world).
        volume.GetRASBounds(bounds)
        center = [0.5 * (bounds[0] + bounds[1]),
                  0.5 * (bounds[2] + bounds[3]),
                  0.5 * (bounds[4] + bounds[5])]
        return center, i_axis, j_axis, k_axis

    def _has_oblique_world_frame(self, volume):
        """True when the volume is oblique in world space: either its own
        direction matrix is oblique, or a parent transform tilts its axes off
        RAS. Plain _volume_is_oblique only inspects the direction matrix, so it
        misses series made oblique purely by a registration transform -- the
        very case this camera alignment targets."""
        if volume is None:
            return False
        if self._volume_is_oblique(volume):
            return True
        frame = self._volume_acquisition_frame(volume)
        if frame is None:
            return False
        _center, i_axis, j_axis, k_axis = frame
        for axis in (i_axis, j_axis, k_axis):
            if max(abs(axis[0]), abs(axis[1]),
                   abs(axis[2])) < OBLIQUE_COS_THRESHOLD:
                return True
        return False

    def _slice_view_screen_up(self, view_name):
        """The clicked 2D slice view's on-screen up direction in world RAS
        (its SliceToRAS column 1), or None when the view/slice is unavailable.
        Used to roll the 3D camera so the 2D view's up appears up in 3D."""
        layout_manager = slicer.app.layoutManager()
        slice_widget = (
            layout_manager.sliceWidget(view_name)
            if layout_manager is not None else None)
        slice_node = (
            slice_widget.mrmlSliceNode() if slice_widget is not None else None)
        if slice_node is None:
            return None
        m = slice_node.GetSliceToRAS()
        return [m.GetElement(i, 1) for i in range(3)]

    def _slice_view_toward_viewer(self, view_name):
        """The direction the clicked 2D slice view is looked at FROM, in world
        RAS: its SliceToRAS screen-right (column 0) cross screen-up (column 1).
        This equals +column2 for an un-flipped slice and -column2 for a
        radiologically left-right-flipped one (e.g. the default Red axial), so
        it always points to the real viewer side regardless of the matrix
        handedness. None when the view/slice is unavailable or degenerate."""
        layout_manager = slicer.app.layoutManager()
        slice_widget = (
            layout_manager.sliceWidget(view_name)
            if layout_manager is not None else None)
        slice_node = (
            slice_widget.mrmlSliceNode() if slice_widget is not None else None)
        if slice_node is None:
            return None
        m = slice_node.GetSliceToRAS()
        right = [m.GetElement(i, 0) for i in range(3)]
        up = [m.GetElement(i, 1) for i in range(3)]
        toward = [right[1] * up[2] - right[2] * up[1],
                  right[2] * up[0] - right[0] * up[2],
                  right[0] * up[1] - right[1] * up[0]]
        length = (toward[0] ** 2 + toward[1] ** 2 + toward[2] ** 2) ** 0.5
        if length < 1e-9:
            return None
        return [toward[0] / length, toward[1] / length, toward[2] / length]

    def _view_world_plane(self, view_name):
        """World-RAS (normal, up_candidate) unit vectors for one view's plane,
        or None.

        Prefers the series' acquisition frame (k = plane normal, j = image
        column / view-up) when oblique alignment is on and the series is oblique
        in world space; otherwise reads the 2D slice plane (SliceToRAS n/v).
        Shared by the camera 'face view' logic so every view's normal comes from
        the same source."""
        layout_manager = slicer.app.layoutManager()
        slice_widget = (
            layout_manager.sliceWidget(view_name)
            if layout_manager is not None else None)
        slice_node = (
            slice_widget.mrmlSliceNode() if slice_widget is not None else None)
        volume = self._view_background_volume(view_name)
        if (self._get_oblique_camera_align()
                and volume is not None
                and self._has_oblique_world_frame(volume)):
            frame = self._volume_acquisition_frame(volume)
            if frame is not None:
                _center, _i_axis, j_axis, k_axis = frame
                return list(k_axis), list(j_axis)
        if slice_node is None:
            return None
        geometry = self._slice_frame_geometry(slice_node, volume)
        if geometry is None:
            return None
        _center, _u, v, n, _hu, _hv = geometry
        return list(n), list(v)

    def _rotate_camera_to_view(self, view_name):
        """Orthographically 'face' the plane of `view_name` along the line where
        the OTHER two views' planes intersect.

        Forward = normalize(cross(normal_of_other_a, normal_of_other_b)) -- e.g.
        facing Red looks along cross(Yellow_normal, Green_normal). For mutually
        orthogonal planes this equals the clicked view's own normal; it differs
        only for oblique / multi-series planes. The sign is fixed by the clicked
        view's normal on its preferred side (Red=above, Yellow=front, Green=left)
        so the camera never flips 180 deg between calls. The current focal point
        and distance are kept (pure reorientation); view-up is the clicked view's
        2D screen-up (its SliceToRAS column 1) rolled about the view axis, so the
        3D up matches what the 2D view shows, re-orthogonalized against the
        forward. No-op without a 3D view/camera or valid plane geometry."""
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None or layout_manager.threeDViewCount == 0:
            return
        view_node = layout_manager.threeDWidget(0).mrmlViewNode()
        try:
            cameras_logic = slicer.modules.cameras.logic()
            camera_node = cameras_logic.GetViewActiveCameraNode(view_node)
        except Exception:  # noqa: BLE001 - cameras module/camera not available
            camera_node = None
        if camera_node is None:
            return
        camera = camera_node.GetCamera()

        def _dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def _cross(a, b):
            return [a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]]

        def _unit(a):
            length = _dot(a, a) ** 0.5
            if length < 1e-9:
                return None
            return [a[0] / length, a[1] / length, a[2] / length]

        own = self._view_world_plane(view_name)
        if own is None:
            return
        own_normal, up_candidate = own

        others = [v for v in ("Red", "Yellow", "Green") if v != view_name]
        plane_a = self._view_world_plane(others[0])
        plane_b = self._view_world_plane(others[1])
        target = None
        if plane_a is not None and plane_b is not None:
            crossed = _cross(plane_a[0], plane_b[0])
            if _dot(crossed, crossed) ** 0.5 >= 1e-3:  # the two planes diverge
                target = _unit(crossed)
        if target is None:  # single/parallel series -> face the clicked plane
            target = _unit(own_normal)
            if target is None:
                return

        # Sign: align to the clicked view's normal on its conventional side so
        # the result is deterministic (no 180 deg flip between invocations).
        pref = TRIPLANAR_CAMERA_PREFERRED_SIDE.get(view_name, (0.0, 0.0, 1.0))
        on = _unit(own_normal) or [0.0, 0.0, 1.0]
        if _dot(on, pref) < 0:
            on = [-on[0], -on[1], -on[2]]
        if _dot(target, on) < 0:
            target = [-target[0], -target[1], -target[2]]
        cam_axis = target  # focal -> camera direction (camera sits on this side)
        forward = [-cam_axis[0], -cam_axis[1], -cam_axis[2]]

        # Front/back: put the camera on the side the 2D view is looked at from --
        # its screen-right x screen-up (SliceToRAS col0 x col1). That is the true
        # viewer side for every view, including the radiologically flipped Red
        # axial (where col0 x col1 = -col2), so all three faces reproduce their
        # 2D view's front/back (and left/right). Not the fixed preferred side.
        toward_viewer = self._slice_view_toward_viewer(view_name)
        if toward_viewer is not None and _dot(cam_axis, toward_viewer) < 0:
            cam_axis = [-cam_axis[0], -cam_axis[1], -cam_axis[2]]
            forward = [-cam_axis[0], -cam_axis[1], -cam_axis[2]]

        # View-up: roll about the view axis so the clicked view's 2D screen-up
        # (its SliceToRAS column 1) appears up in 3D. Read straight from the
        # slice (natural sign = what is shown); no Superior correction. Projected
        # orthogonal to forward below.
        up = self._slice_view_screen_up(view_name) or list(up_candidate)
        proj = _dot(up, forward)
        up = _unit([up[i] - forward[i] * proj for i in range(3)])
        if up is None:  # column parallel to forward: pick any off-axis vector
            fallback = ([0.0, 0.0, 1.0] if abs(forward[2]) < 0.9
                        else [0.0, 1.0, 0.0])
            proj = _dot(fallback, forward)
            up = _unit([fallback[i] - forward[i] * proj for i in range(3)])
        if up is None:
            return
        # Gram-Schmidt so the basis stays orthonormal (no roll/shear).
        right = _cross(forward, up)
        up = _unit(_cross(right, forward)) or up

        focal = list(camera.GetFocalPoint())
        distance = camera.GetDistance()
        if distance <= 1e-6:  # degenerate camera: fall back to a safe distance
            distance = 300.0
        pos = [focal[i] + cam_axis[i] * distance for i in range(3)]

        camera_node.SetFocalPoint(focal[0], focal[1], focal[2])
        camera_node.SetPosition(pos[0], pos[1], pos[2])
        camera_node.SetViewUp(up[0], up[1], up[2])
        try:
            renderer = (layout_manager.threeDWidget(0).threeDView()
                        .renderWindow().GetRenderers().GetFirstRenderer())
            renderer.ResetCameraClippingRange()
        except Exception:  # noqa: BLE001 - clipping reset is a convenience
            pass
        self._force_render_3d_views()

    # -- Show the segmentation 3D surface in tri-planar (debounced) -----------

    def _get_show_3d_triplanar(self):
        """Persisted 'show segmentation in 3D while tri-planar' pref (default On)."""
        return self._get_qsetting(SETTING_SHOW_3D_TRIPLANAR, True, cast=bool)

    def _on_show_3d_triplanar_toggled(self, checked):
        """Persist the toggle; build the surface now when turned on, hide the 3D
        surface when turned off (2D fill and locator planes are untouched)."""
        self._set_qsetting(SETTING_SHOW_3D_TRIPLANAR, bool(checked))
        if checked:
            self._schedule_triplanar_3d_surface()
            return
        seg_node = self._existing_segmentation_node()
        if seg_node is None:
            return
        display_node = seg_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetVisibility3D(False)
        self._force_render_3d_views()

    def _schedule_triplanar_3d_surface(self):
        """Debounce the (heavy) closed-surface rebuild: only the last schedule in
        a burst of rapid interactions actually runs, once the user pauses."""
        self._tp3d_token = getattr(self, "_tp3d_token", 0) + 1
        token = self._tp3d_token
        qt.QTimer.singleShot(
            TRIPLANAR_3D_SURFACE_DEBOUNCE_MS,
            lambda t=token: self._rebuild_triplanar_3d_surface(t))

    def _rebuild_triplanar_3d_surface(self, token):
        """Rebuild + show the current segment's 3D closed surface. No-op if a
        newer interaction has been scheduled since (token mismatch), if not in
        tri-planar mode, or if the feature is off."""
        if token != getattr(self, "_tp3d_token", 0):
            return
        if not self._triplanar_mode or not self._get_show_3d_triplanar():
            return
        seg_node = self._existing_segmentation_node()
        if seg_node is None:
            return
        seg_id = self.get_current_segment_id()
        display_node = seg_node.GetDisplayNode()
        try:
            segmentation = seg_node.GetSegmentation()
            if self._get_display_smooth_enabled():
                segmentation.SetConversionParameter(
                    "Smoothing factor",
                    str(self._current_display_smooth_strength()))
            # Drop the stale surface so it is rebuilt from the current labelmap.
            segmentation.RemoveRepresentation(self._closed_surface_name())
            seg_node.CreateClosedSurfaceRepresentation()
            if display_node is not None:
                display_node.SetVisibility3D(True)
                if seg_id:
                    display_node.SetSegmentVisibility3D(seg_id, True)
        except Exception as exc:  # noqa: BLE001 - never break the prompt flow
            print("[nni] tri-planar 3D surface build failed: %s" % exc)
            return
        self._force_render_3d_views()

    def _ensure_triplanar_slice_frames(self, reason=""):
        """Build the locator planes if we are in tri-planar mode, they are not
        built yet, and at least two distinct series are assigned to the views.

        Covers the lazy-restore gap: when tri-planar is restored as CHECKED at
        startup the toggle->setup path that builds the planes never runs, so the
        planes are created here instead the moment series get applied. Idempotent
        and cheap: a no-op once the planes exist (their geometry is kept current
        by the slice-node observers)."""
        if not self._triplanar_mode:
            return
        if self._triplanar_frame_nodes:
            return
        bg = {v: self._view_background_volume(v)
              for v in PLANE_DISPLAY_SELECTOR_NAMES}
        distinct = {n.GetID() for n in bg.values() if n is not None}
        if len(distinct) < 2:
            return
        self._enable_triplanar_slice_frames()

    def _enable_triplanar_slice_frames(self):
        """Create one hidden filled-plane model per realized standard view and
        observe its slice node so the 3D locator plane follows scroll/rotation.
        Idempotent: any existing frames/observers are torn down first."""
        self._disable_triplanar_slice_frames()
        for view_name, _slice_logic, slice_node in (
            self._iter_standard_slice_logics()
        ):
            model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            model_node.SetName(TRIPLANAR_FRAME_NODE_NAMES[view_name])
            model_node.HideFromEditorsOn()
            # Not pickable, so placing a markup (e.g. a magic wand seed) in the
            # 3D view snaps to the segmentation surface, not these planes.
            model_node.SetSelectable(False)
            model_node.CreateDefaultDisplayNodes()
            display_node = model_node.GetDisplayNode()
            if display_node is not None:
                color = self._get_frame_color(view_name, slice_node)
                display_node.SetColor(*color)
                display_node.SetEdgeColor(*color)
                display_node.SetLineWidth(2)
                # Translucent filled plane so the segmentation shows through;
                # the colored border (line cells) is drawn on top. Opacity is
                # user-adjustable via the sldPlaneOpacity slider.
                display_node.SetOpacity(self._get_plane_opacity())
                display_node.SetBackfaceCulling(False)
                # Flat color (no shading) for a clean translucent sheet.
                display_node.SetLighting(False)
                # 2D views already draw slice intersection lines; only show the
                # plane in 3D.
                display_node.SetVisibility2D(False)
                display_node.SetVisibility3D(True)
                # Per-view show/hide, controlled by the overlay R/Y/G buttons.
                display_node.SetVisibility(self._get_frame_visible(view_name))
            self._triplanar_frame_nodes[view_name] = model_node
            # Single shared callback: each slice Modified refreshes all frames.
            callback = lambda caller, event: (
                self._update_triplanar_slice_frames()
            )
            tag = slice_node.AddObserver(vtk.vtkCommand.ModifiedEvent, callback)
            self._triplanar_frame_observers.append((slice_node, tag))
        self._update_triplanar_slice_frames()
        # Overlay the R/Y/G show/hide buttons on the 3D view.
        self._create_triplanar_frame_buttons()
        # On the FIRST entry the 3D view's model displayable manager has not yet
        # built actors for the just-added nodes (the scene is still settling
        # from the heavy setup), so the synchronous render above paints nothing
        # and the planes only appeared after toggling the mode off/on. Re-render
        # on later event-loop turns, once actors exist (matches the off/on fix
        # automatically). Same singleShot idiom used elsewhere in this file.
        # On the FIRST entry the 3D view's model displayable manager may not have
        # built actors for the just-added nodes yet (the scene is still settling
        # from the heavy setup), so the synchronous render above can paint
        # nothing and the planes would only appear after an off/on toggle.
        # Re-render on later event-loop turns, once actors exist.
        for delay_ms in (0, 250):
            qt.QTimer.singleShot(delay_ms, self._deferred_frame_render)

    def _deferred_frame_render(self):
        """Deferred re-render so the locator planes show on first entry without
        an off/on toggle. No-op if tri-planar was left in the meantime."""
        if not (self._triplanar_mode and self._triplanar_frame_nodes):
            return
        self._update_triplanar_slice_frames()
        bar = getattr(self, "_triplanar_frame_button_bar", None)
        if bar is not None:
            bar.raise_()

    def _update_triplanar_slice_frames(self):
        """Refresh every locator plane's geometry + color from its slice node's
        current orientation/offset. Cheap (4 points each); safe on every slice
        ModifiedEvent. Never modifies slice nodes, so there is no observer
        recursion (do NOT call enable_slice_intersections here -- it would call
        Modified on the slice nodes and re-fire this callback in a loop)."""
        if not self._triplanar_frame_nodes:
            return
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return
        for view_name, model_node in list(self._triplanar_frame_nodes.items()):
            if (model_node is None
                    or not slicer.mrmlScene.IsNodePresent(model_node)):
                continue
            slice_widget = layout_manager.sliceWidget(view_name)
            if slice_widget is None:
                continue
            slice_node = slice_widget.mrmlSliceNode()
            if slice_node is None:
                continue
            poly = self._make_slice_frame_polydata(
                slice_node, self._view_background_volume(view_name))
            if poly is None:
                continue
            model_node.SetAndObservePolyData(poly)
            display_node = model_node.GetDisplayNode()
            if display_node is not None:
                color = self._get_frame_color(view_name, slice_node)
                display_node.SetColor(*color)
                display_node.SetEdgeColor(*color)
                # Keep each plane's persisted show/hide state across rebuilds.
                display_node.SetVisibility(self._get_frame_visible(view_name))
        # Programmatic polydata changes do not always repaint the 3D view on
        # their own (e.g. built at the end of _setup_triplanar_views), so force
        # a render; the plane actors are cheap.
        self._force_render_3d_views()

    def _force_render_3d_views(self):
        """Force an immediate repaint of every 3D view. Needed because model
        polydata set programmatically does not always trigger a 3D render."""
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return
        for i in range(layout_manager.threeDViewCount):
            widget = layout_manager.threeDWidget(i)
            if widget is not None:
                widget.threeDView().forceRender()

    def _disable_triplanar_slice_frames(self):
        """Tear down the 3D locator planes: drop observers, remove the model
        nodes, destroy the overlay buttons, then clear the containers (all or it
        leaks/leaves ghosts)."""
        for slice_node, tag in getattr(self, "_triplanar_frame_observers", []):
            if slice_node is not None:
                slice_node.RemoveObserver(tag)
        self._triplanar_frame_observers = []
        for model_node in getattr(self, "_triplanar_frame_nodes", {}).values():
            if (model_node is not None
                    and slicer.mrmlScene.IsNodePresent(model_node)):
                slicer.mrmlScene.RemoveNode(model_node)
        self._triplanar_frame_nodes = {}
        self._destroy_triplanar_frame_buttons()

    # -- Overlay R/Y/G show/hide buttons (top-left of the 3D view) -----------

    def _style_frame_button(self, button, view_name, checked):
        """Color a plane toggle button: lit (filled view color) when the plane
        is shown, dimmed (dark with colored text/border) when hidden."""
        color = TRIPLANAR_FRAME_FALLBACK_COLORS.get(view_name, (1.0, 1.0, 1.0))
        r, g, b = (int(round(255 * c)) for c in color)
        if checked:
            button.setStyleSheet(
                "QToolButton { background-color: rgb(%d,%d,%d); color: black; "
                "border: 1px solid black; border-radius: 3px; "
                "font-weight: bold; padding: 2px; }" % (r, g, b))
        else:
            button.setStyleSheet(
                "QToolButton { background-color: rgb(60,60,60); "
                "color: rgb(%d,%d,%d); border: 1px solid rgb(%d,%d,%d); "
                "border-radius: 3px; padding: 2px; }" % (r, g, b, r, g, b))

    def _create_triplanar_frame_buttons(self):
        """Overlay three small colored R/Y/G toggle buttons at the top-left of
        the first 3D view, one per locator plane. Idempotent."""
        self._destroy_triplanar_frame_buttons()
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None or layout_manager.threeDViewCount == 0:
            return
        widget = layout_manager.threeDWidget(0)
        view = widget.threeDView() if widget is not None else None
        if view is None:
            return
        bar = qt.QWidget(view)
        bar.setObjectName("TriPlanarFrameButtonBar")
        # Transparent container so only the buttons paint over the 3D render.
        bar.setStyleSheet(
            "QWidget#TriPlanarFrameButtonBar { background: transparent; }")
        layout = qt.QHBoxLayout(bar)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)
        for view_name in PLANE_DISPLAY_SELECTOR_NAMES:
            button = qt.QToolButton(bar)
            button.setText(view_name[0])
            button.setCheckable(True)
            checked = self._get_frame_visible(view_name)
            button.setChecked(checked)
            button.setToolTip(
                "Show/hide the %s locator plane in 3D" % view_name)
            self._style_frame_button(button, view_name, checked)
            button.toggled.connect(
                lambda state, v=view_name: self._on_frame_button_toggled(
                    v, state))
            layout.addWidget(button)
            self._triplanar_frame_buttons[view_name] = button
        self._triplanar_frame_button_bar = bar
        bar.move(8, 8)
        bar.show()
        bar.raise_()

    def _on_frame_button_toggled(self, view_name, checked):
        """Show/hide one locator plane, persist the choice, restyle the button,
        and repaint the 3D view."""
        self._set_frame_visible(view_name, checked)
        model_node = self._triplanar_frame_nodes.get(view_name)
        if model_node is not None and slicer.mrmlScene.IsNodePresent(model_node):
            display_node = model_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetVisibility(checked)
        button = self._triplanar_frame_buttons.get(view_name)
        if button is not None:
            self._style_frame_button(button, view_name, checked)
        self._force_render_3d_views()

    def _destroy_triplanar_frame_buttons(self):
        """Remove the overlay toggle buttons and their container."""
        for button in getattr(self, "_triplanar_frame_buttons", {}).values():
            if button is not None:
                button.setParent(None)
                button.deleteLater()
        self._triplanar_frame_buttons = {}
        bar = getattr(self, "_triplanar_frame_button_bar", None)
        if bar is not None:
            bar.setParent(None)
            bar.deleteLater()
        self._triplanar_frame_button_bar = None

    def _setup_triplanar_views(self):
        """Apply the user's manual Red/Yellow/Green series assignment as the
        sticky per-plane display, then enable the high-resolution output grid and
        series fusion. Series selection is the user's job (Multi-plane display
        panel: Red=axial, Yellow=sagittal, Green=coronal by convention); this only
        locks in what they chose and turns on the prerequisites. Returns True when
        at least two views show distinct series."""
        _perf_log("[DEBUG triplanar.perf] setup_triplanar start")
        # Tri-planar requires co-registered series, so mark them aligned up front
        # and skip auto-registration (BRAINSFit on these series can crash Slicer).
        if hasattr(self, "ui"):
            blocked = self.ui.cbConfirmSeriesAligned.blockSignals(True)
            self.ui.cbConfirmSeriesAligned.setChecked(True)
            self.ui.cbConfirmSeriesAligned.blockSignals(blocked)
        # Lock in / re-apply whatever the user selected in the multi-plane panel.
        self.on_apply_plane_display_volumes_clicked()
        _perf_log("[DEBUG triplanar.perf] setup_triplanar: after on_apply")
        # A fine isotropic output grid sharpens the fused result; fusion must be
        # on so each routed per-view result is collected per series.
        self._enable_high_res_for_smoothing()
        _perf_log("[DEBUG triplanar.perf] setup_triplanar: after enable_high_res")
        if hasattr(self, "ui"):
            blocked = self.ui.cbEnableSeriesFusion.blockSignals(True)
            self.ui.cbEnableSeriesFusion.setChecked(True)
            self.ui.cbEnableSeriesFusion.blockSignals(blocked)
        self._fusion_results = {}
        self._fusion_grid_shape = None
        self._invalidate_fusion_coverage("tri-planar setup")
        # The output grid is rebuilt to cover the union of the assigned series;
        # force a rebuild and drop now-stale geometry-bound state (undo snapshots
        # would no longer match the new grid shape).
        self._output_geometry_triplanar_sig = None
        self._clear_selection_op_undo_stack(
            "Output grid rebuilt for tri-planar; undo history cleared.")
        bg = {v: self._view_background_volume(v) for v in PLANE_DISPLAY_SELECTOR_NAMES}
        view_series = {
            v: (n.GetName() if n is not None else None) for v, n in bg.items()
        }
        print("[DEBUG triplanar.setup] view backgrounds: {}".format(view_series))
        distinct_ids = {n.GetID() for n in bg.values() if n is not None}
        if len(distinct_ids) < 2:
            print("[DEBUG triplanar.setup] WARN: fewer than 2 distinct series in views")
            self._disable_triplanar_slice_frames()
            slicer.util.showStatusMessage(
                "Tri-planar mode: assign your series to the Red/Yellow/Green views "
                "in the Multi-plane display panel (and Apply) first.", 6000
            )
            return False
        # Show the 3D slice locator planes (3D counterpart of the 2D
        # intersection lines) now that the views carry distinct series. Use the
        # ensure variant so we do not rebuild if on_apply (above) already built
        # them via the same _ensure hook.
        self._ensure_triplanar_slice_frames("setup")
        # Show any existing segmentation in 3D now that we are tri-planar.
        if self._get_show_3d_triplanar():
            self._schedule_triplanar_3d_surface()
        slicer.util.showStatusMessage(
            "Tri-planar mode on: " + ", ".join(
                "{}={}".format(v, n) for v, n in view_series.items()), 5000
        )
        return True

    def _on_triplanar_mode_toggled(self, checked):
        """Enter or leave tri-planar multi-series mode."""
        print("[DEBUG triplanar.mode] toggled -> {}".format(bool(checked)))
        self._set_qsetting(SETTING_TRIPLANAR_ENABLED, bool(checked))
        self._triplanar_mode = bool(checked)
        self._active_inference_volume_override = None
        if checked:
            ok = self._setup_triplanar_views()
            print("[DEBUG triplanar.mode] view setup returned {}".format(ok))
        else:
            # Leaving tri-planar: the output grid reverts to the source-based one,
            # so the union-grid geometry-bound state is stale.
            self._output_geometry_triplanar_sig = None
            self._invalidate_fusion_coverage("left tri-planar mode")
            self._fusion_results = {}
            self._fusion_grid_shape = None
            self._clear_selection_op_undo_stack(
                "Output grid reverted to source; undo history cleared.")
            self._disable_triplanar_slice_frames()
        self._refresh_native_series_inference_ui()

    @staticmethod
    def _last_control_point_ras(caller):
        """World RAS of the markup's most recently placed control point, or None."""
        try:
            n = caller.GetNumberOfControlPoints()
            if n <= 0:
                return None
            ras = [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(n - 1, ras)
            return list(ras)
        except Exception:  # noqa: BLE001 - defensive around VTK accessors
            return None

    def _mask_centroid_ras(self, mask, volume_node):
        """RAS centroid of the nonzero voxels of `mask` on volume_node's grid
        (assumes volume_node has no parent transform, which holds for the Segment
        Editor source volume the scribble is painted on). Returns None if empty."""
        if volume_node is None:
            return None
        idx = np.argwhere(np.asarray(mask) > 0)
        if idx.size == 0:
            return None
        cz, cy, cx = idx.mean(axis=0)  # numpy (z, y, x) == IJK (k, j, i)
        m = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(m)
        ras = m.MultiplyPoint([float(cx), float(cy), float(cz), 1.0])
        return [ras[0], ras[1], ras[2]]

    def _view_for_ras_point(self, ras, tolerance_mm=2.0):
        """Standard view (Red/Yellow/Green) whose slice plane the RAS point lies
        on, or None. A markup placed in a 2D slice view lies on that view's
        plane, so this recovers which view an interaction happened in for
        tri-planar routing. Returns None when no plane is within tolerance (e.g.
        a point placed in the 3D view)."""
        if ras is None:
            return None
        best_view, best_dist, dists = None, None, {}
        for view_name, _logic, slice_node in self._iter_standard_slice_logics():
            m = slice_node.GetSliceToRAS()
            origin = (m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3))
            normal = (m.GetElement(0, 2), m.GetElement(1, 2), m.GetElement(2, 2))
            nlen = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
            if nlen == 0:
                continue
            dist = abs(
                sum((ras[i] - origin[i]) * normal[i] for i in range(3)) / nlen
            )
            dists[view_name] = round(dist, 2)
            if best_dist is None or dist < best_dist:
                best_view, best_dist = view_name, dist
        chosen = (
            best_view
            if best_view is not None
            and best_dist is not None
            and best_dist <= tolerance_mm
            else None
        )
        print("[DEBUG triplanar.view] ras={} plane_dists={} -> {} (tol={}mm)".format(
            [round(float(c), 1) for c in ras], dists, chosen, tolerance_mm))
        return chosen

    def _route_prompt_to_view(self, ras):
        """In tri-planar mode, set the per-prompt inference override to the series
        shown in the view containing `ras`. Resets any previous override first.
        Returns the routed view name or None. The caller MUST clear the override
        (self._active_inference_volume_override = None) when the prompt finishes."""
        self._active_inference_volume_override = None
        if not self._triplanar_mode or ras is None:
            return None
        view = self._view_for_ras_point(ras)
        if view is None:
            print("[DEBUG triplanar.route] no view plane matched; default volume")
            return None
        series = self._view_background_volume(view)
        if series is None:
            print("[DEBUG triplanar.route] view {} has no background series".format(
                view))
            return None
        self._active_inference_volume_override = series
        print("[DEBUG triplanar.route] routed to view {} (series '{}')".format(
            view, series.GetName()))
        # Auto-rotate the 3D camera to face the series the user is working in.
        if self._get_auto_camera_rotation():
            self._rotate_camera_to_view(view)
        return view

    def _resample_result_to_output(self, working_mask):
        """Bring an nnInteractive result from the inference grid onto the
        canonical output grid (smoothed if active), falling back to the source
        grid if the high-res resample fails. Returns an output-grid mask."""
        working_volume = self.get_inference_volume_node()
        try:
            _out_voxels = int(np.prod(self._output_grid_shape() or (0,)))
        except Exception:
            _out_voxels = -1
        _perf_log("[DEBUG triplanar.perf] resample result: working_shape={}, "
              "out_voxels={}".format(
                  tuple(np.asarray(working_mask).shape), _out_voxels), flush=True)
        if self._smoothing_active():
            out = self._interpolate_mask_to_output_grid(working_mask, working_volume)
            if out is not None:
                self._log_result_resample_loss(working_mask, out)
                return out
        out = self._resample_mask_between_volumes(
            working_mask, working_volume, self.get_output_volume_node()
        )
        if out is None:
            self._disable_high_res_output(
                "High-resolution output resample failed; reverted to source grid."
            )
            out = self._resample_mask_between_volumes(
                working_mask, working_volume, self.get_output_volume_node()
            )
        if out is None:
            out = np.asarray(working_mask)
        self._log_result_resample_loss(working_mask, out)
        return out

    def _log_result_resample_loss(self, working_mask, output_mask):
        """Diagnostic: voxels lost when the routed result is moved onto the output
        grid. A large `lost` on real data means the anatomy falls OUTSIDE the
        source-derived output grid, i.e. the output grid must be extended to the
        union of series bounding boxes (deferred follow-up); a small `lost` is the
        expected resampling rounding."""
        w = int(np.asarray(working_mask).astype(bool).sum())
        o = int(np.asarray(output_mask).astype(bool).sum())
        print("[DEBUG fusion.cover] result resample: working sum={}, output "
              "sum={}, lost={}".format(w, o, max(0, w - o)))

    def _handle_triplanar_result(self, segmentation_mask):
        """Tri-planar: accumulate the routed series result into the canonical
        segment with FOV authority -- the series governs ONLY inside its own
        field of view; everything outside is preserved. This replaces the old
        SDF-average autofuse, which let the other series' (stale, independently-
        sessioned) masks out-vote and erase a region the user just added, giving
        the 'flat plane cuts the bone and nothing can be added' symptom.

        Supports both add and subtract: nnInteractive already returns the full
        cumulative segmentation for that series' session (positive + negative
        prompts baked in), so REPLACING inside the FOV applies whatever the user
        did. Overlap zones resolve to last-editor-wins, which is predictable and
        far better than averaging that silently erodes. Direction-weighted SDF
        smoothing is now opt-in via the manual Fuse button."""
        # Safety net for the lazy-restore gap: any tri-planar interaction means
        # we are active with series assigned, so make sure the locator planes
        # exist (no-op once built).
        self._ensure_triplanar_slice_frames("result")
        output_mask = self._resample_result_to_output(segmentation_mask)
        out_shape = self._output_grid_shape()
        new = np.asarray(output_mask).astype(bool)
        routed_vid = self.get_inference_volume_node().GetID()
        cov = self._series_coverage_mask(routed_vid)
        baseline = self.previous_states.get("segment_data")
        if baseline is None or tuple(np.asarray(baseline).shape) != tuple(out_shape):
            baseline = self.get_segment_data(self.get_output_volume_node())
        if baseline is not None and tuple(np.asarray(baseline).shape) != tuple(out_shape):
            baseline = None
        base = None if baseline is None else np.asarray(baseline).astype(bool)
        print("[DEBUG triplanar.result] routed result sum={}, shape={}, "
              "output_grid={}, match={}, cov={}".format(
                  int(new.sum()), tuple(new.shape),
                  tuple(out_shape) if out_shape is not None else None,
                  (out_shape is not None and tuple(new.shape) == tuple(out_shape)),
                  "yes" if cov is not None else "none"))
        if cov is not None and base is not None:
            merged = (new & cov) | (base & ~cov)
            print("[DEBUG triplanar.merge] FOV-authoritative id={} new_in_fov={} "
                  "kept_outside={} merged={}".format(
                      routed_vid, int((new & cov).sum()),
                      int((base & ~cov).sum()), int(merged.sum())))
        elif base is not None:
            # No coverage info (resample failed): union preserves adds (but then
            # cannot subtract) -- the safe degraded default since coverage
            # normally builds fine.
            merged = new | base
            print("[DEBUG triplanar.merge] no coverage -> union fallback "
                  "merged={}".format(int(merged.sum())))
        else:
            merged = new
            print("[DEBUG triplanar.merge] no baseline -> show new merged={}"
                  .format(int(merged.sum())))
        # Keep per-series raw results so the OPTIONAL manual Fuse (SDF smoothing)
        # still works; this no longer drives the live display.
        self._maybe_collect_fusion_result(output_mask)
        self.show_segmentation(merged.astype(np.uint8))

    def _handle_server_segmentation_result(self, segmentation_mask):
        """Write normal results directly, but stage supplemental results as previews."""
        _perf_log("[DEBUG triplanar.perf] handle_server_result enter shape={}".format(
            tuple(np.asarray(segmentation_mask).shape)), flush=True)
        if getattr(self, "_fusion_capture_active", False):
            # One-click three-series fusion: stash this series' result on the
            # output grid (keyed by the active inference volume) instead of
            # displaying it. For point-seed series the handler fires once per
            # seed; the last (cumulative) result wins, which is what we want.
            out_mask = self._resample_result_to_output(segmentation_mask)
            inf_node = self.get_inference_volume_node()
            if inf_node is not None:
                self._fusion_capture_store[inf_node.GetID()] = (
                    np.asarray(out_mask).astype(np.uint8)
                )
                print("[DEBUG fuse3] captured result for '{}' sum={}".format(
                    inf_node.GetName(), int(np.asarray(out_mask).sum())))
            return
        if self._triplanar_mode:
            # Per-view routed result: collect it for fusion and show the combined
            # direction-weighted result instead of a single-series preview.
            self._handle_triplanar_result(segmentation_mask)
            return
        if not self._is_native_series_inference_active():
            # The result is on the inference grid (== source here). Resample onto
            # the canonical output grid before writing (no-op if they are equal).
            inference_mask = segmentation_mask
            if self._smoothing_active():
                smoothed = self._interpolate_mask_to_output_grid(
                    inference_mask, self.get_inference_volume_node()
                )
                if smoothed is not None:
                    self._maybe_collect_fusion_result(smoothed)
                    self.show_segmentation(smoothed)
                    return
                # Smoothing failed; fall through to plain resampling below.
            resampled = self._resample_mask_between_volumes(
                inference_mask,
                self.get_inference_volume_node(),
                self.get_output_volume_node(),
            )
            if resampled is None:
                # Resample to the high-res grid failed; keep the result by
                # reverting to the source grid rather than dropping it.
                self._disable_high_res_output(
                    "High-resolution output resample failed; "
                    "reverted to the source grid."
                )
                resampled = inference_mask
            self._maybe_collect_fusion_result(resampled)
            self.show_segmentation(resampled)
            return

        segmentation_mask = self._apply_lasso_slice_clip(segmentation_mask)
        self._last_lasso_slice = None
        self._update_inference_preview(segmentation_mask)

    def on_sync_inference_result_clicked(self, checked=False):
        """Merge the current supplemental-series preview into the source segment."""
        if self._inference_result_source_mask is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Native-series inference",
                "Run at least one prompt on the supplemental series first.",
            )
            return
        if self.get_current_segment_id() != self._inference_preview_target_segment_id:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Native-series inference",
                "The selected source segment changed after the preview was "
                "created. Clear the preview and run the prompt again.",
            )
            return
        source_volume = self.get_volume_node()
        if (
            source_volume is None
            or source_volume.GetID() != self._inference_preview_source_volume_id
        ):
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Native-series inference",
                "The source volume changed after the preview was created. "
                "Clear the preview and run the prompt again.",
            )
            return

        merged_mask = self.compute_inference_sync_mask(
            self.get_segment_data(),
            self._inference_result_source_mask,
            self.ui.cbInferenceSyncMode.currentIndex,
        )
        self.show_segmentation(merged_mask)
        self._destroy_inference_preview()
        result = self.upload_segment_to_server()
        if result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Native-series inference",
                "The preview was merged locally, but syncing the merged mask "
                "to the server failed.",
            )
            return
        slicer.util.showStatusMessage(
            "Supplemental-series preview merged into the source segment.", 4000
        )

    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """
        self.ui.uploadProgressGroup.setVisible(False)

        # Load the saved server URL (default to an empty string if not set)
        savedServer = self._get_qsetting(SETTING_SERVER, "")
        self.ui.Server.text = savedServer
        self.server = savedServer.rstrip("/")

        self.ui.Server.editingFinished.connect(self.update_server)
        self.ui.pbTestServer.clicked.connect(self.test_server_connection)

        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)
        self.ui.pbApplyPlaneDisplayVolumes.clicked.connect(
            self.on_apply_plane_display_volumes_clicked
        )
        self.ui.pbResetPlaneDisplayVolumes.clicked.connect(
            self.on_reset_plane_display_volumes_clicked
        )
        self.ui.cbEnableNativeSeriesInference.toggled.connect(
            self._on_native_series_inference_settings_changed
        )
        self.ui.cbInferenceWorkingVolume.currentNodeChanged.connect(
            self._on_native_series_inference_settings_changed
        )
        self.ui.pbSyncInferenceResult.clicked.connect(
            self.on_sync_inference_result_clicked
        )
        self.ui.pbClearInferencePreview.clicked.connect(
            self.on_clear_inference_preview_clicked
        )
        self.ui.cbEnableSeriesFusion.toggled.connect(
            self._on_series_fusion_toggled
        )
        self.ui.pbFuseSeries.clicked.connect(self.on_fuse_series_clicked)
        self.ui.pbFuseThreeSeriesRoi.clicked.connect(
            self.onFuseFromThreeSeriesWithROI
        )
        # Tri-planar multi-series mode. Restore the persisted preference into the
        # checkbox with signals blocked (so we do not rearrange views at startup,
        # before series are loaded); the view setup runs when the user toggles.
        self.ui.cbTriPlanarMode.toggled.connect(self._on_triplanar_mode_toggled)
        blocked = self.ui.cbTriPlanarMode.blockSignals(True)
        self.ui.cbTriPlanarMode.setChecked(self._get_triplanar_enabled())
        self.ui.cbTriPlanarMode.blockSignals(blocked)
        self._triplanar_mode = self._get_triplanar_enabled()
        # Lazy start: the checkbox is restored CHECKED without firing the toggle,
        # so the planes are built later by _ensure_triplanar_slice_frames when
        # the user applies their series (see _apply_plane_display_volumes).
        # Locator plane opacity slider (controls all three planes together).
        self.ui.sldPlaneOpacity.valueChanged.connect(
            self._on_plane_opacity_changed)
        plane_pct = int(round(self._get_plane_opacity() * 100))
        blocked = self.ui.sldPlaneOpacity.blockSignals(True)
        self.ui.sldPlaneOpacity.setValue(plane_pct)
        self.ui.sldPlaneOpacity.blockSignals(blocked)
        self.ui.lblPlaneOpacityValue.setText("%d %%" % plane_pct)
        # Auto-rotate the 3D camera to face the active series.
        self.ui.cbAutoRotateCamera.toggled.connect(
            self._on_auto_camera_rotation_toggled)
        blocked = self.ui.cbAutoRotateCamera.blockSignals(True)
        self.ui.cbAutoRotateCamera.setChecked(self._get_auto_camera_rotation())
        self.ui.cbAutoRotateCamera.blockSignals(blocked)
        # Align the auto-rotate / "face series" camera to each series' oblique
        # acquisition frame (its direction matrix + registration transform).
        self.ui.cbObliqueCameraAlign.toggled.connect(
            self._on_oblique_camera_align_toggled)
        blocked = self.ui.cbObliqueCameraAlign.blockSignals(True)
        self.ui.cbObliqueCameraAlign.setChecked(self._get_oblique_camera_align())
        self.ui.cbObliqueCameraAlign.blockSignals(blocked)
        # Show the segmentation 3D surface while in tri-planar (debounced).
        self.ui.cbShow3DTriPlanar.toggled.connect(
            self._on_show_3d_triplanar_toggled)
        blocked = self.ui.cbShow3DTriPlanar.blockSignals(True)
        self.ui.cbShow3DTriPlanar.setChecked(self._get_show_3d_triplanar())
        self.ui.cbShow3DTriPlanar.blockSignals(blocked)
        # Manual "face series in 3D" buttons (always active, ignore the toggle).
        self.ui.pbFaceRed.clicked.connect(
            lambda *a, v="Red": self._rotate_camera_to_view(v))
        self.ui.pbFaceYellow.clicked.connect(
            lambda *a, v="Yellow": self._rotate_camera_to_view(v))
        self.ui.pbFaceGreen.clicked.connect(
            lambda *a, v="Green": self._rotate_camera_to_view(v))
        self.ui.pbRegisterSupplemental.clicked.connect(
            self.on_register_supplemental_clicked
        )
        self.ui.pbClearAlignment.clicked.connect(
            self.on_clear_alignment_clicked
        )
        # Both alignment preferences change whether/how registration happens, so
        # toggling either re-evaluates alignment (and discards a stale preview).
        self.ui.cbAutoRegisterSupplemental.toggled.connect(
            self._on_native_series_inference_settings_changed
        )
        self.ui.cbConfirmSeriesAligned.toggled.connect(
            self._on_native_series_inference_settings_changed
        )
        self._refresh_native_series_inference_ui()

        # High-resolution output geometry (load persisted prefs with signals
        # blocked so setting the widgets does not trigger a migration at startup).
        self.ui.cbEnableHighResOutput.toggled.connect(self._on_high_res_output_changed)
        self.ui.sbOutputSpacing.valueChanged.connect(self._on_output_spacing_changed)
        blocked = self.ui.sbOutputSpacing.blockSignals(True)
        self.ui.sbOutputSpacing.setValue(self._get_output_spacing_setting())
        self.ui.sbOutputSpacing.blockSignals(blocked)
        blocked = self.ui.cbEnableHighResOutput.blockSignals(True)
        self.ui.cbEnableHighResOutput.setChecked(self._get_high_res_enabled_setting())
        self.ui.cbEnableHighResOutput.blockSignals(blocked)

        # Snap slice scrolling to the original voxel grid (no interpolated frames).
        # Observers are already installed from the constructor per the persisted
        # setting, so set the widget with signals blocked to avoid a redundant
        # reinstall here.
        self.ui.cbSnapSlicesToGrid.toggled.connect(self._on_snap_slices_changed)
        blocked = self.ui.cbSnapSlicesToGrid.blockSignals(True)
        self.ui.cbSnapSlicesToGrid.setChecked(self._get_snap_slices_setting())
        self.ui.cbSnapSlicesToGrid.blockSignals(blocked)

        # Smooth (interpolated) results. Only honor the persisted preference when
        # the high-resolution output it depends on is actually enabled; otherwise
        # start unchecked rather than silently building a fine grid at startup.
        self.ui.cbSmoothInterpolate.toggled.connect(
            self._on_smooth_interpolate_changed
        )
        self.ui.pbSmoothCurrentSegment.clicked.connect(
            self.on_smooth_current_segment_clicked
        )
        blocked = self.ui.cbSmoothInterpolate.blockSignals(True)
        self.ui.cbSmoothInterpolate.setChecked(
            self._get_smooth_interpolate_setting()
            and self._get_high_res_enabled_setting()
        )
        self.ui.cbSmoothInterpolate.blockSignals(blocked)

        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(
            self.on_prompt_type_positive_clicked
        )
        self.ui.pbPromptTypeNegative.clicked.connect(
            self.on_prompt_type_negative_clicked
        )

        self.ui.pbInteractionLassoCancel.setVisible(False)
        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)

        self.ui.pbInteractionLassoCancel.clicked.connect(self.on_lasso_cancel_clicked)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)

        # Selection operations (boolean editing) and manual server sync
        self.ui.pbSyncToServer.clicked.connect(self.on_sync_to_server_clicked)
        self.ui.pbApplySelectionOp.clicked.connect(self.on_apply_selection_op_clicked)
        self.ui.cbOperandSource.currentIndexChanged.connect(self._on_operand_source_changed)
        self.ui.cbRoiShape.currentIndexChanged.connect(self._on_roi_shape_changed)
        self.ui.pbPlaceRoi.clicked.connect(self.on_place_roi_clicked)
        self.ui.pbClearRoi.clicked.connect(self.on_clear_roi_clicked)
        self.ui.pbPlaceCropRoi.clicked.connect(self.on_place_crop_roi_clicked)
        self.ui.pbCropSegmentByBox.clicked.connect(
            self.on_crop_segment_by_box_clicked)
        self.ui.pbPlaceWandSeed.clicked.connect(self.on_place_wand_seed_clicked)
        self.ui.pbClearWandSeed.clicked.connect(self.on_clear_wand_seed_clicked)
        self.ui.pbPreviewWand.clicked.connect(self.on_preview_wand_clicked)
        self.ui.pbClearPreviewWand.clicked.connect(self.on_clear_preview_wand_clicked)
        self.ui.pbDrawLasso3d.clicked.connect(self.on_draw_lasso3d_clicked)
        self.ui.pbClearLasso3d.clicked.connect(self.on_clear_lasso3d_clicked)
        self.ui.pbPreviewLasso3d.clicked.connect(self.on_preview_lasso3d_clicked)
        self.ui.pbClearPreviewLasso3d.clicked.connect(
            self.on_clear_preview_lasso3d_clicked
        )
        self.ui.pbUndoSelectionOp.clicked.connect(self.on_undo_selection_op_clicked)
        self.ui.sldSegmentOpacity.valueChanged.connect(self._on_segment_opacity_changed)
        # Non-destructive display smoothing (load persisted prefs with signals
        # blocked so restoring the widgets does not apply smoothing at startup).
        self.ui.cbDisplaySmooth.toggled.connect(
            self._on_display_smooth_enabled_changed
        )
        self.ui.sbDisplaySmoothStrength.valueChanged.connect(
            self._on_display_smooth_strength_changed
        )
        self.ui.pbBakeDisplaySmooth.clicked.connect(
            self.on_bake_display_smooth_clicked
        )
        blocked = self.ui.sbDisplaySmoothStrength.blockSignals(True)
        self.ui.sbDisplaySmoothStrength.setValue(self._get_display_smooth_strength())
        self.ui.sbDisplaySmoothStrength.blockSignals(blocked)
        blocked = self.ui.cbDisplaySmooth.blockSignals(True)
        self.ui.cbDisplaySmooth.setChecked(self._get_display_smooth_enabled())
        self.ui.cbDisplaySmooth.blockSignals(blocked)
        self._refresh_display_smooth_ui()
        self.ui.cbEnableLassoClip.toggled.connect(self._on_lasso_clip_enabled_changed)
        self.ui.sbLassoClipN.valueChanged.connect(self._on_lasso_clip_n_changed)
        # Load persisted lasso-clip prefs into the widgets (block signals so
        # setting the value does not immediately re-save it).
        blocked = self.ui.cbEnableLassoClip.blockSignals(True)
        self.ui.cbEnableLassoClip.setChecked(self._get_lasso_clip_enabled())
        self.ui.cbEnableLassoClip.blockSignals(blocked)
        blocked = self.ui.sbLassoClipN.blockSignals(True)
        self.ui.sbLassoClipN.setValue(self._get_lasso_clip_n())
        self.ui.sbLassoClipN.blockSignals(blocked)
        # Multi-view lasso controls
        self.ui.cbLassoMultiView.toggled.connect(self._on_lasso_multiview_toggled)
        self.ui.pbLassoMultiViewSubmit.clicked.connect(
            self._on_lasso_multiview_submit_clicked
        )
        self.ui.pbLassoMultiViewClear.clicked.connect(
            self._on_lasso_multiview_clear_clicked
        )
        blocked = self.ui.cbLassoMultiView.blockSignals(True)
        self.ui.cbLassoMultiView.setChecked(self._get_lasso_multiview_enabled())
        self.ui.cbLassoMultiView.blockSignals(blocked)
        self._update_multiview_lasso_ui()
        self.populate_operand_selector()
        self._install_selection_op_observers()
        # Initialize operand-row visibility and Apply enable state for the
        # default source.
        self._on_operand_source_changed(self.ui.cbOperandSource.currentIndex)
        self._sync_opacity_slider_from_segment()

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "Shift+L": self.submit_lasso_if_present,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
        }
        self.shortcut_items = {}

        for shortcut_key, shortcut_event in shortcuts.items():
            debug_print(f"Added shortcut for {shortcut_key}: {shortcut_event}")
            shortcut = qt.QShortcut(
                qt.QKeySequence(shortcut_key), slicer.util.mainWindow()
            )
            shortcut.activated.connect(shortcut_event)
            self.shortcut_items[shortcut_key] = shortcut

    def remove_shortcut_items(self):
        """
        Called at cleanup to remove all the shortcuts we attached.
        """
        if hasattr(self, "shortcut_items"):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()
                shortcut = None

    def install_dependencies(self):
        """
        Checks for (and installs if needed) python packages needed by the module.
        """
        dependencies = {
            "requests_toolbelt": "requests_toolbelt",
            "skimage": "scikit-image",
            "matplotlib": "matplotlib",
        }

        for dependency in dependencies:
            if self.check_dependency_installed(dependency, dependencies[dependency]):
                continue
            self.run_with_progress_bar(
                self.pip_install_wrapper,
                (dependencies[dependency],),
                "Installing dependencies: %s" % dependency,
            )

    def check_dependency_installed(self, import_name, module_name_and_version):
        """
        Checks if a package is installed with the correct version.
        """
        if "==" in module_name_and_version:
            module_name, module_version = module_name_and_version.split("==")
        else:
            module_name = module_name_and_version
            module_version = None

        spec = importlib.util.find_spec(import_name)
        if spec is None:
            # Not installed
            return False

        if module_version is not None:
            import importlib.metadata as metadata
            try:
                version = metadata.version(module_name)
                if version != module_version:
                    # Version mismatch
                    return False
            except metadata.PackageNotFoundError:
                debug_print(f"Could not determine version for {module_name}.")

        return True

    def pip_install_wrapper(self, command, event):
        """
        Installs pip packages.
        """
        slicer.util.pip_install(command)
        event.set()

    def run_with_progress_bar(self, target, args, title):
        """
        Runs a function in a background thread, while showing a progress bar in the UI
        as a pop up window.
        """
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)

        parallel_event = threading.Event()
        dep_thread = threading.Thread(
            target=target,
            args=(
                *args,
                parallel_event,
            ),
        )
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()

    def _teardown_scribble_observer(self):
        """Detach the manually-added scribble Paint observer if present.

        It is added via AddObserver (not VTKObservationMixin), so it is not
        covered by removeObservers(); detach it here so a scribble left
        mid-stroke does not leak the observer on module close.
        """
        if not hasattr(self, "_scribble_labelmap_callback_tag"):
            return
        tag = self._scribble_labelmap_callback_tag.get("tag", None)
        node = getattr(self, "scribble_segment_node", None)
        if tag and node is not None and slicer.mrmlScene.IsNodePresent(node):
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        del self._scribble_labelmap_callback_tag

    def cleanup(self):
        """
        Clean up resources when the module is closed.
        """
        self.removeObservers()
        self._remove_slice_snap_observers()
        self._teardown_scribble_observer()
        # Tri-planar 3D locator frames (model nodes + slice-node observers).
        self._disable_triplanar_slice_frames()

        # Tear down the hidden lasso (3D) Scissors editor and region/preview nodes.
        self._destroy_lasso3d()
        # Selection Operations scaffolding: ROI (+ preview model/transform),
        # wand seeds (+ their placement observer) and the hidden wand preview
        # segmentation would otherwise survive a module close/Reload.
        self._destroy_selection_roi()
        self._destroy_crop_roi()
        self._destroy_wand_seed()
        self._destroy_wand_preview_segmentation()
        self._destroy_inference_preview()
        self._remove_output_geometry_node()
        self._cancel_active_registration()
        self._remove_alignment_transforms()

        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()

    def __del__(self):
        """
        Called when the widget is destroyed.
        """
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()

        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(
                prompt_type["name"]
            ):
                debug_print("Skipping", prompt_name)
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()

            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)

            prompt_type["button"].setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )

            self.prev_caller = None

            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    prompt_type["on_placed_function"],
                )

            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name)) 
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/{prompt_type['button_icon_filename']}"))
            prompt_type["button"].setIcon(icon)

        if (
            not skip_if_exists
            or slicer.mrmlScene.GetFirstNodeByName(self.scribble_segment_node_name)
            is None
        ):
            self.setup_scribble_prompt()

            self.ui.pbInteractionScribble.setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )
            self.all_prompt_buttons["scribble"] = self.ui.pbInteractionScribble

        # To make sure that when segment is reset, no interaction is selected (without this code
        # the last interaction tool gets selected)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        # Rebuilding prompts re-runs setup_scribble_prompt, whose
        # setSegmentationNode call pushes the source volume back into every slice
        # view background. Restore the sticky per-plane display selections after
        # the whole rebuild (and its callers, e.g. the lasso prompt's _next)
        # finish. No-op unless the multi-plane display override is active.
        self._schedule_plane_display_reapply()

    def setup_scribble_prompt(self):
        """
        Creates a hidden "Segment Editor" for the scribble prompt.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt

        # Create a background (headless) segment editor
        self.scribble_editor_widget = (
            qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        )
        self.scribble_editor_widget.setMRMLScene(slicer.mrmlScene)
        self.scribble_editor_widget.setMaximumNumberOfUndoStates(10)

        # Create a separate SegmentEditorNode
        self.scribble_editor_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode"
        )
        self.scribble_editor_widget.setMRMLSegmentEditorNode(self.scribble_editor_node)

        self.scribble_segment_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        self.scribble_segment_node.SetReferenceImageGeometryParameterFromVolumeNode(
            self.get_volume_node()
        )
        self.scribble_segment_node.SetName(self.scribble_segment_node_name)

        # Make sure the node exists and is set
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)

        self.scribble_segment_node.CreateDefaultDisplayNodes()
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "bg", "bg", [0.0, 0.0, 1.0]
        )
        self.scribble_segment_node.GetSegmentation().AddEmptySegment(
            "fg", "fg", [0.0, 0.0, 1.0]
        )
        dn = self.scribble_segment_node.GetDisplayNode()

        opacity = 0.2
        dn.SetSegmentOpacity2DFill("bg", opacity)
        dn.SetSegmentOpacity2DOutline("bg", opacity)
        dn.SetSegmentOpacity2DFill("fg", opacity)
        dn.SetSegmentOpacity2DOutline("fg", opacity)

        self._prev_scribble_mask = None
            
        light_dark_mode = self.is_ui_dark_or_light_mode()
        icon = qt.QIcon(self.resourcePath(f"Icons/prompts/{light_dark_mode}/scribble_icon.svg"))
        self.ui.pbInteractionScribble.setIcon(icon)

    def is_ui_dark_or_light_mode(self):
        # Returns whether the current appearance of the UI is dark mode (will return "dark")
        # or light mode (will return "light")
        current_style = slicer.app.settings().value("Styles/Style")

        if current_style == "Dark Slicer":
            return "dark"
        elif current_style == "Light Slicer":
            return "light"
        elif current_style == "Slicer":
            app_palette = QApplication.instance().palette()
            window_color = app_palette.color(QPalette.Active, QPalette.Window)
            lightness = window_color.lightness()
            dark_mode_threshold = 128

            if lightness < dark_mode_threshold:
                return "dark"
            else:
                return "light"
        return "light"

    def remove_prompt_nodes(self):
        """
        Removes all the Markups/Fiducials prompts.
        """

        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)

        for prompt_type in list(self.prompt_types.values()):
            _remove(prompt_type["name"])

        self.ui.pbInteractionLassoCancel.setVisible(False)

        _remove(self.scribble_segment_node_name)

    def on_interaction_node_modified(self, caller, event):
        """
        Deselect prompt button if interaction mode is not place point anymore
        """

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        for prompt_type in self.prompt_types.values():
            if interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place:
                if prompt_type["name"] == "LassoPrompt" and (self.ui.pbInteractionLasso.isChecked()):
                    self.submit_lasso_if_present()
                prompt_type["button"].setChecked(False)
            elif interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
                placingThisNode = (selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID())
                prompt_type["button"].setChecked(placingThisNode)

        # Stop scribble if placing markup
        if interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
            self.ui.pbInteractionScribble.setChecked(False)
            # Also disarm the lasso (3D) Scissors tool so it does not eat clicks.
            if self.ui.pbDrawLasso3d.isChecked():
                self.ui.pbDrawLasso3d.setChecked(False)
                self._deactivate_lasso3d_scissors()

    def remove_all_but_last_prompt(self):
        """
        Removes all but the most recently placed markup points
        (helpful when segment change was detected).
        """
        last_modified_node = None
        all_nodes = []

        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)

                    all_nodes.append(node)
                    if (
                        last_modified_node is None
                        or node.GetMTime() > last_modified_node.GetMTime()
                    ):
                        last_modified_node = node

        for node in all_nodes:
            n = node.GetNumberOfControlPoints()

            if node == last_modified_node:
                if node.GetName() == "LassoPrompt":
                    continue
                n -= 1

            for i in range(n):
                node.RemoveNthControlPoint(0)

        self._reset_multiview_lasso_accumulation()

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if checked:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(self.prompt_types[prompt_name]["node_class"])
            selectionNode.SetActivePlaceNodeID(self.prompt_types[prompt_name]["node"].GetID())
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            if prompt_name == "lasso":
                self.submit_lasso_if_present()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def display_node_markup_point(self, display_node):
        """
        Handles the appearance of the point display node.
        """
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(0.0, 0.0, 1.0)  # Green color
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views

    def display_node_markup_bbox(self, display_node):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    def display_node_markup_lasso(self, display_node):
        """
        Handles the appearance of the lasso display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetGlyphScale(1)
        display_node.SetLineThickness(0.3)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)
        display_node.SetVisibility3D(False)

    ###############################################################################
    # Event handlers for prompts
    ###############################################################################

    #
    #  -- Point
    #
    def on_point_placed(self, caller, event):
        """
        Called when a point is placed in the scene. Grabs the point position
        and sends it to the server.
        """
        if self._alignment_in_progress:
            # A registration is running and the prompt would be blocked anyway;
            # drop the just-placed marker so it does not linger unsent.
            n = caller.GetNumberOfControlPoints()
            if n > 0:
                caller.RemoveNthControlPoint(n - 1)
            slicer.util.showStatusMessage(
                "Series registration in progress; point was not sent.", 4000
            )
            return

        # Tri-planar routing: run this prompt against the series shown in the
        # view the point was placed in (no-op outside tri-planar mode).
        self._route_prompt_to_view(self._last_control_point_ras(caller))
        try:
            xyz = self.xyz_from_caller(
                caller, volume_node=self.get_inference_volume_node()
            )

            volume_node = self.get_volume_node()
            if volume_node:
                self.point_prompt(xyz=xyz, positive_click=self.is_positive)
        finally:
            self._active_inference_volume_override = None

    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        """
        Uploads point prompt to the server.
        """
        # A stale lasso slice marker (e.g. from a lasso whose submit failed
        # mid-way) must never clip a point result in show_segmentation.
        self._last_lasso_slice = None
        url = f"{self.server}/add_point_interaction"

        seg_response = self.request_to_server(
            url, json={"voxel_coord": xyz[::-1], "positive_click": positive_click}
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        debug_print("unpacked_segmentation.sum():", unpacked_segmentation.sum())
        debug_print(seg_response)
        debug_print(f"{positive_click} point prompt triggered! {xyz}")

        self._handle_server_segmentation_result(unpacked_segmentation)

    #
    #  -- Bounding Box
    #
    def on_bbox_placed(self, caller, event):
        """
        Every time a control point is placed/moved for the bounding box ROI node.
        Once two corners are placed, we send the bounding box to the server.
        """
        if self._alignment_in_progress:
            # A registration is running; discard the partial box and reset the
            # two-corner state so the next placement starts cleanly.
            caller.RemoveAllControlPoints()
            self.prev_caller = None
            slicer.util.showStatusMessage(
                "Series registration in progress; box was not sent.", 4000
            )
            return

        # Tri-planar routing: run this box against the series shown in the view
        # the corner was placed in (no-op outside tri-planar mode). Routed before
        # voxel coordinates are computed so both corners land in that series' grid.
        self._route_prompt_to_view(self._last_control_point_ras(caller))
        try:
            xyz = self.xyz_from_caller(
                caller, volume_node=self.get_inference_volume_node()
            )

            if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
                roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
                current_size = list(roi_node.GetSize())
                drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
                current_size[drawn_in_axis] = 0
                roi_node.SetSize(current_size)

                volume_node = self.get_volume_node()
                if volume_node:
                    outer_point_two = self.prev_bbox_xyz

                    outer_point_one = [
                        xyz[0] * 2 - outer_point_two[0],
                        xyz[1] * 2 - outer_point_two[1],
                        xyz[2] * 2 - outer_point_two[2],
                    ]

                    self.bbox_prompt(
                        outer_point_one=outer_point_one,
                        outer_point_two=outer_point_two,
                        positive_click=self.is_positive,
                    )

                    def _next():
                        self.setup_prompts()
                        # Start placing a new box
                        self.ui.pbInteractionBBox.click()

                    qt.QTimer.singleShot(0, _next)

                self.prev_caller = None
            else:
                self.prev_bbox_xyz = xyz

            self.prev_caller = caller
        finally:
            self._active_inference_volume_override = None

    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """
        Uploads BBox prompt to the server.
        """
        # A stale lasso slice marker must never clip a bbox result.
        self._last_lasso_slice = None
        url = f"{self.server}/add_bbox_interaction"

        seg_response = self.request_to_server(
            url,
            json={
                "outer_point_one": outer_point_one[::-1],
                "outer_point_two": outer_point_two[::-1],
                "positive_click": positive_click,
            },
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        self._handle_server_segmentation_result(unpacked_segmentation)

    #
    #  -- Lasso
    #
    def on_lasso_placed(self, caller, event):
        """
        Called whenever a new point is added to the lasso.
        """
        pointsDefined = self.prompt_types["lasso"]["node"].GetNumberOfControlPoints() > 0
        self.ui.pbInteractionLassoCancel.setVisible(pointsDefined)

    def on_lasso_cancel_clicked(self):
        """
        Called when the user clicks the cancel button for the lasso.
        """
        self.prompt_types["lasso"]["node"].RemoveAllControlPoints()
        self.ui.pbInteractionLassoCancel.setVisible(False)

    def submit_lasso_if_present(self):
        """
        Submits the currently open lasso. We gather all the control points,
        rasterize them into a mask, and send the mask to the server.
        """
        print("[DEBUG submit_lasso] called, multiview={}".format(
            self._get_lasso_multiview_enabled()))
        caller = self.prompt_types["lasso"]["node"]

        # Grab raw RAS curve points for world-space planarity check (robust for
        # oblique volumes where the RAS->IJK transform scatters voxel indices).
        from vtk.util.numpy_support import vtk_to_numpy as _vtk_to_numpy
        _vtk_pts = caller.GetCurvePointsWorld()
        _ras_pts = _vtk_to_numpy(_vtk_pts.GetData()) if _vtk_pts is not None else np.zeros((0, 3))
        print("[DEBUG submit_lasso] _ras_pts count:", len(_ras_pts))

        # Cache the world-space curve points so the one-click three-series fusion
        # button can reuse this lasso after the live node is consumed/cleared.
        if len(_ras_pts) >= 3:
            self._last_lasso_world_points = np.array(_ras_pts)

        # Tri-planar routing: a lasso is drawn on one view's plane, so route it to
        # that view's series BEFORE computing voxel coords / rasterizing. In
        # tri-planar mode multi-view accumulation is bypassed (each view shows a
        # different series), so the lasso submits immediately and routed.
        if self._triplanar_mode and len(_ras_pts):
            self._route_prompt_to_view(list(_ras_pts[0]))
        multiview_effective = (
            self._get_lasso_multiview_enabled() and not self._triplanar_mode
        )
        inference_volume = self.get_inference_volume_node()

        xyzs = self.xyz_from_caller(
            caller,
            point_type="curve_point",
            volume_node=inference_volume,
        )
        print("[DEBUG submit_lasso] xyzs count:", len(xyzs))

        if len(xyzs) < 3:
            slicer.util.showStatusMessage("Lasso needs at least 3 points.", 4000)
            print("[DEBUG submit_lasso] RETURN: fewer than 3 points")
            self._active_inference_volume_override = None
            return

        # The lasso prompt only supports points on a single slice plane.
        # If on_interaction_node_modified auto-submits a lasso whose control
        # points span multiple slices, lasso_points_to_mask raises -- swallow
        # the error, clear the lasso, and tell the user.
        try:
            mask = self.lasso_points_to_mask(xyzs, ras_points=_ras_pts,
                                             volume_node=inference_volume)
            print("[DEBUG submit_lasso] mask ok, sum={}".format(mask.sum()))
        except ValueError as e:
            print("[DEBUG submit_lasso] RETURN: ValueError:", e)
            slicer.util.showStatusMessage(
                "Lasso must be drawn on a slice aligned with the "
                "segmentation/inference volume; cleared.",
                4000,
            )
            caller.RemoveAllControlPoints()
            self.ui.pbInteractionLassoCancel.setVisible(False)
            self._active_inference_volume_override = None
            return

        volume_node = self.get_volume_node()
        if not volume_node:
            slicer.util.showStatusMessage("No source volume selected.", 4000)
            print("[DEBUG submit_lasso] RETURN: no volume node")
            self._active_inference_volume_override = None
            return

        if multiview_effective:
            # Multi-view mode: keep this plane's mask, don't submit yet. Each
            # plane is sent as its own lasso interaction on Submit.
            self._multiview_lasso_masks.append(mask.copy())
            self._multiview_lasso_count = len(self._multiview_lasso_masks)
            # Disable slice-range clip; multi-view spans multiple planes.
            self._last_lasso_slice = None
            self._update_multiview_lasso_ui()
            slicer.util.showStatusMessage(
                "Multi-view lasso: {} view(s) accumulated. "
                "Click Submit to run.".format(self._multiview_lasso_count),
                3000,
            )

            def _next_mv():
                # Rename the just-accumulated node so remove_prompt_nodes
                # won't destroy it, then style it as an "accumulated" overlay.
                current_node = self.prompt_types["lasso"]["node"]
                if current_node is not None:
                    idx = len(self._multiview_lasso_nodes)
                    current_node.SetName("LassoPromptMV_{}".format(idx))
                    dn = current_node.GetDisplayNode()
                    if dn:
                        dn.SetSliceProjection(True)
                        dn.SetHandlesInteractive(False)
                        dn.SetOutlineOpacity(0.35)
                        dn.SetColor(0.4, 0.6, 1.0)
                        dn.SetVisibility3D(False)
                    self._multiview_lasso_nodes.append(current_node)
                self.setup_prompts()
                self.ui.pbInteractionLasso.click()

            qt.QTimer.singleShot(0, _next_mv)
        else:
            self.lasso_or_scribble_prompt(
                mask=mask,
                positive_click=self.is_positive,
                tp="lasso",
                mask_volume_node=inference_volume,
            )
            self._active_inference_volume_override = None

            def _next():
                self.setup_prompts()
                # Start placing a new lasso
                self.ui.pbInteractionLasso.click()

            qt.QTimer.singleShot(0, _next)

    #
    #  -- Scribble
    #
    def on_scribble_clicked(self, checked=False):
        """
        Activates/deactivates the hidden Segment Editor's Paint effect on the
        scribble segment (bg or fg, depending on prompt type).
        """
        self.setup_prompts(skip_if_exists=True)

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

        if not checked:
            # Deactivate paint effect
            if self.scribble_editor_widget:
                self.scribble_editor_widget.setActiveEffectByName(
                    ""
                )  # Clears the active effect

            # Optionally clear or reset the segmentation node
            self._teardown_scribble_observer()

            return

        segment_id = "fg" if self.is_positive else "bg"

        # Set segmentation and segment
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)
        self.scribble_editor_node.SetSelectedSegmentID(segment_id)

        # Set reference volume
        volume_node = self.get_volume_node()
        self.scribble_editor_widget.setSourceVolumeNode(volume_node)
        # setSourceVolumeNode resets every slice background to the source volume,
        # so restore the sticky per-plane display selections afterward.
        self._schedule_plane_display_reapply()

        # Activate paint effect
        self.scribble_editor_widget.setActiveEffectByName("Paint")
        self.scribble_editor_widget.updateWidgetFromMRML()

        paint_effect = self.scribble_editor_widget.activeEffect()
        if paint_effect:
            paint_effect.setParameter("BrushUseAbsoluteSize", "0")  # Use relative mode
            paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            paint_effect.setParameter("BrushRelativeDiameter", ".75")
            self._scribble_labelmap_callback_tag = {
                "tag": self.scribble_segment_node.AddObserver(
                    vtk.vtkCommand.AnyEvent, self.on_scribble_finished
                ),
                "label_name": segment_id,
            }
        debug_print(f"Scribble mode (hidden editor) activated on '{segment_id}'")

    #
    #  -- Lasso/scribble
    #
    @ensure_synched
    def lasso_or_scribble_prompt(
        self,
        mask,
        positive_click=False,
        tp="lasso",
        mask_volume_node=None,
    ):
        """
        Uploads lasso or scribble prompt to the server.
        """
        _perf_log("[DEBUG triplanar.perf] lasso_or_scribble_prompt body (sync passed) "
              "tp={} mask_sum={}".format(tp, int(np.sum(np.asarray(mask)))),
              flush=True)
        if tp != "lasso":
            # Scribble results must never be clipped by a stale lasso marker.
            self._last_lasso_slice = None
        inference_volume = self.get_inference_volume_node()
        if mask_volume_node is None:
            mask_volume_node = inference_volume
        if mask_volume_node.GetID() != inference_volume.GetID():
            mask = self._resample_mask_between_volumes(
                mask, mask_volume_node, inference_volume
            )
        if np.sum(mask) == 0:
            # Nothing will be shown, so the marker would otherwise leak into
            # the next prompt's result.
            self._last_lasso_slice = None
            slicer.util.showStatusMessage(
                "Lasso/scribble produced an empty mask; nothing to send.", 4000
            )
            return

        url = f"{self.server}/add_{tp}_interaction"
        try:
            buffer = io.BytesIO()
            np.save(buffer, mask)
            compressed_data = gzip.compress(buffer.getvalue())

            from requests_toolbelt import MultipartEncoder

            fields = {
                "file": ("volume.npy.gz", compressed_data, "application/octet-stream"),
                "positive_click": str(
                    positive_click
                ),  # Make sure to send it as a string.
            }
            encoder = MultipartEncoder(fields=fields)
            seg_response = self.request_to_server(
                url,
                data=encoder,
                headers={
                    "Content-Type": encoder.content_type,
                    "Content-Encoding": "gzip",
                },
            )

            if seg_response.status_code == 200:
                unpacked_segmentation = self.unpack_binary_segmentation(
                    seg_response.content, decompress=False
                )
                self._handle_server_segmentation_result(unpacked_segmentation)
            else:
                debug_print(
                    f"lasso_or_scribble_prompt upload failed with status code: {seg_response.status_code}"
                )
                self._last_lasso_slice = None
                slicer.util.showStatusMessage(
                    f"Server rejected {tp} prompt (status "
                    f"{seg_response.status_code}).",
                    4000,
                )
        except Exception as e:
            debug_print(f"Error in lasso_or_scribble_prompt: {e}")
            self._last_lasso_slice = None
            slicer.util.showStatusMessage(
                f"Failed to send {tp} prompt: {e}", 4000
            )

    def on_scribble_finished(self, caller, event):
        """
        Called when the user completes a scribble stroke in the Paint effect.
        We calculate the diff in the drawn region and send it to the server.
        """
        debug_print("Scribble stroke finished - labelmap modified!")

        # Clean up observer if you only want it once
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            caller.RemoveObserver(self._scribble_labelmap_callback_tag["tag"])
            label_name = self._scribble_labelmap_callback_tag["label_name"]
            del self._scribble_labelmap_callback_tag
        else:
            return

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.scribble_segment_node, label_name, self.get_volume_node()
        )

        if (
            hasattr(self, "_prev_scribble_mask")
            and self._prev_scribble_mask is not None
        ):
            prev_scribble_mask = self._prev_scribble_mask
        else:
            prev_scribble_mask = mask * 0

        diff_mask = mask - prev_scribble_mask
        self._prev_scribble_mask = mask

        # Tri-planar routing: send the scribble to the series shown in the view
        # it was drawn in, detected from the painted plane's RAS centroid. The
        # diff mask is on the source grid; lasso_or_scribble_prompt resamples it
        # onto the routed series before upload.
        if self._triplanar_mode:
            self._route_prompt_to_view(
                self._mask_centroid_ras(diff_mask, self.get_volume_node())
            )
        try:
            self.lasso_or_scribble_prompt(
                mask=diff_mask,
                positive_click=self.is_positive,
                tp="scribble",
                mask_volume_node=self.get_volume_node(),
            )
        finally:
            self._active_inference_volume_override = None

        self.ui.pbInteractionScribble.click()  # turn it off
        self.ui.pbInteractionScribble.click()  # turn it on

    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def make_new_segment(self):
        """
        Creates a new empty segment in the current segmentation, increments a name,
        and sets it as the selected segment.
        """
        self._destroy_inference_preview()
        # After creating a new segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        debug_print("doing make_new_segment")
        segmentation_node = self.get_segmentation_node()

        # Generate a new segment name
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0:
            new_segment_name = "Segment_1"
        else:
            # Find the next available number
            segment_numbers = [
                int(seg.split("_")[-1])
                for seg in segment_ids
                if seg.startswith("Segment_") and seg.split("_")[-1].isdigit()
            ]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"

        # Create and add the new segment
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            new_segment_name
        )
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Make sure the right node is selected
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Apply the user's persisted opacity preference to the new segment so
        # the slider value survives across sessions / new segments.
        display_node = segmentation_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentOpacity(
                new_segment_id, self._get_preferred_segment_opacity()
            )

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment
        and updates the server.
        """
        # After clearing a segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()

        if selected_segment_id:
            debug_print(f"Clearing segment: {selected_segment_id}")
            self._destroy_inference_preview()
            # Drop collected per-series results too: otherwise the next autofuse
            # rebuilds the just-cleared segmentation from the stale stored masks.
            if self._fusion_results:
                print("[DEBUG fusion.collect] cleared store on clear-segment")
                self._fusion_results = {}
            self.show_segmentation(
                np.zeros(self._output_grid_shape(), dtype=np.uint8)
            )
            self.setup_prompts()
            self.upload_segment_to_server()
        else:
            debug_print("No segment selected to clear.")

    def show_segmentation(self, segmentation_mask):
        """
        Updates the currently selected segment with the given binary mask array.
        """
        t0 = time.time()
        segmentation_mask = self._apply_lasso_slice_clip(segmentation_mask)
        self._last_lasso_slice = None  # consume; non-lasso paths must not clip
        self.previous_states["segment_data"] = segmentation_mask

        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())

        _perf_log("[DEBUG triplanar.perf] show_segmentation: write labelmap "
              "shape={}".format(tuple(np.asarray(segmentation_mask).shape)),
              flush=True)
        with slicer.util.RenderBlocker():  # avoid flashing of 3D view
            self.ui.editor_widget.saveStateForUndo()
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segmentation_mask,
                segmentationNode,
                selectedSegmentID,
                self.get_output_volume_node(),
            )
            # Tri-planar fuses + redraws after every interaction; rebuilding the
            # closed surface (marching cubes) on the large output grid each time
            # is a heavy CPU/GPU hit and a likely freeze/crash source, so skip it
            # in tri-planar mode (the user can re-enable 3D when ready).
            if was_3d_shown and not self._triplanar_mode:
                _perf_log("[DEBUG triplanar.perf] CreateClosedSurfaceRepresentation "
                      "start", flush=True)
                segmentationNode.CreateClosedSurfaceRepresentation()
                _perf_log("[DEBUG triplanar.perf] CreateClosedSurfaceRepresentation "
                      "done", flush=True)
        # Tri-planar skips the per-interaction surface rebuild above; instead
        # show the 3D surface via a debounced rebuild so rapid interactions only
        # pay the heavy marching-cubes cost once, after the user pauses.
        if self._triplanar_mode and self._get_show_3d_triplanar():
            self._schedule_triplanar_3d_surface()
        _perf_log("[DEBUG triplanar.perf] show_segmentation: done", flush=True)

        # Mark the segment as being edited (can be useful for selective saving of only modified segments)
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        if segmentation_mask.sum() > 0:
            # If we do this when segmentation_mask.sum() == 0, sometimes Slicer will throw "bogus" OOM errors
            # (see https://github.com/coendevente/SlicerNNInteractive/issues/38)
            segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()

        # Writing the result labelmap does not touch slice-view backgrounds, so no
        # plane-display reapply is needed here. Only the hidden editors' explicit
        # setSourceVolumeNode calls reset backgrounds, and they reapply themselves.
        del segmentation_mask

        debug_print(f"show_segmentation took {time.time() - t0}")

    def _internal_segmentation_node_names(self):
        """Names of hidden scaffolding segmentations excluded from user lookups."""
        return {
            self.scribble_segment_node_name,
            self.wand_preview_segment_node_name,
            self.lasso3d_input_segment_node_name,
            self.lasso3d_preview_segment_node_name,
            self.inference_preview_segment_node_name,
        }

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one. Internal scaffolding nodes
        (scribble, magic wand preview) are excluded from this lookup.
        """
        internal_names = self._internal_segmentation_node_names()

        # If the segmentation widget has a currently selected segmentation node, return it.
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node:
            if segmentation_node.GetName() not in internal_names:
                return segmentation_node

        # Otherwise, fall back to getting the first suitable segmentation node
        segmentation_node = None
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segmentation_node in segmentation_nodes:
            if segmentation_node.GetName() in internal_names:
                segmentation_node = None
                continue

        # Create new segmentation node if none suitable found
        if not segmentation_node:
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Set segmentation node in widget
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        # The canonical output grid (source volume by default, or a high-res
        # isotropic grid when that feature is enabled).
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
            self.get_output_volume_node()
        )

        return segmentation_node

    def get_selected_segmentation_node_and_segment_id(self):
        """
        Retrieve the currently selected segmentation node & segment ID.
        If none, create one.
        """
        debug_print("doing get_selected_segmentation_node_and_segment_id")
        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            return self.make_new_segment()

        return segmentation_node, selected_segment_id

    def get_current_segment_id(self):
        """
        Returns the ID of the segment currently selected in the segment editor.
        """
        return self.ui.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

    def _segment_is_empty(self, segmentation_node, seg_id):
        """True when the segment has no foreground voxels, determined WITHOUT
        exporting to a reference geometry (Slicer's GenerateSharedLabelmap fails
        and logs a VTK error on empty segments). Reads the binary-labelmap
        representation directly. Returns False on any uncertainty so the caller
        still attempts the guarded export rather than dropping real data."""
        try:
            if segmentation_node is None or not seg_id:
                return True
            segmentation = segmentation_node.GetSegmentation()
            seg = segmentation.GetSegment(seg_id) if segmentation else None
            if seg is None:
                return True
            labelmap = seg.GetRepresentation(self._binary_labelmap_name())
            if labelmap is None or labelmap.GetNumberOfPoints() == 0:
                return True
            return labelmap.GetScalarRange()[1] <= 0
        except Exception:  # noqa: BLE001 - uncertain -> let the export try
            return False

    def get_segment_data(self, reference_volume_node=None):
        """
        Gets the selected segment on a requested scalar-volume geometry.

        The default is the canonical output grid (source volume, or the high-res
        isotropic grid when that feature is enabled).
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )
        if reference_volume_node is None:
            reference_volume_node = self.get_output_volume_node()

        try:
            ref_shape = slicer.util.arrayFromVolume(reference_volume_node).shape
        except Exception:
            ref_shape = None

        # Avoid exporting an EMPTY segment to a different reference geometry:
        # Slicer's GenerateSharedLabelmap / ResampleOrientedImageToReferenceGeometry
        # fails (and logs a VTK error) on an empty segment. Detect emptiness from
        # the binary-labelmap representation directly (no export, no resample) and
        # short-circuit to an all-background mask on the reference grid. Most
        # visible with the high-resolution output grid (and tri-planar mode that
        # forces it on) on oblique series.
        if self._segment_is_empty(segmentation_node, selected_segment_id):
            return (
                np.zeros(ref_shape, dtype=bool) if ref_shape is not None else None
            )

        # Non-empty: export onto the requested reference grid, guarded so a
        # resample failure (None / wrong shape / raise) degrades to an
        # all-background mask instead of crashing the prompt-sync chain.
        try:
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, selected_segment_id, reference_volume_node
            )
        except Exception as exc:  # noqa: BLE001
            print("[DEBUG segdata] reference export raised: {}".format(exc))
            mask = None
        if mask is None or (
            ref_shape is not None and tuple(mask.shape) != tuple(ref_shape)
        ):
            print("[DEBUG segdata] reference export empty/failed -> zeros "
                  "(mask_shape={}, ref_shape={})".format(
                      None if mask is None else tuple(mask.shape), ref_shape))
            if ref_shape is None:
                return None
            return np.zeros(ref_shape, dtype=bool)
        return mask.astype(bool)

    def selected_segment_changed(self):
        """
        Checks if the current segment mask has changed from our `self.previous_states`.
        """
        segment_data = self.get_segment_data()
        if segment_data is None:
            # Reference geometry unavailable; report "changed" so the upload
            # path runs and handles (or surfaces) the failure itself instead
            # of crashing the sync chain here.
            debug_print("segment_data is None; treating as changed")
            return True
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.array_equal(
            old_segment_data.astype(bool), segment_data.astype(bool)
        )

        debug_print(f"segment_data.sum(): {segment_data.sum()}")

        if old_segment_data is not None:
            debug_print(f"old_segment_data.sum(): {old_segment_data.sum()}")
        else:
            debug_print("old_segment_data is None")

        debug_print(f"selected_segment_changed: {selected_segment_changed}")

        return selected_segment_changed

    # -- Per-segment display opacity (right-side panel slider) --

    def _current_segmentation_display_node(self):
        seg_node = self.get_segmentation_node()
        if seg_node is None:
            return None
        return seg_node.GetDisplayNode()

    def _sync_opacity_slider_from_segment(self):
        """Push the active segment's current opacity onto the slider UI."""
        display_node = self._current_segmentation_display_node()
        seg_id = self.get_current_segment_id()
        enabled = display_node is not None and bool(seg_id)
        self.ui.sldSegmentOpacity.setEnabled(enabled)
        if not enabled:
            self.ui.lblSegOpacityValue.setText("--")
            return
        # vtkMRMLSegmentationDisplayNode exposes SetSegmentOpacity (master) but
        # not a matching GetSegmentOpacity; read back via the 3D dimension,
        # which SetSegmentOpacity also writes to.
        value = display_node.GetSegmentOpacity3D(seg_id)
        pct = int(round(float(value) * 100))
        blocked = self.ui.sldSegmentOpacity.blockSignals(True)
        try:
            self.ui.sldSegmentOpacity.setValue(pct)
        finally:
            self.ui.sldSegmentOpacity.blockSignals(blocked)
        self.ui.lblSegOpacityValue.setText(f"{pct} %")

    def _on_segment_opacity_changed(self, value):
        """Slider drag -- push opacity to the current segment and persist as
        a user preference applied to future newly-created segments."""
        self.ui.lblSegOpacityValue.setText(f"{int(value)} %")
        fraction = float(value) / 100.0
        self._save_preferred_segment_opacity(fraction)
        display_node = self._current_segmentation_display_node()
        seg_id = self.get_current_segment_id()
        if display_node is None or not seg_id:
            return
        display_node.SetSegmentOpacity(seg_id, fraction)

    def _get_preferred_segment_opacity(self):
        """Read the persisted preferred segment opacity (0..1). Default 1.0."""
        v = self._get_qsetting(SETTING_SEGMENT_OPACITY, 1.0, cast=float)
        return max(0.0, min(1.0, v))

    def _save_preferred_segment_opacity(self, value):
        """Persist the slider's current value as a user preference."""
        self._set_qsetting(SETTING_SEGMENT_OPACITY, float(value))

    # -- Non-destructive display smoothing (2D + 3D from the closed surface) --

    def _existing_segmentation_node(self):
        """The current editor segmentation node, or None (never creates one)."""
        node = self.ui.editor_widget.segmentationNode()
        if node is None:
            return None
        internal_names = self._internal_segmentation_node_names()
        if node.GetName() in internal_names:
            return None
        return node

    @staticmethod
    def _closed_surface_name():
        return (
            slicer.vtkSegmentationConverter
            .GetSegmentationClosedSurfaceRepresentationName()
        )

    @staticmethod
    def _binary_labelmap_name():
        return (
            slicer.vtkSegmentationConverter
            .GetSegmentationBinaryLabelmapRepresentationName()
        )

    def _get_display_smooth_enabled(self):
        """Read whether non-destructive display smoothing is on. Default False."""
        return self._get_qsetting(SETTING_DISPLAY_SMOOTH_ENABLED, False, cast=bool)

    def _get_display_smooth_strength(self):
        """Read the display smoothing strength in [0, 1]. Default 0.5."""
        v = self._get_qsetting(SETTING_DISPLAY_SMOOTH_STRENGTH, 0.5, cast=float)
        return float(min(1.0, max(0.0, v)))

    def _current_display_smooth_strength(self):
        """Strength from the spin box, falling back to the persisted value."""
        if hasattr(self, "ui"):
            try:
                v = float(self.ui.sbDisplaySmoothStrength.value)
            except (TypeError, ValueError):
                v = self._get_display_smooth_strength()
            return float(min(1.0, max(0.0, v)))
        return self._get_display_smooth_strength()

    def _apply_display_smoothing(self):
        """Smooth the closed surface and render 2D + 3D from it (non-destructive).

        Sets a segmentation-level "Smoothing factor" (a windowed-sinc pass that
        Slicer re-applies whenever the surface is rebuilt) and points the segment
        display node's 2D and 3D preferred representations at the closed surface.
        The binary labelmap (the stored data) is never modified.
        """
        seg_node = self._existing_segmentation_node()
        if seg_node is None:
            return
        display_node = seg_node.GetDisplayNode()
        if display_node is None:
            return
        try:
            segmentation = seg_node.GetSegmentation()
            strength = self._current_display_smooth_strength()
            segmentation.SetConversionParameter("Smoothing factor", str(strength))
            # Drop any stale surface so it is rebuilt with the current factor.
            segmentation.RemoveRepresentation(self._closed_surface_name())
            seg_node.CreateClosedSurfaceRepresentation()
            surface_name = self._closed_surface_name()
            display_node.SetPreferredDisplayRepresentationName2D(surface_name)
            display_node.SetPreferredDisplayRepresentationName3D(surface_name)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print("[nni] apply display smoothing failed: %s" % exc)

    def _clear_display_smoothing(self):
        """Render 2D + 3D from the binary labelmap again (data untouched)."""
        seg_node = self._existing_segmentation_node()
        if seg_node is None:
            return
        display_node = seg_node.GetDisplayNode()
        if display_node is None:
            return
        try:
            labelmap_name = self._binary_labelmap_name()
            display_node.SetPreferredDisplayRepresentationName2D(labelmap_name)
            display_node.SetPreferredDisplayRepresentationName3D(labelmap_name)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print("[nni] clear display smoothing failed: %s" % exc)

    def _refresh_display_smooth_ui(self):
        """Enable strength/bake only when display smoothing is on."""
        if not hasattr(self, "ui"):
            return
        enabled = self.ui.cbDisplaySmooth.isChecked()
        self.ui.sbDisplaySmoothStrength.setEnabled(enabled)
        self.ui.pbBakeDisplaySmooth.setEnabled(enabled)

    def _on_display_smooth_enabled_changed(self, checked):
        """Persist the toggle and apply or clear the display-only smoothing."""
        self._set_qsetting(SETTING_DISPLAY_SMOOTH_ENABLED, bool(checked))
        if checked:
            self._apply_display_smoothing()
        else:
            self._clear_display_smoothing()
        self._refresh_display_smooth_ui()

    def _on_display_smooth_strength_changed(self, value):
        """Persist the strength and re-apply if display smoothing is active."""
        self._set_qsetting(
            SETTING_DISPLAY_SMOOTH_STRENGTH,
            self._current_display_smooth_strength(),
        )
        if hasattr(self, "ui") and self.ui.cbDisplaySmooth.isChecked():
            self._apply_display_smoothing()

    def _reapply_display_smoothing_if_active(self):
        """Re-apply display smoothing after the active segmentation changes."""
        if hasattr(self, "ui") and self.ui.cbDisplaySmooth.isChecked():
            self._apply_display_smoothing()

    def on_bake_display_smooth_clicked(self, checked=False):
        """Bake the currently displayed smooth surface back into the segment.

        This is the explicit, destructive "export" step: convert the smoothed
        closed surface to a binary labelmap on the output grid and store it as
        the segment data. A snapshot of the original labelmap is restored if any
        step fails, so a bad conversion never corrupts the segment.
        """
        seg_node = self._existing_segmentation_node()
        segment_id = self.get_current_segment_id()
        if seg_node is None or not segment_id:
            slicer.util.showStatusMessage("No segment selected to bake.", 4000)
            return
        output_volume = self.get_output_volume_node()
        if output_volume is None:
            slicer.util.showStatusMessage("No output volume available.", 4000)
            return

        original = None
        try:
            original = slicer.util.arrayFromSegmentBinaryLabelmap(
                seg_node, segment_id, output_volume
            )
            if original is not None:
                original = original.copy()

            segmentation = seg_node.GetSegmentation()
            strength = self._current_display_smooth_strength()
            segmentation.SetConversionParameter("Smoothing factor", str(strength))
            segmentation.RemoveRepresentation(self._closed_surface_name())
            seg_node.CreateClosedSurfaceRepresentation()

            # Rebuild the binary labelmap from the smoothed surface on the output
            # grid, then read it back as the new segment data.
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(
                output_volume
            )
            segmentation.RemoveRepresentation(self._binary_labelmap_name())
            seg_node.CreateBinaryLabelmapRepresentation()
            baked = slicer.util.arrayFromSegmentBinaryLabelmap(
                seg_node, segment_id, output_volume
            )
            if baked is None:
                raise RuntimeError("Could not rebuild labelmap from surface.")

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                baked.astype(np.uint8), seg_node, segment_id, output_volume
            )
        except Exception as exc:  # noqa: BLE001 - never leave bad data
            print("[nni] bake smoothed surface failed: %s" % exc)
            if original is not None:
                try:
                    slicer.util.updateSegmentBinaryLabelmapFromArray(
                        original.astype(np.uint8),
                        seg_node,
                        segment_id,
                        output_volume,
                    )
                except Exception:
                    pass
            slicer.util.showStatusMessage(
                "Bake failed; the original segment was restored.", 5000
            )
            return

        # The stored data is now smooth, so drop the display-only override and
        # push the baked mask to the server.
        if hasattr(self, "ui"):
            blocked = self.ui.cbDisplaySmooth.blockSignals(True)
            self.ui.cbDisplaySmooth.setChecked(False)
            self.ui.cbDisplaySmooth.blockSignals(blocked)
            self._set_qsetting(SETTING_DISPLAY_SMOOTH_ENABLED, False)
        self._clear_display_smoothing()
        self._refresh_display_smooth_ui()
        self.upload_segment_to_server()
        slicer.util.showStatusMessage(
            "Smoothed surface baked into the segment.", 4000
        )

    # -- Lasso slice-range clipping (keep only the lasso slice +/- N) --

    def _get_lasso_clip_enabled(self):
        """Read whether lasso slice-range clipping is enabled. Default False."""
        return self._get_qsetting(SETTING_LASSO_CLIP_ENABLED, False, cast=bool)

    def _get_lasso_clip_n(self):
        """Read the persisted +/- N slices for lasso clipping. Default 0."""
        return max(0, self._get_qsetting(SETTING_LASSO_CLIP_N, 0, cast=int))

    def _on_lasso_clip_enabled_changed(self, checked):
        """Persist the lasso-clip enable checkbox."""
        self._set_qsetting(SETTING_LASSO_CLIP_ENABLED, bool(checked))

    def _on_lasso_clip_n_changed(self, value):
        """Persist the lasso-clip +/- N slices spin box."""
        self._set_qsetting(SETTING_LASSO_CLIP_N, int(value))

    # -- Multi-view lasso accumulation --

    def _get_lasso_multiview_enabled(self):
        return self._get_qsetting(
            SETTING_LASSO_MULTIVIEW_ENABLED, False, cast=bool
        )

    def _set_lasso_multiview_enabled(self, val):
        self._set_qsetting(SETTING_LASSO_MULTIVIEW_ENABLED, bool(val))

    def _on_lasso_multiview_toggled(self, checked):
        self._set_lasso_multiview_enabled(checked)
        self._multiview_lasso_masks = []
        self._multiview_lasso_count = 0
        self._clear_multiview_lasso_nodes()
        self._reset_active_lasso_node()
        self._update_multiview_lasso_ui()

    def _on_lasso_multiview_submit_clicked(self):
        masks = self._multiview_lasso_masks
        if not masks:
            return
        self._multiview_lasso_masks = []
        self._multiview_lasso_count = 0
        self._clear_multiview_lasso_nodes()
        self._update_multiview_lasso_ui()
        inference_volume = self.get_inference_volume_node()
        if inference_volume is None:
            slicer.util.showStatusMessage("No inference volume available.", 4000)
            return
        # Send each plane as its own lasso interaction; the server session
        # accumulates them so the result is constrained by all views.
        for m in masks:
            self.lasso_or_scribble_prompt(
                mask=m,
                positive_click=self.is_positive,
                tp="lasso",
                mask_volume_node=inference_volume,
            )

    def _on_lasso_multiview_clear_clicked(self):
        self._multiview_lasso_masks = []
        self._multiview_lasso_count = 0
        self._clear_multiview_lasso_nodes()
        self._reset_active_lasso_node()
        self._update_multiview_lasso_ui()
        slicer.util.showStatusMessage(
            "Multi-view lasso accumulation cleared.", 2000
        )

    def _update_multiview_lasso_ui(self):
        enabled = self._get_lasso_multiview_enabled()
        has_mask = len(self._multiview_lasso_masks) > 0
        self.ui.pbLassoMultiViewSubmit.setVisible(enabled)
        self.ui.pbLassoMultiViewClear.setVisible(enabled)
        self.ui.pbLassoMultiViewSubmit.setEnabled(bool(has_mask))
        self.ui.pbLassoMultiViewClear.setEnabled(bool(has_mask))
        count = len(self._multiview_lasso_masks)
        self.ui.pbLassoMultiViewSubmit.setText("Submit ({})".format(count))

    def _clear_multiview_lasso_nodes(self):
        for node in self._multiview_lasso_nodes:
            try:
                if slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass
        self._multiview_lasso_nodes = []

    def _reset_active_lasso_node(self):
        """Remove all drawn points from the current active lasso node so it
        stops rendering in all views (including 3D) without destroying the node."""
        try:
            node = self.prompt_types["lasso"]["node"]
            if node is not None:
                node.RemoveAllControlPoints()
        except Exception:
            pass
        if hasattr(self, "ui"):
            self.ui.pbInteractionLassoCancel.setVisible(False)

    def _reset_multiview_lasso_accumulation(self):
        self._multiview_lasso_masks = []
        self._multiview_lasso_count = 0
        self._clear_multiview_lasso_nodes()
        if hasattr(self, "ui"):
            self._update_multiview_lasso_ui()

    def _get_high_res_enabled_setting(self):
        """Read whether the high-resolution output grid is enabled. Default False."""
        return self._get_qsetting(SETTING_HIGH_RES_ENABLED, False, cast=bool)

    def _get_output_spacing_setting(self):
        """Read the persisted isotropic output spacing in mm (0 = auto)."""
        return max(0.0, self._get_qsetting(SETTING_OUTPUT_SPACING, 0.0, cast=float))

    def _remove_output_geometry_node(self):
        """Drop the hidden output-geometry volume and reset its bookkeeping."""
        node = self._output_geometry_node
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
        self._output_geometry_node = None
        self._output_geometry_spacing = None
        self._output_geometry_source_id = None
        self._output_geometry_triplanar_sig = None

    def _disable_high_res_output(self, reason):
        """Turn the high-res output feature off after a failure, without recursion."""
        self._clear_selection_op_undo_stack()
        self._remove_output_geometry_node()
        if hasattr(self, "ui"):
            blocked = self.ui.cbEnableHighResOutput.blockSignals(True)
            self.ui.cbEnableHighResOutput.setChecked(False)
            self.ui.cbEnableHighResOutput.blockSignals(blocked)
        self._set_qsetting(SETTING_HIGH_RES_ENABLED, False)
        slicer.util.showStatusMessage(reason, 6000)
        print("[nni] high-resolution output disabled: %s" % reason)

    def _rebuild_output_geometry_and_migrate(self):
        """Re-derive the current segment onto the (new) output grid after a change."""
        _perf_log("[DEBUG triplanar.perf] rebuild_output start")
        source_volume = self.get_volume_node()
        if source_volume is None:
            return
        self._clear_selection_op_undo_stack(
            "Output grid changed; Selection Operations undo history was "
            "cleared."
        )
        # Capture the current segment on the invariant source grid before the
        # output grid changes.
        try:
            src_mask = self.get_segment_data(reference_volume_node=source_volume)
        except Exception:
            src_mask = None
        # A stale supplemental-series preview lives on the old grid; discard it.
        self._destroy_inference_preview()
        if hasattr(self, "previous_states"):
            self.previous_states.pop("segment_data", None)
        if not self._high_res_output_enabled():
            self._remove_output_geometry_node()
        # Touch the output volume so the geometry node is (re)built when enabled,
        # then point the segmentation reference geometry at it.
        output_volume = self.get_output_volume_node()
        try:
            _dims = (
                output_volume.GetImageData().GetDimensions()
                if output_volume is not None
                and output_volume.GetImageData() is not None
                else None
            )
        except Exception:  # noqa: BLE001
            _dims = None
        _perf_log("[DEBUG triplanar.perf] rebuild_output: output grid dims=%s"
                  % (_dims,))
        seg_node = self.get_segmentation_node()
        if seg_node is not None and output_volume is not None:
            seg_node.SetReferenceImageGeometryParameterFromVolumeNode(output_volume)
        # Rewrite the current segment onto the new grid. If the resample fails
        # (e.g. the grid is too large), fall back to the source grid instead of
        # crashing the toggle handler.
        if src_mask is not None:
            migrated = self._to_output_grid(src_mask)
            if migrated is None:
                self._disable_high_res_output(
                    "High-resolution output grid could not be built; "
                    "reverted to the source grid."
                )
                if seg_node is not None:
                    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(
                        source_volume
                    )
                migrated = src_mask
            _perf_log("[DEBUG triplanar.perf] rebuild_output: migrate show")
            self.show_segmentation(migrated)
        _perf_log("[DEBUG triplanar.perf] rebuild_output done")

    def _get_smooth_interpolate_setting(self):
        """Read whether smooth (interpolated) results are enabled. Default False."""
        return self._get_qsetting(
            SETTING_SMOOTH_INTERPOLATE_ENABLED, False, cast=bool
        )

    def _on_high_res_output_changed(self, checked):
        """Persist the high-res output toggle and migrate the current segment."""
        self._set_qsetting(SETTING_HIGH_RES_ENABLED, bool(checked))
        # Smoothing interpolates onto the fine grid, so it cannot run without
        # the high-resolution output. Turn it off in lockstep.
        if (
            not checked
            and hasattr(self, "ui")
            and self.ui.cbSmoothInterpolate.isChecked()
        ):
            blocked = self.ui.cbSmoothInterpolate.blockSignals(True)
            self.ui.cbSmoothInterpolate.setChecked(False)
            self.ui.cbSmoothInterpolate.blockSignals(blocked)
            self._set_qsetting(SETTING_SMOOTH_INTERPOLATE_ENABLED, False)
            slicer.util.showStatusMessage(
                "Smoothing disabled (needs high-resolution output).", 4000
            )
        self._rebuild_output_geometry_and_migrate()
        if hasattr(self, "ui"):
            self._refresh_native_series_inference_ui()

    def _enable_high_res_for_smoothing(self):
        """Turn on the fine output grid that smoothing needs (no-op if active)."""
        if self._output_geometry_active():
            _perf_log("[DEBUG triplanar.perf] enable_high_res: already active")
            return
        _perf_log("[DEBUG triplanar.perf] enable_high_res start")
        blocked = self.ui.cbEnableHighResOutput.blockSignals(True)
        self.ui.cbEnableHighResOutput.setChecked(True)
        self.ui.cbEnableHighResOutput.blockSignals(blocked)
        self._set_qsetting(SETTING_HIGH_RES_ENABLED, True)
        self._rebuild_output_geometry_and_migrate()
        _perf_log("[DEBUG triplanar.perf] enable_high_res: after rebuild")
        self._refresh_native_series_inference_ui()
        slicer.util.showStatusMessage(
            "High-resolution output enabled for smoothing.", 4000
        )

    def _on_smooth_interpolate_changed(self, checked):
        """Persist the smoothing toggle; auto-enable the fine output grid it needs."""
        self._set_qsetting(SETTING_SMOOTH_INTERPOLATE_ENABLED, bool(checked))
        if checked:
            self._enable_high_res_for_smoothing()

    def on_smooth_current_segment_clicked(self, checked=False):
        """Re-run shape-based smoothing on the whole current segment.

        Manual edits (the built-in Erase/Paint/Scissors effects) write the
        segment labelmap directly and never pass through the server result
        chokepoint, so their boundaries stay stair-stepped. This button reuses
        the same SDF interpolation to smooth the current segment on demand.
        """
        source_volume = self.get_volume_node()
        if source_volume is None:
            slicer.util.showStatusMessage("No source volume selected.", 4000)
            return
        self._enable_high_res_for_smoothing()
        if not self._output_geometry_active():
            slicer.util.showStatusMessage(
                "Could not enable high-resolution output for smoothing.", 4000
            )
            return
        # The source volume is fine in-plane and only coarse through-plane, so
        # reading the segment on its grid keeps in-plane detail; the SDF pass
        # then interpolates smoothly through-plane onto the fine output grid.
        coarse = self.get_segment_data(reference_volume_node=source_volume)
        if coarse is None or int(coarse.sum()) == 0:
            slicer.util.showStatusMessage(
                "Current segment is empty; nothing to smooth.", 4000
            )
            return
        smoothed = self._interpolate_mask_to_output_grid(coarse, source_volume)
        if smoothed is None:
            slicer.util.showStatusMessage("Smoothing failed.", 4000)
            return
        self.show_segmentation(smoothed)
        slicer.util.showStatusMessage("Current segment smoothed.", 3000)

    def _on_output_spacing_changed(self, value):
        """Persist the output spacing and, if enabled, rebuild the output grid."""
        self._set_qsetting(SETTING_OUTPUT_SPACING, float(value))
        if self._high_res_output_enabled():
            self._rebuild_output_geometry_and_migrate()

    def _apply_lasso_slice_clip(self, mask):
        """If enabled and the last prompt was a lasso, keep only the slices in
        [center-N, center+N] along the lasso plane axis and zero out the rest.
        mask is (z, y, x) uint8. Returns a new array, or the original on no-op.
        """
        if not self._get_lasso_clip_enabled():
            return mask
        # The recorded slice index is in source/inference voxels; it does not map
        # to the high-resolution output grid, so skip clipping when that is active.
        if self._output_geometry_active():
            return mask
        info = self._last_lasso_slice
        if info is None:
            return mask
        axis, center = info
        if axis not in (0, 1, 2):
            return mask
        n = self._get_lasso_clip_n()
        dim = mask.shape[axis]
        lo = max(0, center - n)
        hi = min(dim, center + n + 1)  # exclusive upper bound
        if lo >= hi:
            return mask
        clipped = np.zeros_like(mask)
        idx = [slice(None)] * 3
        idx[axis] = slice(lo, hi)
        idx = tuple(idx)
        clipped[idx] = mask[idx]
        return clipped

    ###############################################################################
    # Selection operations (boolean editing)
    ###############################################################################

    def get_operand_segment_ids(self):
        """
        Returns a list of (segment_id, segment_name) for every segment in the
        active segmentation node except the current target segment.
        """
        result = []
        segmentation_node = self.get_segmentation_node()
        if segmentation_node is None:
            return result

        target_id = self.get_current_segment_id()
        segmentation = segmentation_node.GetSegmentation()
        for segment_id in segmentation.GetSegmentIDs():
            if segment_id == target_id:
                continue
            segment = segmentation.GetSegment(segment_id)
            name = segment.GetName() if segment else segment_id
            result.append((segment_id, name))

        return result

    def populate_operand_selector(self):
        """
        Refreshes the operand segment combo box. The stable segment ID is stored
        as item data so renames do not break the current selection.
        """
        combo = self.ui.cbSelectionOperand
        previous_id = combo.currentData if combo.count > 0 else None

        combo.blockSignals(True)
        combo.clear()
        for segment_id, name in self.get_operand_segment_ids():
            combo.addItem(name, segment_id)

        if previous_id is not None:
            for i in range(combo.count):
                if combo.itemData(i) == previous_id:
                    combo.setCurrentIndex(i)
                    break
        combo.blockSignals(False)

        self.ui.cbSelectionOperand.setEnabled(combo.count > 0)
        self._refresh_apply_enabled()

    def segment_id_to_mask(self, segment_id):
        """
        Returns the binary (bool) mask of an arbitrary segment, sampled on the
        current volume geometry so it is shape-aligned with the target segment.
        """
        segmentation_node = self.get_segmentation_node()
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, segment_id, self.get_volume_node()
        )
        if mask is None:
            return np.zeros(self.get_image_data().shape, dtype=bool)
        return mask.astype(bool)

    @staticmethod
    def compute_boolean_mask(target_mask, operand_mask, operation):
        """
        Pure set-algebra helper. `operation` is 0=Add, 1=Subtract, 2=Intersect.
        Returns a uint8 mask. Raises ValueError on shape mismatch or bad operation.
        """
        target_mask = target_mask.astype(bool)
        operand_mask = operand_mask.astype(bool)

        if target_mask.shape != operand_mask.shape:
            raise ValueError(
                "Target and operand masks have different shapes: "
                f"{target_mask.shape} vs {operand_mask.shape}."
            )

        if operation == 0:  # Add: S OR M
            result_mask = target_mask | operand_mask
        elif operation == 1:  # Subtract: S AND NOT M
            result_mask = target_mask & ~operand_mask
        elif operation == 2:  # Intersect: S AND M
            result_mask = target_mask & operand_mask
        else:
            raise ValueError(f"Unknown operation index: {operation}")

        return result_mask.astype(np.uint8)

    def apply_boolean_operation(self, operand_mask, operation):
        """
        Computes a boolean set operation between the current segment and the
        operand mask. Does not write back or upload.
        """
        # Operands are rasterized on the source grid; the target segment lives on
        # the canonical output grid. Bridge the operand so shapes match (no-op
        # when the output grid equals the source grid).
        bridged = self._to_output_grid(operand_mask)
        if bridged is None:
            self._disable_high_res_output(
                "High-resolution output resample failed; "
                "reverted to the source grid."
            )
            bridged = operand_mask
        return self.compute_boolean_mask(
            self.get_segment_data(), bridged, operation
        )

    def on_apply_selection_op_clicked(self, checked=False):
        """
        Applies the selected boolean operation to the current segment, writes the
        result back, and syncs it to the server. The operand can be either another
        segment or a 3D-positioned ROI box, depending on cbOperandSource.
        """
        self.populate_operand_selector()

        target_id = self.get_current_segment_id()
        if not target_id:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "Please select a target segment first.",
            )
            return

        # Operand source order: 0=ROI box, 1=Magic wand, 2=Segment, 3=Lasso (3D).
        source = self.ui.cbOperandSource.currentIndex
        if source == OPERAND_SOURCE_ROI:
            if not self._is_selection_roi_valid():
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Click 'Place / Show ROI' to position an ROI before applying.",
                )
                return
            operand_mask = self.roi_node_to_mask(self._sel_op_roi_node)
        elif source == OPERAND_SOURCE_WAND:
            if not self._is_selection_wand_seed_valid():
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Click 'Add Seed' and place a seed before applying.",
                )
                return
            operand_mask = self._compute_magic_wand_mask()
            if operand_mask is None:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Magic wand failed (no positive seed, server unreachable, "
                    "or seeds out of volume).",
                )
                return
        elif source == OPERAND_SOURCE_LASSO3D:
            if not self._is_selection_lasso3d_valid():
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Draw a lasso region in the 3D view before applying.",
                )
                return
            operand_mask = self._compute_lasso3d_mask()
            if operand_mask is None:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "Lasso (3D) failed (empty region, server unreachable, "
                    "or out of volume).",
                )
                return
        else:
            operand_id = self.ui.cbSelectionOperand.currentData
            if not operand_id:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "No operand segment is available. Add another segment first.",
                )
                return

            if operand_id == target_id:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Selection Operations",
                    "The operand segment must be different from the target segment.",
                )
                return

            operand_mask = self.segment_id_to_mask(operand_id)

        if operand_mask.sum() == 0:
            slicer.util.showStatusMessage(
                "Operand is empty; the operation may have no effect.", 3000
            )

        operation = self.ui.cbSelectionOperation.currentIndex
        try:
            result_mask = self.apply_boolean_operation(operand_mask, operation)
        except ValueError as e:
            QMessageBox.critical(
                slicer.util.mainWindow(),
                "Selection Operations",
                f"Could not apply the operation:\n\n{e}",
            )
            return

        # Snapshot the pre-Apply target so our own Undo can restore it
        # reliably -- the embedded Segment Editor's history stack is not
        # always populated for these programmatic edits.
        pre_state = self.get_segment_data()
        if pre_state is None:
            slicer.util.showStatusMessage(
                "Could not read the current segment; the operation was not "
                "applied.",
                4000,
            )
            return
        pre_state = pre_state.astype(np.uint8).copy()
        self._record_selection_op_undo(target_id, pre_state)

        self.show_segmentation(result_mask)
        # setup_prompts rebuilds the hidden scribble editor, which resets slice
        # backgrounds; it schedules a sticky per-plane reapply on its own.
        self.setup_prompts()
        # The wand preview reflected the about-to-apply mask -- clear it now
        # that the operation has landed on the actual target segment.
        self._clear_wand_preview_segment()
        # Same for the lasso (3D) region/preview, and disarm the Scissors tool.
        self.ui.pbDrawLasso3d.setChecked(False)
        self._deactivate_lasso3d_scissors()
        self._clear_lasso3d_input_segment()
        self._clear_lasso3d_preview_segment()

        sync_result = self.upload_segment_to_server()
        if sync_result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The operation was applied locally, but syncing to the server "
                "failed. You can retry with the 'Sync to server' button.",
            )
        else:
            slicer.util.showStatusMessage(
                "Selection operation applied and synced to server.", 3000
            )

    def on_sync_to_server_clicked(self, checked=False):
        """
        Pushes the current segment's mask to the server. Useful after editing the
        segment with native Segment Editor effects.
        """
        if self.image_changed(do_prev_image_update=False):
            image_result = self.upload_image_to_server()
            if image_result is None:
                QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Sync to server",
                    "Failed to sync the inference image to the server. Please "
                    "check the server connection in the 'Configuration' tab.",
                )
                return
        result = self.upload_segment_to_server()
        if result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Sync to server",
                "Failed to sync the current segment to the server. Please check "
                "the server connection in the 'Configuration' tab.",
            )
            return

        # Keep previous_states in sync so the next prompt's @ensure_synched does
        # not re-upload the identical mask.
        self.previous_states["segment_data"] = self.get_segment_data()
        slicer.util.showStatusMessage("Current segment synced to server.", 3000)

    def _install_selection_op_observers(self):
        """
        Observes segmentation and segment-editor changes so the operand selector
        stays up to date. Registration is idempotent.
        """
        self._safe_add_observer(
            self.segment_editor_node,
            vtk.vtkCommand.ModifiedEvent,
            self.on_segment_editor_node_modified,
        )

        self._observe_active_segmentation()

    def _observe_active_segmentation(self):
        """
        (Re)attaches observers on the active segmentation so adding, removing or
        renaming segments refreshes the operand selector.
        """
        segmentation_node = self.get_segmentation_node()
        if segmentation_node is None:
            return

        segmentation = segmentation_node.GetSegmentation()
        previous = getattr(self, "_observed_segmentation", None)
        if previous is segmentation:
            return

        events = (
            slicer.vtkSegmentation.SegmentAdded,
            slicer.vtkSegmentation.SegmentRemoved,
            slicer.vtkSegmentation.SegmentModified,
        )
        if previous is not None:
            for event in events:
                self._safe_remove_observer(
                    previous, event, self.on_segmentation_modified
                )

        for event in events:
            self.addObserver(segmentation, event, self.on_segmentation_modified)

        self._observed_segmentation = segmentation

    def on_segmentation_modified(self, caller, event):
        """Refresh the operand selector when segments are added/removed/renamed."""
        self.populate_operand_selector()
        self._sync_opacity_slider_from_segment()
        self._reapply_display_smoothing_if_active()

    def on_segment_editor_node_modified(self, caller, event):
        """Refresh observers and operand list when the node/segment selection changes."""
        self._observe_active_segmentation()
        self.populate_operand_selector()
        self._sync_opacity_slider_from_segment()
        self._reapply_display_smoothing_if_active()

    # -- ROI operand --

    def roi_node_to_mask(self, roi_node, shape_idx=None):
        """
        Rasterize a vtkMRMLMarkupsROINode into a bool numpy mask by testing
        each candidate voxel inside the ROI's local (OBB) frame. This handles
        obliquely-acquired volumes and rotated ROIs correctly (an earlier
        world-AABB approach over-included voxels heavily for oblique scans).
        The candidate IJK box is first restricted by the ROI's world AABB
        (clamped to the volume) so iteration stays bounded.

        shape_idx overrides the ROI shape (ROI_SHAPE_BOX/SPHERE/ELLIPSOID);
        when None it follows the cbRoiShape selector. Crop-by-box passes
        ROI_SHAPE_BOX so it never depends on the Selection Operations UI.
        """
        # --- Production geometry fetch ---
        radius = [0.0, 0.0, 0.0]
        roi_node.GetRadiusXYZ(radius)
        rx, ry, rz = radius
        volume = self.get_volume_node()
        shape = self.get_image_data().shape
        mask = np.zeros(shape, dtype=bool)
        if min(rx, ry, rz) <= 0.0 or volume is None:
            return mask

        world_bounds = [0.0] * 6
        roi_node.GetRASBounds(world_bounds)
        xmin, xmax, ymin, ymax, zmin, zmax = world_bounds
        corners_ras = [
            (xmin, ymin, zmin), (xmax, ymin, zmin),
            (xmin, ymax, zmin), (xmax, ymax, zmin),
            (xmin, ymin, zmax), (xmax, ymin, zmax),
            (xmin, ymax, zmax), (xmax, ymax, zmax),
        ]
        corners_ijk = [self.ras_to_xyz(list(c)) for c in corners_ras]
        ijk_arr = np.array(corners_ijk)
        i_lo_raw = int(ijk_arr[:, 0].min())
        i_hi_raw = int(ijk_arr[:, 0].max())
        j_lo_raw = int(ijk_arr[:, 1].min())
        j_hi_raw = int(ijk_arr[:, 1].max())
        k_lo_raw = int(ijk_arr[:, 2].min())
        k_hi_raw = int(ijk_arr[:, 2].max())
        i_lo_c = max(0, i_lo_raw)
        i_hi_c = min(shape[2] - 1, i_hi_raw)
        j_lo_c = max(0, j_lo_raw)
        j_hi_c = min(shape[1] - 1, j_hi_raw)
        k_lo_c = max(0, k_lo_raw)
        k_hi_c = min(shape[0] - 1, k_hi_raw)

        # vtkMRMLMarkupsROINode.GetObjectToWorldMatrix() is a 0-arg accessor
        # in this Slicer's PythonQt binding, returning the vtkMatrix4x4 directly
        # -- not the out-parameter style used by vtkMRMLScalarVolumeNode below.
        object_to_world_vtk = roi_node.GetObjectToWorldMatrix()
        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(ijk_to_ras_vtk)

        def _m_to_np(m):
            return np.array(
                [[m.GetElement(r, c) for c in range(4)] for r in range(4)]
            )

        ijk_to_object = (
            np.linalg.inv(_m_to_np(object_to_world_vtk))
            @ _m_to_np(ijk_to_ras_vtk)
        )

        # --- TEMP DIAGNOSTICS (remove once verified) ---
        center = [0.0, 0.0, 0.0]
        roi_node.GetCenter(center)
        local_bounds = [0.0] * 6
        roi_node.GetBounds(local_bounds)
        print("[SelectionOps] roi_node_to_mask diagnostics:")
        print(f"  ROI: name={roi_node.GetName()} id={roi_node.GetID()}")
        print(f"  ROI center (RAS): {center}")
        print(f"  ROI radius:       {radius}")
        print(f"  GetBounds (local? old code): {local_bounds}")
        print(f"  GetRASBounds (world, used): {world_bounds}")
        print(
            f"  Volume: name={volume.GetName()}"
            f" shape(z,y,x)={shape}"
        )
        print(f"  Volume spacing:    {tuple(volume.GetSpacing())}")
        print(f"  Volume origin:     {tuple(volume.GetOrigin())}")
        vrb = [0.0] * 6
        volume.GetRASBounds(vrb)
        print(f"  Volume RAS bounds: {vrb}")
        print("  Volume IJK->RAS matrix:")
        for row in range(4):
            print(
                "    [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                    ijk_to_ras_vtk.GetElement(row, 0),
                    ijk_to_ras_vtk.GetElement(row, 1),
                    ijk_to_ras_vtk.GetElement(row, 2),
                    ijk_to_ras_vtk.GetElement(row, 3),
                )
            )
        print(f"  ROI center IJK:    {self.ras_to_xyz(list(center))}")
        print(f"  8 RAS corners: {corners_ras}")
        print(f"  8 IJK corners: {corners_ijk}")
        print(
            f"  Raw IJK box:     i=[{i_lo_raw},{i_hi_raw}] "
            f"j=[{j_lo_raw},{j_hi_raw}] k=[{k_lo_raw},{k_hi_raw}]"
        )
        print(
            f"  Clamped IJK box: i=[{i_lo_c},{i_hi_c}] "
            f"j=[{j_lo_c},{j_hi_c}] k=[{k_lo_c},{k_hi_c}]"
        )
        seg_node = self.get_segmentation_node()
        print(
            f"  Segmentation node: name={seg_node.GetName() if seg_node else None}"
            f" id={seg_node.GetID() if seg_node else None}"
        )
        # --- END TEMP DIAGNOSTICS ---

        if i_hi_c < i_lo_c or j_hi_c < j_lo_c or k_hi_c < k_lo_c:
            print("  Resulting mask voxel count: 0")
            return mask

        # Exact OBB containment in ROI-local space. Vectorized over the
        # candidate IJK box.
        ii, jj, kk = np.meshgrid(
            np.arange(i_lo_c, i_hi_c + 1, dtype=np.float64),
            np.arange(j_lo_c, j_hi_c + 1, dtype=np.float64),
            np.arange(k_lo_c, k_hi_c + 1, dtype=np.float64),
            indexing="ij",
        )
        M = ijk_to_object
        x = M[0, 0] * ii + M[0, 1] * jj + M[0, 2] * kk + M[0, 3]
        y = M[1, 0] * ii + M[1, 1] * jj + M[1, 2] * kk + M[1, 3]
        z = M[2, 0] * ii + M[2, 1] * jj + M[2, 2] * kk + M[2, 3]

        if shape_idx is None:
            shape_idx = self.ui.cbRoiShape.currentIndex  # 0=Box,1=Sphere,2=Ellipsoid
        if shape_idx == ROI_SHAPE_SPHERE:
            # Sphere: inscribed in the (possibly non-cube) ROI box.
            r = min(rx, ry, rz)
            inside = (x * x + y * y + z * z) <= (r * r)
        elif shape_idx == ROI_SHAPE_ELLIPSOID:
            # Ellipsoid aligned with the ROI axes.
            inside = (
                (x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2
            ) <= 1.0
        else:
            # Box (oriented bounding box, default).
            inside = (np.abs(x) <= rx) & (np.abs(y) <= ry) & (np.abs(z) <= rz)
        if inside.any():
            mask[
                kk[inside].astype(np.int64),
                jj[inside].astype(np.int64),
                ii[inside].astype(np.int64),
            ] = True

        print(f"  Resulting mask voxel count: {int(mask.sum())}")
        return mask

    def _is_selection_roi_valid(self):
        """True iff the operation ROI node still exists in the MRML scene."""
        node = self._sel_op_roi_node
        return node is not None and slicer.mrmlScene.IsNodePresent(node)

    def _configure_selection_roi_display(self, display_node):
        """Style the operation ROI distinctly from the bbox prompt."""
        display_node.SetHandlesInteractive(True)
        # Blender/Godot-style gizmo: keep the three colored axis translation
        # arrows (drag an axis to move precisely along it) and the face scale
        # handles (resize), but drop the rotation rings -- the box is meant to
        # stay axis-aligned and rotation handles are easy to grab by mistake.
        # Each guarded so an older Slicer lacking a setter still works (it keeps
        # that handle at its default).
        for setter, value in (
            ("SetTranslationHandleVisibility", True),
            ("SetScaleHandleVisibility", True),
            ("SetRotationHandleVisibility", False),
            ("SetInteractionHandleScale", 3.0),
        ):
            try:
                getattr(display_node, setter)(value)
            except (AttributeError, TypeError):
                pass
        # Higher fill opacity so the region enclosed by the box is clearly
        # tinted in both 2D slices and 3D (was barely visible at 0.1).
        display_node.SetFillOpacity(0.45)
        display_node.SetOutlineOpacity(0.8)
        # Orange, to stand apart from the blue bbox prompt.
        color = (1.0, 0.55, 0.1)
        display_node.SetSelectedColor(*color)
        display_node.SetColor(*color)
        display_node.SetActiveColor(*color)
        display_node.SetSliceProjectionColor(*color)
        display_node.SetGlyphScale(0)
        display_node.SetTextScale(0)

    def _initialize_selection_roi_geometry(self, node):
        """Place a freshly created ROI at the volume center with half-extent radii."""
        volume_node = self.get_volume_node()
        if volume_node is None:
            return
        ras_bounds = [0.0] * 6
        volume_node.GetRASBounds(ras_bounds)
        center = [
            0.5 * (ras_bounds[0] + ras_bounds[1]),
            0.5 * (ras_bounds[2] + ras_bounds[3]),
            0.5 * (ras_bounds[4] + ras_bounds[5]),
        ]
        radius = [
            0.25 * max(1.0, ras_bounds[1] - ras_bounds[0]),
            0.25 * max(1.0, ras_bounds[3] - ras_bounds[2]),
            0.25 * max(1.0, ras_bounds[5] - ras_bounds[4]),
        ]
        node.SetCenter(center)
        node.SetRadiusXYZ(radius)

    def _get_or_create_selection_roi(self):
        """
        Ensure a vtkMRMLMarkupsROINode named "SelectionOpROI" exists with
        interactive handles. Returns the node.
        """
        name = "SelectionOpROI"
        node = self._sel_op_roi_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLMarkupsROINode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
                node.SetName(name)
                self._initialize_selection_roi_geometry(node)
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            if display_node is not None:
                self._configure_selection_roi_display(display_node)
            self._sel_op_roi_node = node

        node.SetDisplayVisibility(True)

        # Ensure shape preview tracks this ROI's pose and size.
        self._safe_add_observer(
            node, vtk.vtkCommand.ModifiedEvent, self._on_selection_roi_modified
        )
        self._get_or_create_selection_roi_preview()
        self._update_selection_roi_preview()
        return node

    def _destroy_selection_roi(self):
        """Remove the operation ROI node (and its preview) from the scene."""
        node = self._sel_op_roi_node
        if node is not None:
            self._safe_remove_observer(
                node, vtk.vtkCommand.ModifiedEvent, self._on_selection_roi_modified
            )
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._sel_op_roi_node = None
        self._destroy_selection_roi_preview()

    # -- Crop Segment by Box (dedicated ROI) --

    def _segment_ras_bounds(self):
        """World/RAS bounding box [xmin, xmax, ymin, ymax, zmin, zmax] of the
        current segment's non-zero voxels, or None when empty/unavailable.

        Used to size the crop ROI to the segment rather than the whole volume.
        Computed on the output grid (where the segment lives) and mapped to RAS
        via the output volume's IJKToRAS composed with any parent transform, so
        it is correct for oblique/registered series."""
        try:
            mask = self.get_segment_data()
        except Exception:  # noqa: BLE001 - reading the segment may fail
            mask = None
        if mask is None or not np.asarray(mask).any():
            return None
        output_volume = self.get_output_volume_node()
        if output_volume is None:
            return None
        idx = np.argwhere(np.asarray(mask) > 0)  # rows of (k, j, i)
        k_lo, j_lo, i_lo = idx.min(axis=0)
        k_hi, j_hi, i_hi = idx.max(axis=0)
        ijk_to_ras = vtk.vtkMatrix4x4()
        output_volume.GetIJKToRASMatrix(ijk_to_ras)
        tnode = output_volume.GetParentTransformNode()
        local_to_world = None
        if tnode is not None:
            local_to_world = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                tnode, None, local_to_world)
        xs, ys, zs = [], [], []
        for i in (int(i_lo), int(i_hi)):
            for j in (int(j_lo), int(j_hi)):
                for k in (int(k_lo), int(k_hi)):
                    ras = ijk_to_ras.MultiplyPoint(
                        [float(i), float(j), float(k), 1.0])
                    p = [ras[0], ras[1], ras[2]]
                    if local_to_world is not None:
                        p = list(local_to_world.TransformPoint(p))
                    xs.append(p[0])
                    ys.append(p[1])
                    zs.append(p[2])
        return [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]

    def _initialize_crop_roi_geometry(self, node):
        """Place a freshly created crop ROI around the current segment's RAS
        bounding box (so it defaults to framing the segment). Falls back to the
        volume-center geometry when the segment is empty/unavailable."""
        bounds = self._segment_ras_bounds()
        if bounds is None:
            self._initialize_selection_roi_geometry(node)
            return
        center = [0.5 * (bounds[0] + bounds[1]),
                  0.5 * (bounds[2] + bounds[3]),
                  0.5 * (bounds[4] + bounds[5])]
        # 1.05x half-extent so the box starts just outside the segment rather
        # than clipping its border voxels.
        radius = [0.5 * 1.05 * max(1.0, bounds[1] - bounds[0]),
                  0.5 * 1.05 * max(1.0, bounds[3] - bounds[2]),
                  0.5 * 1.05 * max(1.0, bounds[5] - bounds[4])]
        node.SetCenter(center)
        node.SetRadiusXYZ(radius)

    def _get_or_create_crop_roi(self):
        """Ensure a vtkMRMLMarkupsROINode named "CropSegmentROI" exists with
        interactive box handles, sized to the current segment. Independent from
        the Selection Operations operand ROI. Returns the node."""
        name = "CropSegmentROI"
        node = self._crop_roi_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLMarkupsROINode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsROINode")
                node.SetName(name)
                self._initialize_crop_roi_geometry(node)
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            if display_node is not None:
                self._configure_selection_roi_display(display_node)
            self._crop_roi_node = node
        node.SetDisplayVisibility(True)
        return node

    def _destroy_crop_roi(self):
        """Remove the crop ROI node from the scene."""
        node = self._crop_roi_node
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
        self._crop_roi_node = None

    # -- ROI shape preview (sphere / ellipsoid visualization) --

    def _get_or_create_selection_roi_preview(self):
        """
        Create (or recover) the hidden Model + LinearTransform nodes that
        visualize the actual sphere/ellipsoid acted upon by the boolean
        operation. The model carries a unit sphere mesh; the transform places
        and scales it to match the current ROI + cbRoiShape.
        """
        model_name = "SelectionOpROIPreview"
        transform_name = "SelectionOpROIPreviewTransform"

        transform_node = self._sel_op_roi_preview_transform_node
        if transform_node is None or not slicer.mrmlScene.IsNodePresent(transform_node):
            existing = slicer.mrmlScene.GetFirstNodeByName(transform_name)
            if existing is not None and existing.IsA("vtkMRMLLinearTransformNode"):
                transform_node = existing
            else:
                transform_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLinearTransformNode"
                )
                transform_node.SetName(transform_name)
            transform_node.HideFromEditorsOn()
            self._sel_op_roi_preview_transform_node = transform_node

        model_node = self._sel_op_roi_preview_node
        if model_node is None or not slicer.mrmlScene.IsNodePresent(model_node):
            existing = slicer.mrmlScene.GetFirstNodeByName(model_name)
            if existing is not None and existing.IsA("vtkMRMLModelNode"):
                model_node = existing
            else:
                model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
                model_node.SetName(model_name)

            # Unit sphere; transform handles all scale/rotation/translation.
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)
            sphere.SetThetaResolution(24)
            sphere.SetPhiResolution(24)
            sphere.Update()
            model_node.SetAndObservePolyData(sphere.GetOutput())

            model_node.HideFromEditorsOn()
            model_node.CreateDefaultDisplayNodes()
            display_node = model_node.GetDisplayNode()
            if display_node is not None:
                color = (1.0, 0.55, 0.1)
                display_node.SetColor(*color)
                display_node.SetEdgeColor(*color)
                display_node.SetOpacity(0.25)
                display_node.SetSliceIntersectionVisibility(True)
                display_node.SetSliceIntersectionThickness(2)
                display_node.SetVisibility2D(True)
                display_node.SetVisibility(True)
            self._sel_op_roi_preview_node = model_node

        # (Re)attach model to transform.
        model_node.SetAndObserveTransformNodeID(transform_node.GetID())
        return model_node, transform_node

    def _update_selection_roi_preview(self):
        """
        Recompute the preview transform (and visibility) from the current ROI
        + cbRoiShape selection. Safe to call even when the ROI is absent.
        """
        if not self._is_selection_roi_valid():
            self._set_preview_visible(False)
            return

        shape_idx = self.ui.cbRoiShape.currentIndex  # 0=Box, 1=Sphere, 2=Ellipsoid
        if shape_idx == ROI_SHAPE_BOX:
            self._set_preview_visible(False)
            return

        radius = [0.0, 0.0, 0.0]
        self._sel_op_roi_node.GetRadiusXYZ(radius)
        rx, ry, rz = radius
        if min(rx, ry, rz) <= 0.0:
            self._set_preview_visible(False)
            return

        if shape_idx == ROI_SHAPE_SPHERE:
            r = min(rx, ry, rz)
            sx = sy = sz = r
        else:
            sx, sy, sz = rx, ry, rz

        # World = ObjectToWorld @ Scale.
        o2w_vtk = self._sel_op_roi_node.GetObjectToWorldMatrix()
        scaled = vtk.vtkMatrix4x4()
        scaled.DeepCopy(o2w_vtk)
        # Multiply each column of the 3x3 part by its scale.
        scale = (sx, sy, sz)
        for col in range(3):
            for row in range(3):
                scaled.SetElement(
                    row, col, o2w_vtk.GetElement(row, col) * scale[col]
                )

        model_node, transform_node = self._get_or_create_selection_roi_preview()
        transform_node.SetMatrixTransformToParent(scaled)
        self._set_preview_visible(True)

    def _set_preview_visible(self, visible):
        """Toggle visibility of the preview model node (no-op if absent)."""
        model_node = self._sel_op_roi_preview_node
        if model_node is None or not slicer.mrmlScene.IsNodePresent(model_node):
            return
        display_node = model_node.GetDisplayNode()
        if display_node is None:
            return
        display_node.SetVisibility(bool(visible))
        display_node.SetVisibility2D(bool(visible))

    def _destroy_selection_roi_preview(self):
        """Remove the preview model + transform nodes from the scene."""
        for attr in ("_sel_op_roi_preview_node", "_sel_op_roi_preview_transform_node"):
            node = getattr(self, attr, None)
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
            setattr(self, attr, None)

    def _on_selection_roi_modified(self, caller, event):
        """ROI moved/resized/rotated -- keep preview transform in sync."""
        self._update_selection_roi_preview()

    def on_place_roi_clicked(self, checked=False):
        """Create or show the operation ROI in the 3D view."""
        self._get_or_create_selection_roi()
        self._refresh_apply_enabled()
        slicer.util.showStatusMessage(
            "Drag the ROI handles in the 3D view, then click Apply Operation.",
            5000,
        )

    def on_clear_roi_clicked(self, checked=False):
        """Remove the operation ROI."""
        self._destroy_selection_roi()
        self._refresh_apply_enabled()

    def on_place_crop_roi_clicked(self, checked=False):
        """Create or show the crop ROI, sized to the current segment."""
        _seg_node, seg_id = self.get_selected_segmentation_node_and_segment_id()
        if not seg_id:
            slicer.util.showStatusMessage(
                "Select a segment to crop first.", 3000)
            return
        self._get_or_create_crop_roi()
        slicer.util.showStatusMessage(
            "Rotate the 3D view and drag the box to frame the region to KEEP, "
            "then click 'Crop Segment by Box'.", 6000)

    def on_crop_segment_by_box_clicked(self, checked=False):
        """Clear the current segment's voxels lying OUTSIDE the crop box (keep
        the intersection of the segment AND the box). Local edit; undoable via
        the Selection Operations Undo; synced to the server."""
        if self._crop_roi_node is None or not slicer.mrmlScene.IsNodePresent(
                self._crop_roi_node):
            slicer.util.showStatusMessage(
                "Place the crop box first (Place Crop ROI).", 4000)
            return
        _seg_node, seg_id = self.get_selected_segmentation_node_and_segment_id()
        if not seg_id:
            slicer.util.showStatusMessage(
                "Select a segment to crop first.", 3000)
            return
        try:
            # Current segment on the canonical output grid.
            current = self.get_segment_data()
            if current is None:
                slicer.util.showStatusMessage(
                    "Could not read the current segment; crop aborted.", 4000)
                return
            current = current.astype(np.uint8)
            if current.sum() == 0:
                slicer.util.showStatusMessage(
                    "The current segment is empty; nothing to crop.", 3000)
                return
            # Box mask on the source grid, then bridge to the output grid (same
            # path as apply_boolean_operation, so the shapes always match even
            # under high-resolution output / tri-planar mode).
            box_src = self.roi_node_to_mask(
                self._crop_roi_node, shape_idx=ROI_SHAPE_BOX)
            box_out = self._to_output_grid(box_src.astype(np.uint8))
            if box_out is None:
                self._disable_high_res_output(
                    "High-resolution resample failed; reverted to the source "
                    "grid.")
                box_out = box_src.astype(np.uint8)
            if tuple(box_out.shape) != tuple(current.shape):
                slicer.util.showStatusMessage(
                    "Crop box grid mismatch; crop aborted.", 4000)
                return
            cropped = (current.astype(bool) & box_out.astype(bool)).astype(
                np.uint8)
            if int(cropped.sum()) == int(current.sum()):
                slicer.util.showStatusMessage(
                    "The box already contains the whole segment; nothing "
                    "removed.", 3000)
                return
            if int(cropped.sum()) == 0:
                slicer.util.showStatusMessage(
                    "The box does not overlap the segment; the segment was "
                    "cleared (use Undo to revert).", 5000)
            # Snapshot for our own Undo, write back, refresh 3D, sync.
            self._record_selection_op_undo(seg_id, current.copy())
            self.show_segmentation(cropped)
            # setup_prompts rebuilds the hidden scribble editor (which resets
            # slice backgrounds); it schedules a sticky per-plane reapply.
            self.setup_prompts()
            if not self.ui.cbKeepCropRoi.isChecked():
                self._destroy_crop_roi()
            sync_result = self.upload_segment_to_server()
            if sync_result is None:
                slicer.util.showStatusMessage(
                    "Cropped locally, but syncing to the server failed (use "
                    "'Sync to server').", 5000)
            else:
                slicer.util.showStatusMessage("Segment cropped to box.", 3000)
        except Exception as exc:  # noqa: BLE001 - report and keep the UI alive
            print("[DEBUG crop] crop failed: {}".format(exc))
            slicer.util.showStatusMessage(
                "Crop failed: {}".format(exc), 5000)

    def _on_roi_shape_changed(self, index):
        """Status-bar hint clarifying what each ROI shape means, plus preview refresh."""
        if index == ROI_SHAPE_SPHERE:
            slicer.util.showStatusMessage(
                "Sphere mode uses the inscribed sphere "
                "(radius = min of the ROI's three half-extents).",
                5000,
            )
        elif index == ROI_SHAPE_ELLIPSOID:
            slicer.util.showStatusMessage(
                "Ellipsoid mode uses the ellipsoid aligned with the ROI axes.",
                5000,
            )
        self._update_selection_roi_preview()

    def _refresh_apply_enabled(self):
        """Enable Apply only when the current operand source has a usable operand."""
        # Source order: 0=ROI box, 1=Magic wand, 2=Segment, 3=Lasso (3D).
        source = self.ui.cbOperandSource.currentIndex
        if source == OPERAND_SOURCE_ROI:
            enabled = self._is_selection_roi_valid()
        elif source == OPERAND_SOURCE_WAND:
            enabled = self._is_selection_wand_seed_valid()
        elif source == OPERAND_SOURCE_LASSO3D:
            enabled = self._is_selection_lasso3d_valid()
        else:
            enabled = self.ui.cbSelectionOperand.count > 0
        self.ui.pbApplySelectionOp.setEnabled(enabled)

    def _on_operand_source_changed(self, index):
        """
        Toggle the operand rows and clean up after the modes we are leaving so a
        ROI / Magic wand / Lasso preview never lingers across switches.
        Source order: 0=ROI box, 1=Magic wand, 2=Segment, 3=Lasso (3D).
        """
        self.ui.operandRoiContainer.setVisible(index == OPERAND_SOURCE_ROI)
        self.ui.operandMagicWandContainer.setVisible(index == OPERAND_SOURCE_WAND)
        self.ui.operandSegmentContainer.setVisible(index == OPERAND_SOURCE_SEGMENT)
        self.ui.operandLasso3dContainer.setVisible(index == OPERAND_SOURCE_LASSO3D)
        # Leaving ROI -> destroy the ROI box (and its preview).
        if index != OPERAND_SOURCE_ROI:
            self._destroy_selection_roi()
        # Leaving Magic wand -> destroy seeds and the mask preview.
        if index != OPERAND_SOURCE_WAND:
            self._destroy_wand_seed()
            self._clear_wand_preview_segment()
        # Leaving Lasso (3D) -> disarm Scissors and clear region + preview.
        if index != OPERAND_SOURCE_LASSO3D:
            self.ui.pbDrawLasso3d.setChecked(False)
            self._deactivate_lasso3d_scissors()
            self._clear_lasso3d_input_segment()
            self._clear_lasso3d_preview_segment()
        self._refresh_apply_enabled()

    # -- Magic wand seed lifecycle and flood fill --

    def _configure_wand_seed_display(self, display_node):
        """Distinct green so the wand seed is not confused with bbox/ROI."""
        color = (0.2, 0.85, 0.4)
        display_node.SetColor(*color)
        display_node.SetSelectedColor(*color)
        display_node.SetActiveColor(*color)
        display_node.SetGlyphScale(0.9)
        display_node.SetTextScale(0)
        display_node.SetSliceProjection(True)
        display_node.SetSliceProjectionColor(*color)

    def _get_or_create_wand_seed(self):
        """
        Ensure a vtkMRMLMarkupsFiducialNode exists for magic wand seeds. The
        node holds an unlimited list of seeds (each click adds another).
        Returns the node.
        """
        name = "SelectionOpWandSeeds"
        node = self._sel_op_wand_seed_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLMarkupsFiducialNode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode"
                )
                node.SetName(name)
            node.SetMaximumNumberOfControlPoints(-1)
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            if display_node is not None:
                self._configure_wand_seed_display(display_node)
            self._safe_add_observer(
                node,
                slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                self._on_wand_seed_placed,
            )
            self._sel_op_wand_seed_node = node

        node.SetDisplayVisibility(True)
        return node

    def _destroy_wand_seed(self):
        """
        Remove the current wand seed node AND any historically-named orphans
        from the scene, detach the placement observer, and bail out of Place
        mode if we were the active placer.
        """
        # 1) Detach observer on the tracked node before removal so the placement
        #    callback never fires on a half-dead node.
        tracked = self._sel_op_wand_seed_node
        self._safe_remove_observer(
            tracked,
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self._on_wand_seed_placed,
        )

        # 2) Sweep every historical wand seed name out of the scene. Use a
        #    while-loop in case multiple nodes share the same name.
        for name in self._WAND_SEED_NODE_NAMES:
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            while existing is not None:
                if existing.IsA("vtkMRMLMarkupsFiducialNode"):
                    slicer.mrmlScene.RemoveNode(existing)
                else:
                    break
                existing = slicer.mrmlScene.GetFirstNodeByName(name)
        self._sel_op_wand_seed_node = None

        # 3) If we left interaction mode in Place for a fiducial, bail out so
        #    the cursor stops behaving like "about to drop a point".
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        selection_node = slicer.app.applicationLogic().GetSelectionNode()
        if (
            interaction_node.GetCurrentInteractionMode()
            == slicer.vtkMRMLInteractionNode.Place
            and selection_node.GetActivePlaceNodeClassName()
            == "vtkMRMLMarkupsFiducialNode"
        ):
            interaction_node.SetCurrentInteractionMode(
                interaction_node.ViewTransform
            )

    def _is_selection_wand_seed_valid(self):
        """True iff at least one wand seed has been placed."""
        node = self._sel_op_wand_seed_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            return False
        return node.GetNumberOfControlPoints() >= 1

    def _collect_wand_seeds(self):
        """
        Collect placed wand seeds, converting RAS positions to (k, j, i) voxel
        coords. Returns a list of ((k, j, i), True) tuples (all seeds are
        positive prompts) or [].
        """
        inference_volume = self.get_inference_volume_node()
        arr = self.get_inference_image_data()
        if arr is None:
            return []
        node = self._sel_op_wand_seed_node
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            return []
        out = []
        for idx in range(node.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(idx, ras)
            ijk = self.ras_to_xyz(list(ras), volume_node=inference_volume)
            ii, jj, kk = int(ijk[0]), int(ijk[1]), int(ijk[2])
            if (
                0 <= ii < arr.shape[2]
                and 0 <= jj < arr.shape[1]
                and 0 <= kk < arr.shape[0]
            ):
                out.append(((kk, jj, ii), True))
        return out

    def _postprocess_wand_mask(self, mask):
        """Apply Grow/Shrink post-processing to the AI mask."""
        if mask is None or not mask.any():
            return mask
        n_iter = int(self.ui.sbGrowShrinkWand.value)
        if n_iter == 0:
            return mask.astype(bool)
        try:
            from scipy import ndimage
        except Exception:
            return mask.astype(bool)
        if n_iter > 0:
            mask = ndimage.binary_dilation(mask, iterations=n_iter)
        else:
            mask = ndimage.binary_erosion(mask, iterations=-n_iter)
        return mask.astype(bool)

    def _wand_raw_source_mask(self):
        """One nnInteractive wand pass on the CURRENT inference volume (honors
        _active_inference_volume_override), resampled to the source grid. No
        Grow/Shrink (the caller applies it once). Backs up + restores the server
        segment around the seed sweep. Returns a bool mask or None.

        The call cycle: back up the target segment -> POST empty mask to reset
        server interactions -> POST /add_point_interaction per seed (the last
        response holds the cumulative mask) -> restore the target segment.
        voxel_coord uses (z, y, x) order (== ras_to_xyz()[::-1])."""
        volume = self.get_inference_volume_node()
        arr = self.get_inference_image_data()
        if volume is None or arr is None or not self.server:
            return None
        if not self._ensure_inference_image_uploaded():
            return None

        seeds = self._collect_wand_seeds()
        if not seeds:
            return None

        pre_target = self.get_segment_data(
            reference_volume_node=volume
        ).astype(np.uint8).copy()
        shape = arr.shape

        empty = np.zeros(shape, dtype=np.uint8)
        reset_resp = self.request_to_server(
            f"{self.server}/upload_segment",
            files=self.mask_to_np_upload_file(empty),
            headers={"Content-Encoding": "gzip"},
        )
        if reset_resp is None:
            return None

        seg_mask = None
        try:
            last_response = None
            for (kk, jj, ii), is_pos in seeds:
                last_response = self.request_to_server(
                    f"{self.server}/add_point_interaction",
                    json={
                        "voxel_coord": [kk, jj, ii],
                        "positive_click": bool(is_pos),
                    },
                )
                if last_response is None:
                    break
            if last_response is not None:
                seg_mask = self.unpack_binary_segmentation(
                    last_response.content, decompress=False
                ).astype(bool)
                seg_mask = self._resample_mask_between_volumes(
                    seg_mask, volume, self.get_volume_node()
                ).astype(bool)
        finally:
            # Restore the user's target segment on the server so subsequent
            # nnInteractive prompts continue from where they left off.
            restore_resp = self.request_to_server(
                f"{self.server}/upload_segment",
                files=self.mask_to_np_upload_file(pre_target),
                headers={"Content-Encoding": "gzip"},
            )
            if restore_resp is None:
                slicer.util.showStatusMessage(
                    "Magic wand could not restore server state; "
                    "the next prompt may resync automatically.",
                    4000,
                )

        return seg_mask

    def _compute_magic_wand_mask(self, seed_node=None):
        """Compute the magic wand operand mask on the SOURCE grid.

        In tri-planar mode the AI point prompt is run once per assigned series
        (R/Y/G) and the per-series results are unioned, so all three series
        contribute to the grown region; otherwise it runs once on the current
        inference volume (original single-series behavior). Grow/Shrink is
        applied once at the end. `seed_node` is ignored (kept for back-compat).
        Returns a bool numpy mask on the source grid, or None on failure."""
        series_list = (
            self._triplanar_coverage_volumes() if self._triplanar_mode else []
        )
        if len(series_list) < 2:
            mask = self._wand_raw_source_mask()
            return None if mask is None else self._postprocess_wand_mask(mask)

        saved_override = self._active_inference_volume_override
        masks = []
        try:
            for series in series_list:
                self._active_inference_volume_override = series
                mask = self._wand_raw_source_mask()
                print("[DEBUG wand.triplanar] series='{}' sum={}".format(
                    series.GetName(),
                    None if mask is None else int(mask.sum())))
                if mask is not None:
                    masks.append(mask)
        finally:
            # The per-series sweep left the server on the last series' image;
            # restore the real inference image + segment so later prompts work.
            self._active_inference_volume_override = saved_override
            self._ensure_inference_image_uploaded()
            self.upload_segment_to_server()

        if not masks:
            return None
        fused = masks[0]
        for mask in masks[1:]:
            fused = np.logical_or(fused, mask)
        fused = self._postprocess_wand_mask(fused.astype(bool))
        print("[DEBUG wand.triplanar] fused sum={}".format(
            None if fused is None else int(fused.sum())))
        return fused

    def _enter_place_mode_for_wand(self, node, status_msg):
        selection_node = slicer.app.applicationLogic().GetSelectionNode()
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        selection_node.SetReferenceActivePlaceNodeClassName(
            "vtkMRMLMarkupsFiducialNode"
        )
        selection_node.SetActivePlaceNodeID(node.GetID())
        interaction_node.SetPlaceModePersistence(0)
        interaction_node.SetCurrentInteractionMode(interaction_node.Place)
        slicer.util.showStatusMessage(status_msg, 5000)

    def on_place_wand_seed_clicked(self, checked=False):
        """Add another magic wand seed (does NOT clear earlier seeds)."""
        node = self._get_or_create_wand_seed()
        self._enter_place_mode_for_wand(
            node,
            "Click on the 3D model surface or in any 2D view to add a magic "
            "wand seed.",
        )
        self._refresh_apply_enabled()

    def on_clear_wand_seed_clicked(self, checked=False):
        """Remove all magic wand seeds (positive and negative) and the preview."""
        self._destroy_wand_seed()
        self._clear_wand_preview_segment()
        self._refresh_apply_enabled()

    def on_clear_preview_wand_clicked(self, checked=False):
        """Hide the magic wand preview overlay; seeds are kept."""
        self._clear_wand_preview_segment()

    def _on_wand_seed_placed(self, caller, event):
        """Seed was placed -- refresh Apply state only (preview is manual)."""
        self._refresh_apply_enabled()

    # -- Magic wand live preview --

    def _get_or_create_wand_preview_segmentation(self):
        """
        Create (or recover) a hidden segmentation node with a single 'preview'
        segment used to visualize the magic wand region before Apply.
        """
        return self._get_or_create_hidden_segmentation(
            "_sel_op_wand_preview_segment_node",
            self.wand_preview_segment_node_name,
            "_wand_preview_segment_id",
            "MagicWandPreview",
            [0.95, 0.2, 0.85],
        )

    def _clear_segment_labelmap(self, node, seg_id):
        """Zero-fill a segment's labelmap on the source grid and hide it.

        The node itself is kept around so the next show is cheap.
        """
        if node is None or not slicer.mrmlScene.IsNodePresent(node) or not seg_id:
            return
        volume_node = self.get_volume_node()
        image = self.get_image_data()
        if volume_node is None or image is None:
            return
        empty = np.zeros(image.shape, dtype=np.uint8)
        try:
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                empty, node, seg_id, volume_node
            )
        except Exception:
            pass
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentVisibility(seg_id, False)

    def _clear_wand_preview_segment(self):
        """
        Empty the preview segment's labelmap and hide it. The node itself is
        kept around so the next show is cheap.
        """
        self._clear_segment_labelmap(
            self._sel_op_wand_preview_segment_node, self._wand_preview_segment_id
        )

    def _destroy_wand_preview_segmentation(self):
        """Remove the hidden wand preview segmentation node from the scene."""
        node = self._sel_op_wand_preview_segment_node
        if node is None:
            node = slicer.mrmlScene.GetFirstNodeByName(
                getattr(
                    self,
                    "wand_preview_segment_node_name",
                    "MagicWandPreviewSegmentNode (do not touch)",
                )
            )
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
        self._sel_op_wand_preview_segment_node = None
        self._wand_preview_segment_id = None

    def _update_magic_wand_preview(self):
        """
        Recompute the wand mask via nnInteractive from the current seed and
        write it into the preview segment. Safe to call when the wand is not
        the active operand source -- it will just clear the preview.
        """
        if self.ui.cbOperandSource.currentIndex != OPERAND_SOURCE_WAND:
            self._clear_wand_preview_segment()
            return
        if not self._is_selection_wand_seed_valid():
            self._clear_wand_preview_segment()
            return

        wand_mask = self._compute_magic_wand_mask()
        if wand_mask is None:
            self._clear_wand_preview_segment()
            return

        node, seg_id = self._get_or_create_wand_preview_segmentation()
        volume_node = self.get_volume_node()
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            wand_mask.astype(np.uint8), node, seg_id, volume_node
        )
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentVisibility(seg_id, True)
        if int(wand_mask.sum()) > 0:
            node.GetSegmentation().CollapseBinaryLabelmaps()
            node.CreateClosedSurfaceRepresentation()

    def on_preview_wand_clicked(self, checked=False):
        """Run a one-shot AI wand call and write the result into the preview."""
        if not self._is_selection_wand_seed_valid():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "Place a seed before previewing.",
            )
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._update_magic_wand_preview()
        finally:
            qt.QApplication.restoreOverrideCursor()

    # -- Lasso (3D): native Scissors draws a region, nnInteractive lasso AI --
    #    refines it into a preview, mirroring the Magic wand seed/preview flow.

    def _get_or_create_hidden_segmentation(
        self, attr, name, seg_id_attr, seg_name, color
    ):
        """
        Shared helper to create (or recover) a hidden single-segment node on the
        source grid (Scissors input/preview, magic wand preview, etc.). The node
        is hidden from editors and shown as a translucent fill/outline overlay.
        """
        node = getattr(self, attr)
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing is not None and existing.IsA("vtkMRMLSegmentationNode"):
                node = existing
            else:
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode"
                )
                node.SetName(name)
            node.HideFromEditorsOn()
            volume_node = self.get_volume_node()
            if volume_node is not None:
                node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
            node.CreateDefaultDisplayNodes()
            setattr(self, attr, node)

        segmentation = node.GetSegmentation()
        seg_id = getattr(self, seg_id_attr)
        if not seg_id or not segmentation.GetSegment(seg_id):
            seg_id = segmentation.AddEmptySegment(seg_name, seg_name, color)
            setattr(self, seg_id_attr, seg_id)

        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentOpacity2DFill(seg_id, 0.35)
            display_node.SetSegmentOpacity2DOutline(seg_id, 0.9)
            display_node.SetSegmentVisibility(seg_id, True)

        return node, seg_id

    def _get_or_create_lasso3d_input_segmentation(self):
        """Hidden node holding the raw 3D region the user draws with Scissors."""
        return self._get_or_create_hidden_segmentation(
            "_sel_op_lasso3d_input_segment_node",
            self.lasso3d_input_segment_node_name,
            "_lasso3d_input_segment_id",
            "Lasso3dInput",
            [0.95, 0.85, 0.2],
        )

    def _get_or_create_lasso3d_preview_segmentation(self):
        """Hidden node holding the nnInteractive lasso AI result (the preview)."""
        return self._get_or_create_hidden_segmentation(
            "_sel_op_lasso3d_preview_segment_node",
            self.lasso3d_preview_segment_node_name,
            "_lasso3d_preview_segment_id",
            "Lasso3dPreview",
            [0.2, 0.85, 0.95],
        )

    def _setup_lasso3d_editor(self):
        """
        Lazily create a background (headless) Segment Editor used to drive the
        native Scissors effect on the lasso input segment. Mirrors the hidden
        scribble editor.
        """
        if self._lasso3d_editor_widget is not None:
            return
        import qSlicerSegmentationsModuleWidgetsPythonQt

        self._lasso3d_editor_widget = (
            qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        )
        self._lasso3d_editor_widget.setMRMLScene(slicer.mrmlScene)
        self._lasso3d_editor_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode"
        )
        self._lasso3d_editor_widget.setMRMLSegmentEditorNode(self._lasso3d_editor_node)

    def _set_lasso3d_input_visible(self, visible):
        """Show/hide the drawn input region (hidden while the AI preview shows)."""
        node = self._sel_op_lasso3d_input_segment_node
        seg_id = self._lasso3d_input_segment_id
        if node is None or not seg_id:
            return
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentVisibility(seg_id, visible)

    def _activate_lasso3d_scissors(self):
        """
        Bind the hidden editor to the lasso INPUT segment and turn on the native
        Scissors effect, configured for a free-form 3D-view lasso that fills the
        region extruded along the camera direction.
        """
        volume_node = self.get_volume_node()
        if volume_node is None:
            return
        node, seg_id = self._get_or_create_lasso3d_input_segmentation()
        self._set_lasso3d_input_visible(True)
        self._setup_lasso3d_editor()

        self._lasso3d_editor_widget.setSegmentationNode(node)
        self._lasso3d_editor_widget.setSourceVolumeNode(volume_node)
        # setSourceVolumeNode resets every slice background to the source volume,
        # so restore the sticky per-plane display selections afterward.
        self._schedule_plane_display_reapply()
        self._lasso3d_editor_node.SetSelectedSegmentID(seg_id)

        self._lasso3d_editor_widget.setActiveEffectByName("Scissors")
        self._lasso3d_editor_widget.updateWidgetFromMRML()
        effect = self._lasso3d_editor_widget.activeEffect()
        if effect is not None:
            effect.setParameter("Operation", "FillInside")
            effect.setParameter("Shape", "FreeForm")
            effect.setParameter("SliceCutMode", "Unlimited")

        # Watch the input node so Apply enables and the 3D surface refreshes as
        # soon as a region is carved. Use AnyEvent (not ModifiedEvent): Scissors
        # edits the segment labelmap through events that do not fire the node's
        # plain ModifiedEvent, the same reason the scribble Paint observer uses
        # AnyEvent.
        if self._lasso3d_input_observer_tag is None:
            self._lasso3d_input_observer_tag = node.AddObserver(
                vtk.vtkCommand.AnyEvent, self._on_lasso3d_input_modified
            )

        slicer.util.showStatusMessage(
            "Lasso (3D): drag a closed loop in the 3D view, then click Preview. "
            "Enable volume rendering or show a segment surface to aim at.",
            5000,
        )

    def _deactivate_lasso3d_scissors(self):
        """Release the Scissors view interactions and stop watching the input."""
        if self._lasso3d_editor_widget is not None:
            self._lasso3d_editor_widget.setActiveEffectByName("")
        node = self._sel_op_lasso3d_input_segment_node
        if (
            self._lasso3d_input_observer_tag is not None
            and node is not None
            and slicer.mrmlScene.IsNodePresent(node)
        ):
            node.RemoveObserver(self._lasso3d_input_observer_tag)
        self._lasso3d_input_observer_tag = None

    def _on_lasso3d_input_modified(self, caller, event):
        """
        React to Scissors carving the input region: enable Apply and rebuild the
        3D closed surface so the drawn region shows solid in the 3D view.
        """
        if self._lasso3d_in_update:
            return
        self._lasso3d_in_update = True
        try:
            self._refresh_apply_enabled()
            node = self._sel_op_lasso3d_input_segment_node
            seg_id = self._lasso3d_input_segment_id
            if node is None or not seg_id:
                return
            volume_node = self.get_volume_node()
            if volume_node is None:
                return
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(
                node, seg_id, volume_node
            )
            if mask is not None and int(mask.sum()) > 0:
                node.GetSegmentation().CollapseBinaryLabelmaps()
                node.CreateClosedSurfaceRepresentation()
        finally:
            self._lasso3d_in_update = False

    def on_draw_lasso3d_clicked(self, checked=False):
        """Toggle the native Scissors lasso on the hidden input segment."""
        # Drawing uses a Segment Editor effect, not a markup placement, so make
        # sure no markup placement / scribble is competing for view clicks.
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)
        self.ui.pbInteractionScribble.setChecked(False)

        if not checked:
            self._deactivate_lasso3d_scissors()
            return

        if self.get_volume_node() is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "Load a volume before drawing a lasso region.",
            )
            self.ui.pbDrawLasso3d.setChecked(False)
            return

        self._activate_lasso3d_scissors()

    def on_clear_lasso3d_clicked(self, checked=False):
        """Clear the drawn input region and any AI preview."""
        self._clear_lasso3d_input_segment()
        self._clear_lasso3d_preview_segment()
        self._refresh_apply_enabled()

    def _clear_lasso3d_segment(self, node, seg_id):
        """Empty a lasso labelmap and hide its segment; keep the node.

        Guarded so the programmatic clear does not re-trigger the input
        observer (_on_lasso3d_input_modified).
        """
        self._lasso3d_in_update = True
        try:
            self._clear_segment_labelmap(node, seg_id)
        finally:
            self._lasso3d_in_update = False

    def _clear_lasso3d_input_segment(self):
        """Empty the drawn input region and hide it; keep the node."""
        self._clear_lasso3d_segment(
            self._sel_op_lasso3d_input_segment_node, self._lasso3d_input_segment_id
        )

    def _clear_lasso3d_preview_segment(self):
        """Empty the AI preview labelmap and hide it; keep the node."""
        self._clear_lasso3d_segment(
            self._sel_op_lasso3d_preview_segment_node,
            self._lasso3d_preview_segment_id,
        )

    def _lasso3d_input_to_mask(self):
        """Return the drawn input region as a uint8 mask aligned to the volume."""
        node = self._sel_op_lasso3d_input_segment_node
        seg_id = self._lasso3d_input_segment_id
        if node is None or not slicer.mrmlScene.IsNodePresent(node) or not seg_id:
            return None
        volume_node = self.get_volume_node()
        if volume_node is None:
            return None
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(node, seg_id, volume_node)
        if mask is None:
            return None
        return mask.astype(np.uint8)

    def _compute_lasso3d_mask(self):
        """
        Send the drawn 3D region to nnInteractive as a positive lasso prompt and
        return the AI segmentation, without disturbing the user's target session.
        Mirrors _compute_magic_wand_mask: back up target, reset, send, restore.
        Returns a bool numpy mask aligned to the volume, or None on failure.
        """
        source_volume = self.get_volume_node()
        volume = self.get_inference_volume_node()
        arr = self.get_inference_image_data()
        if volume is None or arr is None or not self.server:
            return None
        if not self._ensure_inference_image_uploaded():
            return None

        region = self._lasso3d_input_to_mask()
        if region is None or int(region.sum()) == 0:
            return None
        region = self._resample_mask_between_volumes(
            region, source_volume, volume
        )

        pre_target = self.get_segment_data(
            reference_volume_node=volume
        ).astype(np.uint8).copy()
        shape = arr.shape

        # 1) Reset server interactions by uploading an empty mask.
        empty = np.zeros(shape, dtype=np.uint8)
        reset_resp = self.request_to_server(
            f"{self.server}/upload_segment",
            files=self.mask_to_np_upload_file(empty),
            headers={"Content-Encoding": "gzip"},
        )
        if reset_resp is None:
            return None

        seg_mask = None
        try:
            # 2) Send the drawn region as a positive lasso interaction.
            buffer = io.BytesIO()
            np.save(buffer, region.astype(np.uint8))
            compressed_data = gzip.compress(buffer.getvalue())

            from requests_toolbelt import MultipartEncoder

            encoder = MultipartEncoder(
                fields={
                    "file": (
                        "volume.npy.gz",
                        compressed_data,
                        "application/octet-stream",
                    ),
                    "positive_click": "True",
                }
            )
            resp = self.request_to_server(
                f"{self.server}/add_lasso_interaction",
                data=encoder,
                headers={
                    "Content-Type": encoder.content_type,
                    "Content-Encoding": "gzip",
                },
            )
            if resp is not None and resp.status_code == 200:
                seg_mask = self.unpack_binary_segmentation(
                    resp.content, decompress=False
                ).astype(bool)
                seg_mask = self._resample_mask_between_volumes(
                    seg_mask, volume, source_volume
                ).astype(bool)
        finally:
            # 3) Restore the user's target segment on the server so subsequent
            #    nnInteractive prompts continue from where they left off.
            restore_resp = self.request_to_server(
                f"{self.server}/upload_segment",
                files=self.mask_to_np_upload_file(pre_target),
                headers={"Content-Encoding": "gzip"},
            )
            if restore_resp is None:
                slicer.util.showStatusMessage(
                    "Lasso (3D) could not restore server state; "
                    "the next prompt may resync automatically.",
                    4000,
                )

        return seg_mask

    def _update_lasso3d_preview(self):
        """
        Run the lasso AI on the drawn region and write the result into the
        preview segment. Safe to call when lasso is not the active operand
        source -- it will just clear the preview. Mirrors the magic wand.
        """
        if self.ui.cbOperandSource.currentIndex != OPERAND_SOURCE_LASSO3D:
            self._clear_lasso3d_preview_segment()
            return
        if not self._is_selection_lasso3d_valid():
            self._clear_lasso3d_preview_segment()
            return

        ai_mask = self._compute_lasso3d_mask()
        if ai_mask is None:
            self._clear_lasso3d_preview_segment()
            return

        node, seg_id = self._get_or_create_lasso3d_preview_segmentation()
        volume_node = self.get_volume_node()
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            ai_mask.astype(np.uint8), node, seg_id, volume_node
        )
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSegmentVisibility(seg_id, True)
        if int(ai_mask.sum()) > 0:
            node.GetSegmentation().CollapseBinaryLabelmaps()
            node.CreateClosedSurfaceRepresentation()
        self._schedule_plane_display_reapply()
        # Hide the raw drawn region so it does not occlude the AI preview in 3D.
        self._set_lasso3d_input_visible(False)
        # Belt-and-suspenders: a successful preview means a region was drawn, so
        # Apply must be clickable even if the draw-time observer never fired.
        self._refresh_apply_enabled()

    def on_preview_lasso3d_clicked(self, checked=False):
        """Run a one-shot lasso AI call and write the result into the preview."""
        if not self._is_selection_lasso3d_valid():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "Draw a lasso region in the 3D view before previewing.",
            )
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._update_lasso3d_preview()
        finally:
            qt.QApplication.restoreOverrideCursor()

    def on_clear_preview_lasso3d_clicked(self, checked=False):
        """Hide the AI preview overlay; the drawn region is kept and re-shown."""
        self._clear_lasso3d_preview_segment()
        self._set_lasso3d_input_visible(True)

    def _destroy_lasso3d(self):
        """Remove the lasso input/preview nodes and hidden editor (cleanup)."""
        self._deactivate_lasso3d_scissors()
        for attr in (
            "_sel_op_lasso3d_input_segment_node",
            "_sel_op_lasso3d_preview_segment_node",
            "_lasso3d_editor_node",
        ):
            node = getattr(self, attr)
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
            setattr(self, attr, None)
        self._lasso3d_input_segment_id = None
        self._lasso3d_preview_segment_id = None
        self._lasso3d_editor_widget = None

    def _is_selection_lasso3d_valid(self):
        """True when a non-empty input region has been drawn (Apply runs the AI)."""
        region = self._lasso3d_input_to_mask()
        return region is not None and int(region.sum()) > 0

    def _record_selection_op_undo(self, segment_id, pre_state_uint8):
        """Push a (segment_id, packed mask, shape) snapshot onto our private
        undo stack. Bit-packing keeps the stack's memory bounded on large
        (high-resolution) output grids (8x smaller than raw uint8)."""
        arr = np.asarray(pre_state_uint8)
        self._sel_op_undo_stack.append(
            (segment_id, np.packbits(arr.astype(bool)), arr.shape)
        )
        while len(self._sel_op_undo_stack) > self._sel_op_undo_stack_limit:
            self._sel_op_undo_stack.pop(0)

    def _clear_selection_op_undo_stack(self, reason=None):
        """Drop all Selection Operations undo snapshots. Called when the
        output grid changes: old snapshots no longer match the segment
        geometry and restoring one would write a wrong-shaped labelmap."""
        if not self._sel_op_undo_stack:
            return
        self._sel_op_undo_stack = []
        if reason:
            slicer.util.showStatusMessage(reason, 4000)

    def on_undo_selection_op_clicked(self, checked=False):
        """
        Revert the last Selection Operations Apply from our private undo stack
        (the embedded Segment Editor's history is not always populated for these
        programmatic edits), then resync local state and server.
        """
        if not self._sel_op_undo_stack:
            slicer.util.showStatusMessage(
                "No Selection Operations Apply to undo.", 3000
            )
            return

        segment_id, packed, shape = self._sel_op_undo_stack.pop()
        seg_node = self.get_segmentation_node()
        segmentation = seg_node.GetSegmentation() if seg_node is not None else None
        if segmentation is None or not segmentation.GetSegment(segment_id):
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The target segment of the previous Apply no longer exists.",
            )
            return

        expected_shape = self._output_grid_shape()
        if expected_shape is not None and tuple(shape) != tuple(expected_shape):
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The output grid has changed since this Apply; the snapshot "
                "no longer matches the segment geometry and was discarded.",
            )
            return
        pre_state = (
            np.unpackbits(packed)[: int(np.prod(shape))]
            .reshape(shape)
            .astype(np.uint8)
        )

        self.segment_editor_node.SetSelectedSegmentID(segment_id)
        # show_segmentation re-applies the binary labelmap AND rebuilds the
        # closed surface representation when 3D was being shown, so Show 3D
        # survives Undo. It also updates previous_states and saves to the
        # editor's undo history.
        self.show_segmentation(pre_state)
        self._clear_wand_preview_segment()

        result = self.upload_segment_to_server()
        if result is None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Selection Operations",
                "The segment was reverted locally, but syncing to the server failed.",
            )
        else:
            slicer.util.showStatusMessage("Selection Operation undone.", 3000)

    ###############################################################################
    # Server communication and sync functions
    ###############################################################################

    def update_server(self):
        """
        Reads user-entered server URL from UI, saves to QSettings, updates self.server.
        """
        self.server = self.ui.Server.text.rstrip("/")
        self._set_qsetting(SETTING_SERVER, self.server)
        debug_print(f"Server URL updated and saved: {self.server}")

    def test_server_connection(self):
        """
        Sends a lightweight GET request to see if the configured server responds.
        """
        server_text = self.ui.Server.text
        if not server_text.strip():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Test Connection",
                "Please enter a server URL before testing the connection.",
            )
            return

        self.ui.Server.setText(server_text.strip())
        self.update_server()
        server_url = self.server

        if getattr(self, "_test_server_in_progress", False):
            return
        self._test_server_in_progress = True

        slicer.util.showStatusMessage("Testing nnInteractive server connection...", 2000)
        slicer.app.processEvents()

        response = None
        error_message = None
        try:
            response = requests.get(server_url, timeout=5)
        except requests.exceptions.MissingSchema:
            error_message = (
                "Server URL is invalid. Make sure it starts with 'http://' or 'https://'."
            )
        except requests.exceptions.RequestException as exc:
            error_message = str(exc)
        finally:
            self._test_server_in_progress = False
            slicer.util.showStatusMessage("")

        if response is not None:
            info_message = (
                f"Server at '{server_url}' is reachable."
            )
            QMessageBox.information(
                slicer.util.mainWindow(),
                "Test Connection",
                info_message,
            )
            return
        else:
            QMessageBox.critical(
                slicer.util.mainWindow(),
                "Test Connection",
                f"Failed to reach '{server_url}'.\n\n{error_message}",
            )

    def request_to_server(self, *args, **kwargs):
        """
        Wraps requests.post in a try/except and shows error in pop up windows if necessary.
        """

        with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):

            error_message = None
            try:
                _perf_log("[DEBUG triplanar.perf] POST start url={}".format(
                    args[0] if args else kwargs.get("url")), flush=True)
                # No timeout would block Slicer's main thread indefinitely if the
                # server stalls (e.g. a large-volume inference). Bound it.
                if "timeout" not in kwargs:
                    kwargs["timeout"] = (10, 300)
                response = requests.post(*args, **kwargs)
                _perf_log("[DEBUG triplanar.perf] POST done status={}".format(
                    getattr(response, "status_code", None)), flush=True)
                debug_print('response:', response)
            except requests.exceptions.MissingSchema as e:
                response = None
                if self.server == "":
                    raise RuntimeError("It seems you have not set the server URL yet. You can configure it in the 'Configuration' tab.")
                else:
                    raise RuntimeError(f"Server URL '{self.server}' is unreachable. You can edit the URL in the 'Configuration' tab.")
            except requests.exceptions.ConnectionError as e:
                response = None
                raise RuntimeError(f"Failed to connect to server '{self.server}'. Please make sure the server is running and check the server URL in the 'Configuration' tab.")
            except requests.exceptions.InvalidSchema as e:
                append_text_to_error_message = ""
                if not args[0].startswith("http://"):
                    append_text_to_error_message = "\n\nHint: Perhaps your Server URL in the 'Configuration' tab should start with 'http://'. For example, if your server runs on localhost and port 1527, 'localhost:1527' would not work as a Server URL, while 'http://localhost:1527' would."
                raise RuntimeError(f'{e}{append_text_to_error_message}')

            if response.status_code != 200:
                status_code = response.status_code
                response = None
                raise RuntimeError(f"Something has gone wrong with your request (Status code {status_code}).")

            t0 = time.time()
            # Try to parse JSON and check for a specific error.
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                resp_json = response.json()
                if resp_json.get("status") == "error":
                    if "No image uploaded" in resp_json.get("message", ""):
                        debug_print("No image has been uploaded to the server. Please upload an image first.")
                        self.upload_image_to_server()
                        self.upload_segment_to_server()
                        return self.request_to_server(*args, **kwargs)
                    else:
                        response = None
                        raise RuntimeError(f"Server error: {resp_json.get('message', 'Unknown error')}")

            debug_print('1157 took', time.time() - t0)

        return response

    def upload_image_to_server(self):
        """
        Gets inference-working volume data from Slicer and uploads it.
        """
        debug_print("Syncing image with server...")
        try:
            # Retrieve image data, window, and level.
            t0 = time.time()
            image_data = (
                self.get_inference_image_data()
            )  # Expected to return (image_data, window, level)
            debug_print(f"self.get_inference_image_data took {time.time() - t0}")
            _inf = self.get_inference_volume_node()
            _perf_log("[DEBUG triplanar.perf] upload_image enter: series='{}' shape={}"
                  .format(
                      _inf.GetName() if _inf else None,
                      None if image_data is None
                      else tuple(np.asarray(image_data).shape)),
                  flush=True)

            if image_data is None:
                debug_print("No image data available to upload.")
                return

            t0 = time.time()
            url = (
                f"{self.server}/upload_image"  # Update this with your actual endpoint.
            )

            buffer = io.BytesIO()
            np.save(buffer, image_data)
            raw_data = buffer.getvalue()
            debug_print(f"len(raw_data): {len(raw_data)}")

            files = {"file": ("volume.npy", raw_data, "application/octet-stream")}

            # Create your MultipartEncoder without gzip headers
            from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

            slicer.progress_window = slicer.util.createProgressDialog(autoClose=False)
            slicer.progress_window.minimum = 0
            slicer.progress_window.maximum = 100
            slicer.progress_window.setLabelText("Uploading image...")

            def my_callback(monitor):
                if not hasattr(monitor, "last_update"):
                    monitor.last_update = time.time()
                if time.time() - monitor.last_update <= 0.2:
                    return
                monitor.last_update = time.time()
                slicer.progress_window.setValue(
                    monitor.bytes_read / len(raw_data) * 100
                )
                slicer.progress_window.show()
                slicer.progress_window.activateWindow()
                slicer.progress_window.setLabelText("Uploading image...")
                slicer.app.processEvents()

            encoder = MultipartEncoder(fields=files)
            monitor = MultipartEncoderMonitor(encoder, my_callback)

            try:
                result = self.request_to_server(
                    url, data=monitor, headers={"Content-Type": monitor.content_type}
                )
            finally:
                slicer.progress_window.close()

            if result is not None:
                self._remember_inference_image_state(image_data)
            return result
        except Exception as e:
            debug_print(f"Error in upload_image_to_server: {e}")

    def upload_segment_to_server(self):
        """
        Sends the canonical segment sampled on the inference-working grid.
        """
        debug_print("Syncing segment with server...")
        _perf_log("[DEBUG triplanar.perf] upload_segment enter", flush=True)
        try:
            if not self._ensure_inference_image_uploaded():
                return None
            segment_data = self.get_segment_data(
                reference_volume_node=self.get_inference_volume_node()
            )
            files = self.mask_to_np_upload_file(segment_data)
            url = f"{self.server}/upload_segment"  # Update this with your actual endpoint.

            result = self.request_to_server(
                url, files=files, headers={"Content-Encoding": "gzip"}
            )

            if result is not None:
                self.previous_states["segment_data"] = self.get_segment_data()
            return result
        except Exception as e:
            print("[DEBUG segdata] upload_segment_to_server error: {}".format(e))
            return None

    def _ensure_inference_image_uploaded(self):
        """Upload the active inference image when the server-side image is stale."""
        if not self.image_changed(do_prev_image_update=False):
            return True
        return self.upload_image_to_server() is not None

    ###############################################################################
    # Utility / converters functions
    ###############################################################################

    def get_image_data(self):
        """
        Returns voxel data for the canonical Segment Editor source volume.
        """
        volume_node = self.get_volume_node()
        if volume_node:
            return slicer.util.arrayFromVolume(volume_node)

        return None

    def get_inference_image_data(self):
        """Returns voxel data for the image currently uploaded to nnInteractive."""
        volume_node = self.get_inference_volume_node()
        if volume_node:
            return slicer.util.arrayFromVolume(volume_node)
        return None

    def get_volume_node(self):
        """
        Retrieves the current source volume node chosen in the segment editor widget.
        If nothing is set then use the most recently added scalar volume
        """
        # Get volume node from segment editor widget
        volumeNode = self.ui.editor_widget.sourceVolumeNode()

        if not volumeNode:
            # Get the most recently added volume node, skipping our hidden
            # output-geometry node so it never becomes the source volume.
            geometry_name = getattr(self, "output_geometry_node_name", None)
            volumeNodes = [
                node
                for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                if node.GetName() != geometry_name
            ]
            if volumeNodes:
                volumeNode = volumeNodes[-1]
            # Show this volume node in the segment editor widget
            self.ui.editor_widget.setSourceVolumeNode(volumeNode)

        return volumeNode

    def image_changed(self, do_prev_image_update=True):
        """
        Checks if the inference-working volume changed since the last upload.
        """
        image_data = self.get_inference_image_data()
        if image_data is None:
            debug_print("No volume node found")
            return

        old_image_data = self.previous_states.get("image_data", None)
        inference_volume = self.get_inference_volume_node()
        old_volume_id = self.previous_states.get("image_volume_id", None)

        image_changed = (
            old_image_data is None
            or old_volume_id != inference_volume.GetID()
            or not np.array_equal(old_image_data, image_data)
        )

        if do_prev_image_update:
            self._remember_inference_image_state(image_data)

        return image_changed

    def _remember_inference_image_state(self, image_data=None):
        """Record the inference image that was successfully uploaded."""
        inference_volume = self.get_inference_volume_node()
        if inference_volume is None:
            return
        if image_data is None:
            image_data = slicer.util.arrayFromVolume(inference_volume)
        self.previous_states["image_data"] = copy.deepcopy(image_data)
        self.previous_states["image_volume_id"] = inference_volume.GetID()

    def mask_to_np_upload_file(self, mask):
        """
        Converts a numpy mask into a gzipped file object for POSTing.
        """
        buffer = io.BytesIO()
        np.save(buffer, mask)
        compressed_data = gzip.compress(buffer.getvalue())

        files = {"file": ("volume.npy.gz", compressed_data, "application/octet-stream")}

        return files

    def unpack_binary_segmentation(self, binary_data, decompress=False):
        """
        Unpacks data received from server into a full 3D numpy array (bool).
        """
        _perf_log("[DEBUG triplanar.perf] unpack enter bytes={}".format(
            len(binary_data) if binary_data is not None else None), flush=True)
        if decompress:
            binary_data = binary_data = gzip.decompress(binary_data)

        if self.get_inference_image_data() is None:
            self.capture_image()

        vol_shape = self.get_inference_image_data().shape
        total_voxels = np.prod(vol_shape)
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        unpacked_bits = unpacked_bits[:total_voxels]

        segmentation_mask = (
            unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
        )

        return segmentation_mask

    def ras_to_xyz(self, pos, volume_node=None):
        """
        Converts an RAS position to IJK voxel coords in the requested volume.
        """
        volumeNode = volume_node or self.get_volume_node()

        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas
        )
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)

        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
        xyz = [int(round(c)) for c in point_Ijk[0:3]]
        return xyz


    def xyz_from_caller(
        self,
        caller,
        lock_point=True,
        point_type="control_point",
        volume_node=None,
    ):
        """
        Extract voxel coordinates from a Markups node.
        `point_type` can be either "control_point" or "curve_point".
        """
        if point_type == "control_point":
            n = caller.GetNumberOfControlPoints()
            if n < 0:
                debug_print("No control points found")
                return

            pos = [0, 0, 0]
            caller.GetNthControlPointPositionWorld(n - 1, pos)
            if lock_point:
                caller.SetNthControlPointLocked(n - 1, True)
            xyz = self.ras_to_xyz(pos, volume_node=volume_node)
            return xyz
        elif point_type == "curve_point":
            vtk_pts = caller.GetCurvePointsWorld()
            
            if vtk_pts is not None:
                vtk_pts_data = vtk_to_numpy(vtk_pts.GetData())
                xyz = [
                    self.ras_to_xyz(pos, volume_node=volume_node)
                    for pos in vtk_pts_data
                ]
                debug_print(xyz)
                return xyz

            return []
        else:
            raise ValueError(f'Unknown point_type {point_type}')

    def lasso_points_to_mask(self, points, ras_points=None, volume_node=None):
        """
        Given a list of voxel coords (defining a polygon in one slice),
        returns a 3D mask with that polygon filled in the appropriate slice.
        ras_points: optional array of world-space (RAS, mm) coordinates for the
        same curve points -- used for a more robust planarity check on oblique
        volumes where the RAS->IJK transform inflates voxel-space spread.
        """
        from skimage.draw import polygon

        volume_node = volume_node or self.get_volume_node()
        shape = slicer.util.arrayFromVolume(volume_node).shape
        pts = np.array(points)  # shape (n, 3)

        # Determine the slice axis. Ideally one coordinate is constant, but
        # int-rounding of curve points (ras_to_xyz) can scatter the slice axis
        # across two adjacent voxels when the slice sits near a voxel boundary.
        # Pick the flattest axis and snap it to a single slice; only reject when
        # its spread exceeds the tolerance (a genuinely oblique / multi-slice
        # curve that cannot be treated as a single-slice 2D lasso).
        spreads = [int(pts[:, i].max() - pts[:, i].min()) for i in range(3)]
        const_axis = int(np.argmin(spreads))
        print("[DEBUG lasso_to_mask] pts shape={}, spreads={}, const_axis={}, threshold={}".format(
            pts.shape, spreads, const_axis, LASSO_SLICE_AXIS_MAX_SPREAD))
        print("[DEBUG lasso_to_mask] pts[:,0] range=[{:.2f},{:.2f}], pts[:,1] range=[{:.2f},{:.2f}], pts[:,2] range=[{:.2f},{:.2f}]".format(
            float(pts[:,0].min()), float(pts[:,0].max()),
            float(pts[:,1].min()), float(pts[:,1].max()),
            float(pts[:,2].min()), float(pts[:,2].max())))

        if ras_points is not None and len(ras_points) >= 3:
            # SVD-based planarity check in world space (mm).
            # Works for any slice orientation, including highly oblique volumes
            # where no RAS axis is "constant" even for a valid single-plane lasso.
            # For a planar point cloud the smallest singular value s[2] -> 0;
            # out_of_plane_rms = s[2]/sqrt(N) is the RMS distance from the best-fit
            # plane and should be < 0.1mm for any slice-view lasso.
            rp = np.array(ras_points)
            centered = rp - rp.mean(axis=0)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            out_of_plane_rms = float(s[2]) / np.sqrt(len(rp))
            print("[DEBUG lasso_to_mask] SVD s={}, out_of_plane_rms={:.3f}mm".format(
                [round(float(v), 2) for v in s], out_of_plane_rms))
            if out_of_plane_rms > 2.0:
                print("[DEBUG lasso_to_mask] FAIL(SVD): {:.3f}mm > 2mm".format(
                    out_of_plane_rms))
                raise ValueError(
                    "Expected the lasso points to lie on a single slice plane"
                )
        else:
            # Fallback: voxel-space check when no RAS points supplied.
            if spreads[const_axis] > LASSO_SLICE_AXIS_MAX_SPREAD:
                print("[DEBUG lasso_to_mask] FAIL(voxel): spread[{}]={} > threshold {}".format(
                    const_axis, spreads[const_axis], LASSO_SLICE_AXIS_MAX_SPREAD))
                raise ValueError(
                    "Expected the lasso points to lie on a single slice plane"
                )
        const_val = int(round(np.median(pts[:, const_axis])))

        # Create a blank 3D mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Scan-convert the polygon on its best-fit plane *in voxel space*.
        # The old approach collapsed every point onto a single axis-aligned
        # voxel slice (const_axis/const_val), which is geometrically wrong when
        # the slice view is oblique to the voxel grid (reformatted views of an
        # oblique series): the lasso then spans several voxel slices and the
        # flattening squashes it into a thin, misplaced sheet. Instead we fit
        # the plane the contour actually lies on, rasterize the polygon in that
        # plane's 2D coordinates, then map each filled cell back to voxels.
        cv = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - cv)
        U, V = Vt[0], Vt[1]  # in-plane orthonormal basis (voxel units)
        uv = (pts - cv) @ np.column_stack([U, V])  # (n, 2) plane coords
        umin, vmin = float(uv[:, 0].min()), float(uv[:, 1].min())
        step = 0.5  # 0.5-voxel oversampling keeps the filled region watertight
        pu = (uv[:, 0] - umin) / step
        pv = (uv[:, 1] - vmin) / step
        H = int(np.ceil(pv.max())) + 1
        W = int(np.ceil(pu.max())) + 1
        rr, cc = polygon(pv, pu, shape=(H, W))
        uu = umin + cc * step
        vv = vmin + rr * step
        world = cv[None, :] + uu[:, None] * U[None, :] + vv[:, None] * V[None, :]
        ijk = np.round(world).astype(int)
        xi, yi, zi = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        valid = (
            (xi >= 0) & (xi < shape[2])
            & (yi >= 0) & (yi < shape[1])
            & (zi >= 0) & (zi < shape[0])
        )
        mask[zi[valid], yi[valid], xi[valid]] = 1  # mask is (z, y, x)
        print("[DEBUG lasso_to_mask] oblique raster: filled {} voxels".format(
            int(mask.sum())))

        # Record the lasso plane so show_segmentation can clip the result to
        # this slice +/- N. xyz axis i maps to numpy mask axis (2 - i). Only
        # meaningful for axis-aligned single-view lassos; multi-view sets this
        # to None at the call site so oblique results are never slice-clipped.
        self._last_lasso_slice = (2 - const_axis, const_val)

        return mask

    ###############################################################################
    # Prompt type toggle (positive / negative)
    ###############################################################################

    @property
    def is_positive(self):
        """
        Returns True if the current prompt is set to "positive",
        False if "negative."
        """
        return self.ui.pbPromptTypePositive.isChecked()

    def on_prompt_type_positive_clicked(self, checked=False):
        """
        Called when user presses the "Positive" prompt button.
        """
        # Update UI
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypePositive.setChecked(True)
        self.ui.pbPromptTypeNegative.setChecked(False)
        debug_print("Prompt type set to POSITIVE")

    def on_prompt_type_negative_clicked(self, checked=False):
        """
        Called when user presses the "Negative" prompt button.
        """

        # Update UI
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypePositive.setChecked(False)
        self.ui.pbPromptTypeNegative.setChecked(True)
        debug_print("Prompt type set to NEGATIVE")

    def toggle_prompt_type(self, checked=False):
        """
        Toggle between positive and negative (triggered by 'T' key).
        """
        debug_print("Toggling prompt type (positive <> negative)")
        if self.current_prompt_type_positive:
            self.on_prompt_type_negative_clicked()
        else:
            self.on_prompt_type_positive_clicked()


###############################################################################
# Test hook (used by Reload & Test)
###############################################################################
_test_module_path = (
    Path(__file__).resolve().parents[0]
    / "Testing"
    / "Python"
    / "SlicerNNInteractiveSegmentationTest.py"
)

if _test_module_path.exists():
    import importlib.util as _importlib_util

    _spec = _importlib_util.spec_from_file_location(
        "SlicerNNInteractiveSegmentationTest", str(_test_module_path)
    )
    _test_module = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_test_module)
    SlicerNNInteractiveTest = _test_module.SlicerNNInteractiveSegmentationTest
