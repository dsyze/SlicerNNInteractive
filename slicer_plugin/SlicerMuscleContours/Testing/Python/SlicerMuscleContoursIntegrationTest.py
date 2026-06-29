"""Run inside Slicer with --python-script.

The script writes a result file because the Windows Slicer launcher detaches
from the calling console.
"""

import json
import os
import traceback

import numpy as np
import slicer
import vtk


RESULT_PATH = os.path.join(
    os.path.dirname(__file__), "SlicerMuscleContoursIntegrationTest.result.json"
)


def add_circle(logic, volume, name, slice_index, radius, center=(16.0, 16.0)):
    node = logic.createContourNode(
        volume,
        name,
        "Red",
        slice_index,
        [0.0, 0.0, float(slice_index)],
        [0.0, 0.0, 1.0],
    )
    for angle_degrees in range(0, 360, 45):
        angle = np.deg2rad(angle_degrees)
        node.AddControlPointWorld(
            vtk.vtkVector3d(
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                float(slice_index),
            )
        )
    return node


def run():
    from SlicerMuscleContours import SlicerMuscleContoursLogic

    slicer.mrmlScene.Clear()
    checks = {}

    image = vtk.vtkImageData()
    image.SetDimensions(32, 32, 32)
    image.AllocateScalars(vtk.VTK_SHORT, 1)
    image.GetPointData().GetScalars().Fill(0)
    volume = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLScalarVolumeNode", "IntegrationReference"
    )
    volume.SetAndObserveImageData(image)
    volume.SetSpacing(1.0, 1.0, 1.0)

    logic = SlicerMuscleContoursLogic()
    first = add_circle(logic, volume, "TestMuscle", 8, 7.0)
    second = add_circle(logic, volume, "TestMuscle", 20, 4.0)

    checks["create_two_contours"] = len(
        logic.contourNodes(volume, "TestMuscle")
    ) == 2
    checks["smooth_curve_generated"] = (
        first.GetCurveWorld() is not None
        and first.GetCurveWorld().GetNumberOfPoints() > 8
    )

    first.SetNthControlPointPositionWorld(0, vtk.vtkVector3d(23.0, 16.0, 11.0))
    projected = [0.0, 0.0, 0.0]
    first.GetNthControlPointPositionWorld(0, projected)
    checks["point_constrained_to_plane"] = abs(projected[2] - 8.0) < 1e-6

    copied = logic.copyContourToPlane(
        first,
        volume,
        "Red",
        14,
        [0.0, 0.0, 14.0],
        [0.0, 0.0, 1.0],
    )
    copied_point = [0.0, 0.0, 0.0]
    copied.GetNthControlPointPositionWorld(0, copied_point)
    checks["copy_preserves_points"] = (
        copied.GetNumberOfControlPoints() == first.GetNumberOfControlPoints()
    )
    checks["copy_moves_to_target_plane"] = abs(copied_point[2] - 14.0) < 1e-6

    # Keep interpolation inputs to two key slices; copied contour belongs to the
    # same group, so remove it after validating copy behavior.
    slicer.mrmlScene.RemoveNode(copied)

    first_mask = logic._contourSliceMask(first, volume, 2, 8)
    checks["rasterized_mask_nonempty"] = int(first_mask.sum()) > 50
    checks["rasterized_mask_shape"] = tuple(first_mask.shape) == (32, 32)

    segmentation, segment_id = logic.generateSegmentation(
        volume, "TestMuscle", None
    )
    segment = segmentation.GetSegmentation().GetSegment(segment_id)
    checks["segment_created"] = segment is not None
    checks["segment_named"] = segment is not None and segment.GetName() == "TestMuscle"
    checks["closed_surface_created"] = (
        segmentation.GetSegmentation().ContainsRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        )
    )

    segment_array = slicer.util.arrayFromSegmentBinaryLabelmap(
        segmentation, segment_id, volume
    )
    checks["interpolation_nonempty"] = int(segment_array.sum()) > 0
    checks["middle_slice_filled"] = int(segment_array[14].sum()) > 0
    checks["outside_range_empty"] = (
        int(segment_array[:8].sum()) == 0 and int(segment_array[21:].sum()) == 0
    )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed": failed,
    }


try:
    result = run()
except Exception:
    result = {
        "status": "error",
        "traceback": traceback.format_exc(),
    }

with open(RESULT_PATH, "w", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)

slicer.util.exit(0 if result["status"] == "passed" else 1)
