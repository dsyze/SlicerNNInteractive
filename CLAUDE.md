# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
必须用中文和我交流。这个仓库是用来对这个插件进行二次开发，以满足我们自己的需要。
dev_docs目录是用来放计划文档跟随进度的，你会和codex协同工作，请合理使用说明文档交流。AGENTS.md是codex的记忆文件。
加调试输出debug时，如果我没有明确说问题已解决，就不准擅自删除调试代码。
## Overview

`SlicerNNInteractive` brings [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) (deep-learning-based interactive 3D image segmentation) into [3D Slicer](https://www.slicer.org/). It has two independently deployed halves that talk over HTTP:

- **Server** (`server/`) — a FastAPI app wrapping an `nnInteractiveInferenceSession`. Needs an NVIDIA GPU (~10GB VRAM recommended). Published to PyPI as `nninteractive-slicer-server` and to Docker Hub as `coendevente/nninteractive-slicer-server`.
- **Client** (`slicer_plugin/`) — a scripted 3D Slicer extension (the `nnInteractive` module under the Segmentation category). The client may run on the same machine as the server.

## Commands

### Server

```bash
cd server
python -m build                                          # build the wheel/sdist (what CI does)
uv run nninteractive-slicer-server --host 0.0.0.0 --port 1527   # run locally
docker build -t nninteractive_slicer_server .            # build container (see build.sh)
docker run -p 1527:1527 --gpus all --rm -d nninteractive_slicer_server   # run container (see run.sh)
```

Set `NNI_DEVICE=cpu` (or `cuda:0`) to override device auto-detection. On first run the server downloads model weights from Hugging Face (`nnInteractive/nnInteractive`, `nnInteractive_v1.0`) into `.nninteractive_weights/`.

### Client extension

There is no standalone build. The extension is loaded into 3D Slicer via the Extension Wizard pointed at the `slicer_plugin/` folder, or installed from the Extensions Manager. `cmake -S . -B build` only matters when packaging as an official Slicer extension.

### Tests

The test suite (`slicer_plugin/SlicerNNInteractive/Testing/Python/SlicerNNInteractiveSegmentationTest.py`) runs **inside Slicer**, not via `pytest`. It requires a running server.

1. Start the server and configure its URL in the module's `Configuration` tab (or set `SLICER_NNI_TEST_SERVER_URL`).
2. Enable Developer Mode in Slicer.
3. Open `Self Tests`, pick `SlicerNNInteractive`, click `Reload and Test`.

It downloads the `MRBrainTumor2` sample volume, replays each prompt sequence, and compares output against frozen reference NIfTI masks in `Testing/Data/` using a Dice threshold of 0.99. To regenerate references, set `SLICER_NNI_GENERATE_TEST_MASK=1`, run once, inspect the masks, then rerun without the variable.

## Conventions

- **ASCII only in `.py` files** — CI (`check-utf8.yml`) fails on any non-ASCII character in Python source.
- **Version consistency** — the version in `server/pyproject.toml` must match the release git tag (`vX.Y.Z`) before tagging; CI warns on mismatch. Releases trigger on `v*.*.*` tags pushed to `main`.
- `server/requirements.txt` (loose ranges, used by Docker/`uv`) and `server/pyproject.toml` `dependencies` (pinned, used by the PyPI package) list the same packages — keep them in sync when bumping.

## Architecture

### Client/server protocol

Each interaction type maps to a FastAPI endpoint in `server/.../main.py`:

- `/upload_image`, `/upload_segment` — push the current volume / current segment labelmap to the server.
- `/add_point_interaction`, `/add_bbox_interaction` — JSON body (voxel coords + `positive_click`).
- `/add_lasso_interaction`, `/add_scribble_interaction` — multipart upload of a gzipped `.npy` 3D mask.

All segmentation results are returned as gzip-compressed bit-packed binary masks (`segmentation_binary` / `unpack_binary_segmentation`); the client reshapes them back to the volume shape. The server keeps a single global `PROMPT_MANAGER` holding one image + one inference session — it is **stateful and single-session**, so concurrent clients would clobber each other.

### Coordinate ordering

A recurring source of bugs: numpy volumes are ordered `(z, y, x)`, while voxel coordinates exchanged with the model are `(x, y, z)`. The client reverses coordinate lists (`xyz[::-1]`) before sending point/bbox prompts. Respect this when touching coordinate logic on either side.

### Client sync model (`slicer_plugin/.../SlicerNNInteractive.py`)

The whole client is one large `SlicerNNInteractiveWidget`. Key mechanism: the `@ensure_synched` decorator wraps every prompt method (`point_prompt`, `bbox_prompt`, `lasso_or_scribble_prompt`). Before running a prompt it diffs the current volume and selected segment against `self.previous_states` and re-uploads whichever changed. This is how the server stays in sync without the client tracking server state explicitly. If the server reports "No image uploaded", `request_to_server` transparently re-uploads and retries.

Prompts are driven by Slicer Markups nodes registered in `self.prompt_types` (point = Fiducial, bbox = ROI, lasso = ClosedCurve), each with an `on_placed` observer that fires the corresponding prompt. The **scribble** prompt is different: it uses a hidden background `qMRMLSegmentEditorWidget` with the Paint effect, and sends the *diff* of consecutive strokes.

All segmentation always applies to the currently selected segment in the embedded Segment Editor. Five hidden nodes are internal scaffolding and are deliberately excluded from `get_segmentation_node()` / `_existing_segmentation_node()` -- the exclusion set is centralized in `_internal_segmentation_node_names()`: `"ScribbleSegmentNode (do not touch)"` (Paint scratch space), `"MagicWandPreviewSegmentNode (do not touch)"`, `"Lasso3dInputSegmentNode (do not touch)"`, `"Lasso3dPreviewSegmentNode (do not touch)"`, and `"NativeSeriesInferencePreviewSegmentNode (do not touch)"` (see Custom features below).

The test class is imported into the module file at the bottom and exposed as `SlicerNNInteractiveTest` so Slicer's `Reload and Test` picks it up.

### Custom features (secondary development)

These were added on top of upstream nnInteractive and live **entirely in the client** (`SlicerNNInteractive.py`); the server (`server/.../main.py`) is unchanged from upstream, so none of them add or call new endpoints. Design notes and reviews live in `dev_docs/`, named `feature_name.md` (proposal) plus optional `feature_name_review.md` (review) -- e.g. `dev_docs/semantic_selection_boolean_operations.md` and its `_review.md`.

**Selection Operations** (`dev_docs/semantic_selection_boolean_operations.md`) -- boolean editing of the current segment (Add = OR, Subtract = AND NOT, Intersect = AND) against one of four operands chosen in `cbOperandSource` (indices are the module constants `OPERAND_SOURCE_ROI/WAND/SEGMENT/LASSO3D`):
- **ROI operand** -- box/sphere/ellipsoid (`cbRoiShape` -> `ROI_SHAPE_BOX/SPHERE/ELLIPSOID`); `roi_node_to_mask` rasterizes via OBB containment so it is correct on oblique volumes.
- **Magic Wand operand** (`dev_docs/selection_operand_magic_wand.md`) -- multi-point positive/negative seeds (`SelectionOpWandSeeds`) drive a client-side AI mask preview (hidden node `"MagicWandPreviewSegmentNode (do not touch)"`).
- **Segment operand** -- boolean against another segment.
- **Lasso (3D) operand** (`dev_docs/lasso_3d_selection.md`) -- a hidden Segment Editor runs the native Scissors effect; the nnInteractive lasso AI refines it into a preview.

Programmatic edits are tracked in a private undo stack (`_sel_op_undo_stack`) because the embedded Segment Editor's history does not capture them.

**Slice-view behaviors** (`dev_docs/slice_view_behaviors.md`) -- slice intersections, mouse-wheel scrolling snapped to the original voxel grid (`cbSnapSlicesToGrid`), and automatic re-orientation of RAS views to an oblique series' acquisition plane (`_volume_is_oblique`, `_align_views_to_volume_planes`; `OBLIQUE_COS_THRESHOLD`).

**Multi-plane display volumes** (`dev_docs/multi_plane_display_volumes.md`) -- show a different supplemental volume per slice view (Red/Yellow/Green) as a background, independent of the Segment Editor source volume. Snapshot-then-apply with a sticky reapply after editor effects reset backgrounds.

**Output geometry & smoothing** (`dev_docs/output_geometry_and_smoothing.md`) -- an optional high-resolution isotropic output grid (`cbEnableHighResOutput`, hidden node `OUTPUT_GEOMETRY_NODE_NAME`, voxel-budget guarded) the canonical segment is stored on; coarse->fine SDF/Gaussian interpolation of server results (`cbSmoothInterpolate`, `on_smooth_current_segment_clicked`); and non-destructive 2D+3D display smoothing with bake-on-export (`cbDisplaySmooth`, `pbBakeDisplaySmooth`).

**Native-series inference, registration & fusion** (`dev_docs/multi_series_fusion_and_registration.md`, plus `multi_plane_display_volumes.md`) -- run the AI on a supplemental registered series (`cbEnableNativeSeriesInference`) with a preview-before-merge workflow (sync modes in `cbInferenceSyncMode`); auto-register supplemental series to the source via BRAINSFit when DICOM frames of reference differ (`SERIES_ALIGNMENT_TRANSFORM_NODE_NAME`, a registration queue); and SDF-average several series into one smooth combined mask (`cbEnableSeriesFusion`, `pbFuseSeries`).

**Tri-planar multi-series mode** (`dev_docs/triplanar_multiseries.md`) -- `cbTriPlanarMode` ties the above together for three co-registered series. The user manually assigns their series to the Red/Yellow/Green views via the Multi-plane display panel (no auto-detection -- real data has localizers and oblique series); `_setup_triplanar_views` just applies that assignment and turns on high-res output + fusion. Each interaction is routed to the series shown in the view it was made in (`_view_for_ras_point` + the `_active_inference_volume_override` honored by `get_inference_volume_node`), and after every interaction the three directions are direction-weighted-fused (`_fuse_series_results` blurs each series only along its slice/thick direction via `_series_anisotropic_sigma`, oblique-tolerant by approximating that direction to the nearest output-grid axis) and shown (`_maybe_autofuse`). In this mode multi-view lasso accumulation is bypassed (each view is a different series).

**Per-segment opacity** -- the `sldSegmentOpacity` slider; its default persists across sessions and is applied to newly created segments.

**Lasso slice-range clipping** -- when enabled (`cbEnableLassoClip`), a lasso result is clipped to the prompt slice +/- N slices. The lasso's constant slice plane is recorded in `_last_lasso_slice` at submit time and consumed once in `show_segmentation`; non-lasso prompts are never clipped.

**Multi-view lasso** -- accumulate a lasso mask per drawn plane (`cbLassoMultiView`) and submit each as its own lasso interaction so the server session is constrained by all views.

Cross-session preferences live under the `SlicerNNInteractive/` `QSettings` namespace, accessed through the `_get_qsetting` / `_set_qsetting` helpers with key constants named `SETTING_*` (server URL, snap-slices, output spacing, segment opacity, display-smooth enabled/strength, lasso clip enabled/N, lasso multi-view, high-res output, smooth-interpolate). Repeated idioms have shared helpers: `_get_qsetting`/`_set_qsetting` (QSettings), `_safe_add_observer`/`_safe_remove_observer` (VTKObservationMixin guards), `_get_or_create_hidden_segmentation` (hidden preview/scratch segmentations), `_clear_segment_labelmap`, `_internal_segmentation_node_names`. Several manually-managed observers are NOT covered by `removeObservers()` and are torn down explicitly in `cleanup()` (the scribble Paint observer via `_teardown_scribble_observer`, the lasso-3D input observer, and the registration CLI observer).
