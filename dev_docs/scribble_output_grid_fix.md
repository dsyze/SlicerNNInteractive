# 涂鸦输出网格裁剪修复

## 现象

2D 视图中使用涂鸦时，某些区域像被一条整齐的蒙版边界裁掉：边界一侧可以画，另一侧完全画不上。

## 根因

隐藏的 scribble `qMRMLSegmentEditorWidget` 原先把临时 `ScribbleSegmentNode` 绑定到 Segment Editor 的源 volume。三平面/高分辨率输出模式下，最终 segment 已经存储在 `get_output_volume_node()` 上，且三平面输出网格会覆盖多个序列的 RAS 并集；但临时 Paint 画布仍然只覆盖源 volume 的 FOV。Slicer Paint 会按临时 labelmap 的有效范围裁剪输入，所以会出现一条平直 FOV 边界后一侧无法落笔。

## 修复

- scribble 临时 bg/fg segment 改为每次激活时重建到 `get_output_volume_node()`。
- 普通单序列模式下输出网格等于源 volume，行为不变。
- 高分辨率/三平面模式下隐藏 Paint 的 source volume 也使用输出几何，避免被源 volume FOV 裁剪。
- 直接写入模式仅在输出网格等于源 volume 时继续使用可见 Segment Editor 的即时 Paint；否则走隐藏画布并把笔画合并回当前 segment。
- 隐藏编辑器切换 source volume 后恢复 2D 背景：多平面模式恢复各自背景，普通模式恢复源 volume。

## 回归覆盖

`SlicerNNInteractiveSegmentationTest._test_scribble_scratch_output_geometry` 验证高分辨率输出开启后：

- scribble scratch 使用输出网格；
- 可见编辑器直接写入路径不会被误用；
- 隐藏 Paint source volume 绑定到输出网格。
