# 单序列多角度定位模式

用途：记录 2026-06-26 对三平面功能的边界修复。

## 背景

原先的 `cbTriPlanarMode` 同时承担两个含义：

1. 在 3D 视图显示 Red/Yellow/Green 三个定位平面，并提供相机正对按钮。
2. 启用多序列推理：按点击视图路由到不同序列、启用全序列融合、使用三序列并集输出网格。

当用户只使用一个高清序列，并把同一个序列放在多个视图里观察时，继续走第 2 条多序列逻辑会带来副作用：

- 定位线/手动旋转被禁用。
- 3D 分割模型走三平面延迟重建，容易被误认为 VR 模型消失。
- lasso/scribble 被按视图序列重采样或走全序列退化逻辑，可能出现笔画为空或部分区域画不上。
- all-series fusion 反复发现序列不足再回退，没有收益。

## 新边界

新增内部状态：

- `_triplanar_mode`：用户勾选三平面功能，允许显示定位面和相机正对按钮。
- `_triplanar_multiseries_active`：Red/Yellow/Green 中至少有 2 个不同体数据时才为真。
- `_triplanar_inference_active()`：只有 `_triplanar_mode` 和 `_triplanar_multiseries_active` 同时为真时才启用多序列推理。

因此：

- 单序列多角度观察：只显示定位面，推理仍走单源体积。
- 多序列三平面推理：才启用按视图路由、全序列融合、union output grid、三平面 3D 延迟重建。

## 修复点

- `_setup_triplanar_views()` 先判断视图背景是否包含至少 2 个不同序列。
- `on_apply_plane_display_volumes_clicked()` 在三平面开关已启用时也会刷新该判断，避免用户重新分配视图后旧的多序列状态残留。
- 不足 2 个序列时进入单序列定位模式：
  - 不启用高分辨率 union 输出网格。
  - 不启用序列融合。
  - 不禁用定位线旋转。
  - 不改写 `_active_inference_volume_override`。
  - 清空旧 fusion cache，防止旧多序列结果污染。
- `point/bbox/lasso/scribble/magic wand` 只有在 `_triplanar_inference_active()` 时才走多序列路径。
- `show_segmentation()` 只有在真正多序列推理时才跳过即时 closed-surface 重建；单序列定位模式保留普通 3D/VR 显示行为。
- `cbShow3DTriPlanar` 只有在多序列推理时才影响 3D surface，避免单序列定位模式下误隐藏普通 3D 分割。

## 使用建议

如果当前只有一个高清序列：

1. Red/Yellow/Green 可以都显示同一个源体数据。
2. 可以开启三平面定位面用于观察。
3. 推理、涂鸦、套索仍按单序列执行。
4. 需要真正融合时，至少给 Red/Yellow/Green 中两个视图分配不同且已对齐的序列。
