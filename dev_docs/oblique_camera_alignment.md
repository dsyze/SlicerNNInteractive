# 3D 相机精确对齐斜位序列采集平面

> 用途:记录"自动旋转 3D 相机时,精确对齐到任意斜位(含配准造成的斜位)序列真实采集平面"的设计与实现。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client,server 未改动。
> 是 [[triplanar_multiseries]] "自动旋转 3D 到激活序列"功能的增强,配套 [[multi_series_fusion_and_registration]]、
> [[slice_view_behaviors]] 阅读。

## 背景与根因

原 `_rotate_camera_to_view(view_name)`(~3007)经 `_slice_frame_geometry(slice_node, volume)`
(~2880)读 `slice_node.GetSliceToRAS()` 取平面法线/up,**依赖 2D slice 已被
`RotateToVolumePlane` 旋到斜位**。而当序列斜位**来自父级配准变换**
(`SERIES_ALIGNMENT_TRANSFORM_NODE_NAME`,见 [[multi_series_fusion_and_registration]])、其自身
`GetIJKToRASDirectionMatrix` 近单位阵时,`_volume_is_oblique`(~605,只看方向矩阵)返回 False ->
slice 不旋转 -> 相机退回标准横/冠/矢方位。这正是用户看到"只能转到正上/正前/正左"的根因。

修法(已与用户确认):升级现有自动旋转,**直接从序列体积方向矩阵(世界坐标,含父级配准变换)算相机**,
不依赖 slice 当前朝向;加复选框在"斜位对齐/标准方向"间切换。触发入口
`_route_prompt_to_view`(交互自动旋转)与手动 `pbFaceRed/Yellow/Green` 按钮**无需改**,均调
`_rotate_camera_to_view`,改造后自动生效。

## 新增

- 常量 `SETTING_OBLIQUE_CAMERA_ALIGN`(~177);getter `_get_oblique_camera_align()`(默认 On)/
  toggle `_on_oblique_camera_align_toggled`。
- 复选框 `cbObliqueCameraAlign`("Align 3D camera to oblique series",.ui 内 `cbAutoRotateCamera`
  之后);连接在 `init_ui_functionality` 内 `cbAutoRotateCamera` 连接块之后(blockSignals 模式)。
- `_volume_acquisition_frame(volume)`(~2937):返回世界 RAS 的 `(center, i_axis, j_axis, k_axis)`。
  - 取 `GetIJKToRASDirectionMatrix` 三列 I/J/K 方向;
  - 有父变换时用 `vtkMRMLTransformNode.GetTransformBetweenNodes(tnode, None, vtkGeneralTransform)`,
    对方向向量用**两点差法**旋到世界:`TransformPoint(v) - TransformPoint([0,0,0])`(不能直接
    `TransformPoint(v)`,会混入平移);
  - 归一化,任一退化返回 None;
  - `center` 用 `volume.GetRASBounds()` 世界 AABB 中点。语义:`j_axis`=列方向=view-up,
    `k_axis`=切片堆叠方向=法线。
- `_has_oblique_world_frame(volume)`(~2987):`_volume_is_oblique` 为真即真;否则取
  `_volume_acquisition_frame`,若任一世界轴最大分量 < `OBLIQUE_COS_THRESHOLD` 也判斜位。
  **核心修复点**:让"配准变换造成的斜位"(体积自身方向矩阵近单位)也能命中,不再被裸
  `_volume_is_oblique` 漏掉。

## 改造 `_rotate_camera_to_view`

取到 `slice_node`、`volume = _view_background_volume(view_name)` 后:
```
use_frame = _get_oblique_camera_align() and volume and _has_oblique_world_frame(volume)
if use_frame: frame=_volume_acquisition_frame(volume)
              center; n=k_axis; up=j_axis; hu/hv 由 world AABB 沿 i/j 半跨投影
if center is None:  # 回退:checkbox off / None / 非斜位 / 退化帧
    geometry=_slice_frame_geometry(slice_node, volume)  # 完全保留原行为
    center,_,v,n,hu,hv=geometry; up=list(v)
```
下游全部复用:`TRIPLANAR_CAMERA_PREFERRED_SIDE` 法线符号翻转(Red=+S/Yellow=+A/Green=-R)、
view-up 朝 Superior 修正、距离按"视角 + max(hu,hv)"、`SetFocalPoint/Position/ViewUp`、
`ResetCameraClippingRange`、`_force_render_3d_views`。

## 坑:GetBounds vs GetRASBounds(已修)

`vtkMRMLVolumeNode.GetBounds()` 返回**未变换的局部** RAS 边界(**不含**父级变换);
`GetRASBounds()` 才是**含父级变换的世界**边界。初版误用 `GetBounds()` 取 center,导致对
**配准斜位**序列:方向轴已正确旋到世界、但 center 用了变换前局部中心 -> 相机焦点偏离、目标脱框
(恰恰是本功能头号用例)。已改为 `GetRASBounds()`(`_volume_acquisition_frame` 与
`_rotate_camera_to_view` frame 分支两处),与全文件其它世界边界用法一致。

## 回退与边界

checkbox 关闭 / volume 为 None / 世界帧非斜位 / acquisition frame 退化 -> 全部落到原
`_slice_frame_geometry` 路径,标准方向行为不回归。单序列与三平面模式均通过
`_view_background_volume(view_name)` 取该视图背景序列算 frame,单序列下三视图取同一体积也正确。
