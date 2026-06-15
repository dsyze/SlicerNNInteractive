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

## `_rotate_camera_to_view`(v2:forward = 另外两视图平面交线方向)

> 第一版 forward 用"点击视图序列自身的法线"(正对该视图平面)。**已修正**为:相机沿
> **另外两个视图平面法向量的叉乘**方向正交观察,并改为**保持当前焦点/距离、只转朝向**。

### 取每视图世界平面 `_view_world_plane(view_name)`(~3007)
抽取原先内联的"每视图法线/up"逻辑,返回世界 `(normal, up_candidate)` 或 None:
- `use_frame = _get_oblique_camera_align() and volume and _has_oblique_world_frame(volume)`
  -> `_volume_acquisition_frame(volume)`,返回 `(k_axis, j_axis)`(法线/列方向);
- 否则 `_slice_frame_geometry(slice_node, volume)` 返回 `(n, v)`;都不可用返回 None。
- `cbObliqueCameraAlign` 仍决定用采集帧还是 slice 法线。

### `_rotate_camera_to_view`(~3037)
1. 先取 `camera_node` / `camera = camera_node.GetCamera()`。
2. `own = _view_world_plane(view_name)`(own_normal + up_candidate);
   `others = 另两视图`,`target = unit(cross(planeA.normal, planeB.normal))`。
   - 例:Face Red -> `cross(nY, nG)`;Yellow -> `cross(nR, nG)`;Green -> `cross(nR, nY)`。
   - 正交三平面下 `cross(nY,nG)==±nR`,与"正对该视图"一致;只有斜位/三序列才不同。
   - **退化**(`|cross| < 1e-3`,单序列/同序列/平行)-> 回退 `target = own_normal`。
3. **符号(防 180° 翻转,确定性)**:`on = own_normal` 经 `TRIPLANAR_CAMERA_PREFERRED_SIDE`
   翻到惯例侧(Red=+S/Yellow=+A/Green=-R);`if dot(target,on)<0: target=-target`。
   `cam_axis = target`(focal->相机侧),`forward = -cam_axis`。符号只由固定参考决定、
   与相机历史无关,反复点不翻转。
4. **Up**:`up_candidate` 朝 Superior 修正后,投影去掉 forward 分量并归一化;退化(平行)
   取兜底向量;最后 Gram-Schmidt `right=cross(forward,up); up=unit(cross(right,forward))`
   保证正交基(无滚转/错切)。
5. **保持焦点/距离**:`focal=camera.GetFocalPoint()`、`distance=camera.GetDistance()`
   (`<=1e-6` 兜底 300mm);`pos=focal+cam_axis*distance`。`SetFocalPoint/Position/ViewUp`
   -> `ResetCameraClippingRange` -> `_force_render_3d_views`。**不再重新取景**(删除了原
   center/hu/hv/视角/dist 计算)。
6. 调用方不变:`_route_prompt_to_view` 自动旋转 + `pbFaceRed/Yellow/Green` 按钮。

## 坑:GetBounds vs GetRASBounds(已修)

`vtkMRMLVolumeNode.GetBounds()` 返回**未变换的局部** RAS 边界(**不含**父级变换);
`GetRASBounds()` 才是**含父级变换的世界**边界。初版误用 `GetBounds()` 取 center,导致对
**配准斜位**序列:方向轴已正确旋到世界、但 center 用了变换前局部中心 -> 相机焦点偏离、目标脱框
(恰恰是本功能头号用例)。已改为 `GetRASBounds()`,与全文件其它世界边界用法一致。
> 注:v2 重写后 `_rotate_camera_to_view` 不再用体积 bounds(焦点/距离取自相机当前值),
> 该 `GetRASBounds` 现仅存于 `_volume_acquisition_frame` 的 center 计算。

## 回退与边界

checkbox 关闭 / volume 为 None / 世界帧非斜位 / acquisition frame 退化 -> 全部落到原
`_slice_frame_geometry` 路径,标准方向行为不回归。单序列与三平面模式均通过
`_view_background_volume(view_name)` 取该视图背景序列算 frame,单序列下三视图取同一体积也正确。
