# 闭合曲线 3D 裁剪当前分割(Draw Crop Region)

> 用途:记录"在 3D 视图画一条闭合曲线、沿视线方向把当前 segment 裁成内部/外部"的设计与实现。
> 实现全部在 client(`SlicerNNInteractive.py` + `Resources/UI/SlicerNNInteractive.ui`),server 未改动,
> 不新增任何 server 端点。本文不写死行号(行号随代码漂移)。
> 配套阅读 [[crop_segment_by_box]](盒子裁剪,共享同一条写回/撤销/同步链路)、
> [[semantic_selection_boolean_operations]]、[[lasso_3d_selection]]、
> [[output_geometry_and_smoothing]]、[[triplanar_multiseries]]。

## 背景与目标

盒子裁剪([[crop_segment_by_box]])只能用轴对齐/旋转的长方体,无法贴着不规则结构"沿当前看到的
轮廓"裁。本功能让用户像在模型表面套索一样:**在 3D 视图里点出一条闭合曲线,然后沿当前视线方向把
当前段裁成曲线内部或外部**。

设计决策(已与用户确认):
- **绘制方式 = 点击放控制点的闭合曲线**(`vtkMRMLMarkupsClosedCurveNode`,与套索同款 markup;
  亮黄线 + 半透明填充),不是拖拽 Scissors。
- **内/外选择 = 两个明确按钮**(`Crop Inside` / `Crop Outside`),绘制完成后才启用;无弹窗、无下拉。
- **纯几何裁剪**:把闭合曲线沿视线拉伸成无限棱柱,与当前段做布尔交/差。**不调用 AI**。
- 仅作用于**当前选中 segment**;只删体素、从不新增。

## 两个不变量(与盒子裁剪一致)

1. **统一几何基准 = 输出网格**。读用 `get_segment_data()`、写用 `show_segmentation()`,都基于
   `get_output_volume_node()`。区域掩码**直接在输出网格体素坐标上构建**(遍历当前段前景体素 → 投影
   到屏幕 → 点在多边形内),因此天然兼容高分辨率/三平面/斜位,且**不需要** `_to_output_grid` 桥接
   (不在源网格栅格化)。
2. **相机基在绘制结束瞬间冻结**。挤出方向 = 绘制完成时的相机视线(DOP)。冻结后旋转视角不影响裁剪
   结果——这是与"实时读相机"的本质区别。

## UI

在 Selection Operations 分组(`selectionOpsGroup`)的 `cbKeepCropRoi` 之后新增:
- `lblCropRegionHint`(提示标签)
- `pbDrawCropRegion`("Draw Crop Region",**checkable**,进入/退出绘制) + `pbCancelDrawingCrop`
  ("Cancel Drawing",初始隐藏)
- `pbCropInsideRegion`("Crop Inside") + `pbCropOutsideRegion`("Crop Outside"),初始 `enabled=false`

信号连接在 `init_ui_functionality` 内 `pbCropSegmentByBox` 连接之后,同时设置 Cancel 隐藏、Crop 禁用。

## 实例变量与节点(独立于套索提示)

`__init__` 中(`self._crop_roi_node` 旁):
- `self._crop_region_curve_node` —— 专用闭合曲线节点(名 `CROP_REGION_CURVE_NODE_NAME = "CropRegionCurve"`)。
- `self._crop_region_curve_observer_tag` —— 曲线节点上的**手动** AddObserver tag(不被 `removeObservers()`
  覆盖,须显式拆除,见 `_destroy_crop_region_curve`/`cleanup`)。
- `self._crop_region_interaction_observer_tag` —— 交互节点 `InteractionModeChangedEvent` 观察者 tag,
  用于双击闭合后自动收尾。
- `self._crop_region_camera_basis` —— 绘制结束瞬间冻结的相机基 dict,初值 None。

## 绘制态状态机

- `on_draw_crop_region_clicked(checked)`:
  - `checked=True`:校验有 3D 视图 + 已载入 volume(否则提示并取消勾选);取消 scribble 勾选避免抢点击;
    `_get_or_create_crop_region_curve()` 建/取曲线并 `RemoveAllControlPoints()`(开新环);清基、禁用 Crop
    按钮、隐藏 Cancel;挂交互节点观察者;进入 placement(`SetReferenceActivePlaceNodeClassName` +
    `SetActivePlaceNodeID` + `SetPlaceModePersistence(1)` + `SetCurrentInteractionMode(Place)`)。
  - `checked=False`(再次点按钮收尾):拆交互观察者 → `ViewTransform` → `_finalize_crop_region_drawing()`。
- `on_crop_region_point_modified`(曲线手动观察者,`PointModifiedEvent`):点数 > 0 时显示 Cancel 按钮。
  **不在此冻结相机**。
- `_on_crop_region_interaction_mode_changed`(交互节点观察者):用户双击/右键闭合使模式从 Place 回到
  ViewTransform 时,屏蔽信号地取消勾选 Draw 按钮 → 拆交互观察者 → `_finalize_crop_region_drawing()`
  (自动收尾)。按钮 toggled 收尾是**保底**手段(兼容某些 Slicer 版本双击后仍停在 Place 模式)。
- `_finalize_crop_region_drawing()`(幂等):点数 < 3 不冻结、保持 Crop 按钮禁用;否则
  `_capture_3d_camera_basis()` 冻结相机基并启用 Crop 按钮。
- `on_cancel_drawing_crop_clicked` / `_exit_crop_region_drawing`:退出 placement、取消勾选、销毁曲线、
  禁用 Crop、隐藏 Cancel。
- `cleanup()`:`_remove_crop_region_interaction_observer()` + `_destroy_crop_region_curve()`。

## 相机基(`_capture_3d_camera_basis`)

取第一个 3D 视图 `cameras.logic().GetViewActiveCameraNode(view_node).GetCamera()`:
`forward = GetDirectionOfProjection()`(挤出方向)、`up = GetViewUp()`、`cam_pos = GetPosition()`;
Gram-Schmidt 得正交 `right = forward x up`、`up = right x forward`(up 与 forward 平行时取 off-axis
兜底);记录 `parallel = GetParallelProjection()`。

## 区域掩码(`_crop_region_to_mask`)

1. `poly_world = vtk_to_numpy(GetCurvePointsWorld().GetData())`(稠密曲线点,平滑轮廓);< 3 点返回 None。
2. `idx = np.argwhere(current_mask_bool)`(行序 k,j,i);空则返回全 False。
3. `_output_ijk_to_world_matrix()` 取输出网格 IJK→世界 4x4(线性父变换并入矩阵 → 全向量化;非线性父
   变换 → `_world_apply_parent_transform` 逐点 `TransformPoint`,仅前景体素)。把体素中心齐次坐标
   `(i,j,k,1)`(注意 numpy 轴序 z,y,x)乘矩阵得世界 RAS。
4. `_project_to_screen(pts, basis)`:平行投影丢弃 forward 分量(`u = pts.right, v = pts.up`);透视投影除以
   深度(`u = (d.right)/(d.forward)`),曲线点与体素**用同一公式**保证一致。
5. `_points_in_polygon`:向量化偶数-奇数射线法(外层循环 M 条边,每步对 N 个体素向量化),判断每个前景
   体素投影是否落在曲线多边形内。
6. 写回 `region[idx] = inside`,返回 (z,y,x) bool。

无限棱柱沿 forward 挤出 == 只看屏幕 (u,v) 投影是否在多边形内,等价 `vtkImplicitSelectionLoop(normal=DOP)`,
但用 numpy 向量化(无 VTK 逐点调用)且只测前景体素 → 高效、斜位/高分辨率正确。

## 应用(`_apply_crop_region` → 镜像 `on_crop_segment_by_box_clicked`)

1. 守卫:曲线存在 + 控制点 >= 3 + 相机基已冻结;否则提示返回。
2. 有选中 segment;否则提示返回。
3. `current = get_segment_data().astype(uint8)`;None/空 → 提示返回。
4. `region = _crop_region_to_mask(current.astype(bool))`;None → "至少 3 点"提示。shape 守卫不匹配则中止。
5. **零重叠**(`(current & region)` 为空)→ 提示"裁剪区域未覆盖任何体素,操作已取消",**不写入**(Inside 会清空、
   Outside 无变化,统一取消)。
6. `Crop Inside = current & region`,`Crop Outside = current & ~region`;结果 == current 提示"未改动"返回。
7. `_record_selection_op_undo(seg_id, current.copy())`(**复用** Selection Ops 私有撤销栈,由
   `pbUndoSelectionOp`/`on_undo_selection_op_clicked` 还原)→ `show_segmentation(cropped)` → `setup_prompts()`
   → `_exit_crop_region_drawing()`(裁后自动退出绘制态、清曲线)→ `upload_segment_to_server()`(本地编辑必须
   同步,否则下次 prompt 的 `@ensure_synched` 用服务器旧 mask 覆盖)。
8. 全程 try/except,失败 `print("[DEBUG crop_region] ...")` + `showStatusMessage`,不破坏现有分割数据。

## 兼容性与边界

- **斜位/高分辨率/三平面**:全程基于 `get_output_volume_node()` + 真实 IJK→世界矩阵,自动正确;
  `show_segmentation` 在三平面走 debounced 3D 重建分支。
- **空段 / < 3 点 / 零重叠**:分别提示,零重叠取消不写入。
- **多 3D 视图**:用 `threeDWidget(0)`,曲线 display 锁到该视图。
- **不干扰套索**:专用节点 + 独立 ID,进入绘制只取消其他工具勾选,不碰 `prompt_types["lasso"]["node"]`。
- **撤销**:复用私有撤销栈,`Undo` 即恢复并重新同步。
- **显示平滑**:复用 `show_segmentation` + `setup_prompts` 刷新链。
- **平行/透视相机**:`_project_to_screen` 按 `GetParallelProjection()` 分支;透视为近似投影。

## 已知限制

- 裁剪沿**冻结的视线方向**拉伸成无限棱柱:绘制结束后旋转视角不改变结果(符合预期),但若想换方向裁剪需
  重画。透视相机用近似投影,平行相机(Slicer 3D 默认)精确。
- 双击闭合是否自动退出 placement 取决于 Slicer 版本;不退出时点 `Draw Crop Region` 按钮收尾即可(保底路径)。
