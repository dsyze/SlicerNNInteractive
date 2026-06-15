# 盒子裁剪当前分割(Crop Segment by Box)

> 用途:记录"用 ROI 盒子一键裁剪当前 segment"的设计与实现。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client,server 未改动。
> 配套阅读 [[semantic_selection_boolean_operations]]、[[selection_operand_magic_wand]]、
> [[output_geometry_and_smoothing]]、[[triplanar_multiseries]]。

## 背景与目标

像 CT 三维重建软件的"裁剪"那样:**转动 3D 视角、拖一个盒子框住要保留的区域,一键去除盒外
部分**,而不是用剪刀(Scissors)逐层慢慢清。

设计决策(已与用户确认):
- 采用**"清除盒子外体素"**(当前 segment 与 ROI 盒子求交),**不**用 Crop Volume 物理裁剪
  参考体积几何。后者会破坏 `get_output_volume_node()` 不变量、与高分辨率输出几何
  (`OUTPUT_GEOMETRY_NODE_NAME`)/三平面联合几何系统冲突,且难撤销。
- 仅作用于**当前选中 segment**。

## 核心正确性:网格桥接(复用既有路径)

关键风险是网格不一致:
- `roi_node_to_mask(roi_node, shape_idx=None)`(~6119)在**源网格**栅格化(内部用
  `get_volume_node()` + `get_image_data().shape`,OBB 容器判定,斜位/旋转 ROI 正确)。
- `get_segment_data()`(~5148)默认在**输出网格** `get_output_volume_node()` 取 mask;
  高分辨率输出/三平面下输出网格 != 源网格。

解决方式与现有 `apply_boolean_operation`(~5874)完全一致:先 `_to_output_grid(box_src)`
(~1056,内部 `_resample_mask_between_volumes`)把源网格 mask 桥接到输出网格(相同网格 no-op,
失败 `_disable_high_res_output` 回退),再与输出网格的 `current` 求交。**不能裸 `&`**。

`roi_node_to_mask` 新增可选参数 `shape_idx`:为 None 时读 `cbRoiShape`(保持原行为,
现有调用方 ~5920 单参数不受影响);裁剪固定传 `ROI_SHAPE_BOX`,不依赖 Selection Ops 的形状
选择器,也不改 UI 状态。

## UI

在 Selection Operations 分组(`selectionOpsGroup`)的 Apply/Undo 动作行之后:
- `pbPlaceCropRoi`("Place Crop ROI")
- `pbCropSegmentByBox`("Crop Segment by Box")
- `cbKeepCropRoi`("Keep crop box after cropping",默认不勾 = 裁后删盒;勾选可调整后再裁)

信号连接在 `init_ui_functionality` 内 `pbClearRoi` 连接之后。

## 专用 crop ROI 生命周期(独立于 SelectionOpROI)

成员 `self._crop_roi_node`(在 `__init__` 声明,~287)。方法群放在 `_destroy_selection_roi`
之后:
- `_segment_ras_bounds()` —— 当前 segment 非零体素的世界 RAS 包围盒。从 `get_segment_data()`
  (输出网格)的 `np.argwhere`(行序 k,j,i)取 IJK 包围盒,用 `get_output_volume_node()` 的
  `GetIJKToRASMatrix` + 父级变换(`GetTransformBetweenNodes(tnode, None, ...)`,与 [[triplanar_multiseries]]
  的变换惯例一致)把 8 个角转世界 RAS。空/失败返回 None。
- `_initialize_crop_roi_geometry(node)` —— 用上面的包围盒设 center/半跨 x1.05;None 时回退
  `_initialize_selection_roi_geometry`(体积中心)。
- `_get_or_create_crop_roi()` —— 结构照搬 `_get_or_create_selection_roi`,节点名
  "CropSegmentROI",display 复用 `_configure_selection_roi_display`(橙色可交互手柄),
  **不建 sphere/ellipsoid 预览**(裁剪永远是盒子)。
- `_destroy_crop_roi()` —— 删节点;在 `cleanup()` 内 `_destroy_selection_roi()` 旁调用。

## 回调流程(`on_crop_segment_by_box_clicked`)

1. 校验 crop ROI 存在 + 有选中 segment。
2. `current = get_segment_data().astype(uint8)`;None/空 -> 提示返回。
3. `box_src = roi_node_to_mask(crop_roi, shape_idx=ROI_SHAPE_BOX)` -> `box_out = _to_output_grid(box_src)`;
   桥接失败 `_disable_high_res_output` 并退回源网格,shape 守卫不匹配则中止(不动 segment)。
4. `cropped = current & box_out`;== current 提示"未改动"返回;== 0 提示"已清空,可 Undo"(仍写入)。
5. `_record_selection_op_undo(seg_id, current.copy())`(**复用** Selection Ops 私有撤销栈
   `_sel_op_undo_stack`,由现有 `pbUndoSelectionOp`/`on_undo_selection_op_clicked` 还原)。
6. `show_segmentation(cropped)`(自动刷新 3D 闭合曲面/三平面 schedule)-> `setup_prompts()`
   (同 Apply,重建隐藏 scribble 编辑器)-> 未勾 `cbKeepCropRoi` 则删盒 -> `upload_segment_to_server()`
   (本地编辑必须同步,否则下次 prompt 的 `@ensure_synched` 用服务器旧 mask 覆盖)。
7. 全程 try/except,失败 `print("[DEBUG crop] ...")` + `showStatusMessage`。

## 兼容性

因全程基于 `get_output_volume_node()`(`get_segment_data` + `show_segmentation` 都用它)且 box
经 `_to_output_grid` 桥接,与现有 Apply 路径等价 -> 高分辨率输出、三平面模式、oblique/配准序列
自动正确。`_segment_ras_bounds` 自身已组合父级变换,无相机功能那种 GetBounds/GetRASBounds 陷阱。
