# 一键三序列融合分割（Fuse from 3 Series, Masked by ROI）

状态：已实现（客户端，纯 `slicer_plugin/SlicerNNInteractive/SlicerNNInteractive.py` + 一个 UI 按钮）。服务端未改动。

## 目标

肩关节 MR 三正交序列（横断 PD 压脂 / 冠状 T2 压脂 / 矢状 T2 压脂），面内 0.5mm、层距 3mm，且三序列 FOV 不完全相同。
用户**只画一次套索** → 一键让三序列分别推理 → 按 FOV 有效区域智能融合成一份平滑高分辨率分割 → 覆盖当前段、同步服务端、自动生成平滑 3D 表面（VR）模型。

## 关键设计决策（已与用户确认）

1. **提示语义 = 混合通路（B）**：套索是单平面 2D 轮廓，原样发给正交序列会退化成薄片。故：
   - **源序列**（套索所在视图平面对应的序列，用 `_view_for_ras_point(套索质心)` 判定）→ 发**真套索**（精确边界）。
   - **另两个正交序列** → 从套索内部派生 **多个 3D 正种子点**（RAS 世界点对任意朝向干净映射），走 `point` 提示。
   - 斜位致源视图判不出时（`src_series=None`），三序列全部走点种子（优雅退化为"纯点"）。
2. **种子 = 多点正种子**：套索在输出网格上填充 → 距离变换取最深点 + 内部等距采样，共 `FUSE3_NUM_SEEDS=5` 个正种子；落某序列体素边界外的种子自动跳过（= 该序列无该处 FOV）。
3. **有效区域 = FOV**：复用 `_series_coverage_mask`（把序列网格 ones 重采样到输出网格），不做强度阈值（避免误删压脂暗组织）。
4. **按钮独立**：内部**自动开启 tri-planar 模式**（`cbTriPlanarMode.setChecked(True)`）以获得**并集输出网格**（覆盖三序列、防裁剪，见 commit c8986ce）+ 高分辨率 + 融合收集；用户只需在多平面面板分配红/黄/绿序列。
5. **VR = 封闭表面 3D 模型**：`CreateClosedSurfaceRepresentation` + 平滑因子 0.5 + 打开 3D 可见。
6. **性能**：融合的 SDF/高斯计算裁剪到套索包围盒 +5mm（`FUSE3_ROI_MARGIN_MM`）的输出网格子框；ROI 外保留现有段。逐序列结果重采样与 FOV 覆盖掩膜仍全网格（带缓存）。
7. **约束**：`.py` 纯 ASCII（`check-utf8.yml`），故所有 Python 文案为英文；`.ui` 按钮文案英文 `Fuse from 3 Series (Masked by ROI)`。

## 新增代码（均在 `SlicerNNInteractiveWidget`）

- 常量：`FUSE3_NUM_SEEDS`、`FUSE3_ROI_MARGIN_MM`。
- 状态（`__init__`）：`_fusion_capture_active`、`_fusion_capture_store`、`_last_lasso_world_points`。
- `submit_lasso_if_present`：缓存最后一次套索世界点到 `_last_lasso_world_points`。
- `_handle_server_segmentation_result` 顶部：捕获分支——`_fusion_capture_active` 时把结果重采样到输出网格、按推理体积 id 存入 `_fusion_capture_store`、直接返回（不显示）。
- `_get_active_lasso()`：live 节点优先，回退缓存；返回 (N,3) RAS。
- `_lasso_interior_seeds(lasso_ras, ref_grid, n_seeds)`：派生多点正种子（RAS）。
- `_send_point_seeds_for_series(series, seeds_ras)`：逐个发 `point_prompt`（越界跳过），返回是否至少发出一个。
- `_get_lasso_roi_world(points, margin_mm)` + `_ras_box_to_output_index_box(mn,mx)`：套索 ROI → 输出网格索引子框。
- `_get_series_valid_region(seriesNode, refGridNode)`：委托 `_series_coverage_mask`。
- `_fuse_masks_in_roi(valid, roi_box, output_volume)`：ROI 内 FOV 加权有向 SDF 融合，ROI 外保留现有段。
- `_export_to_vr(segmentId, smoothing=0.5)`：封闭表面 + 平滑 + 3D 可见。
- `onFuseFromThreeSeriesWithROI(checked=False)`：主编排。
- UI：`pbFuseThreeSeriesRoi` 按钮（`Resources/UI/SlicerNNInteractive.ui`，紧随 `pbFuseSeries`）+ `setup` 连接。

## 复用的现有函数

`_view_for_ras_point` / `_view_background_volume` / `get_inference_volume_node` /
`_active_inference_volume_override` / `point_prompt` / `lasso_or_scribble_prompt` /
`lasso_points_to_mask` / `ras_to_xyz` / `_series_coverage_mask` /
`_series_anisotropic_sigma` / `_resample_result_to_output` /
`get_output_volume_node` / `_output_grid_shape` / `get_segment_data` /
`show_segmentation` / `upload_segment_to_server` / `_record_selection_op_undo` /
`get_current_segment_id` / `get_selected_segmentation_node_and_segment_id` /
`_closed_surface_name` / tri-planar 并集网格（`_ensure_triplanar_output_geometry_node`）。

## 数据流

```
单套索(RAS) --+--> [源序列]  ras->xyz -> lasso_points_to_mask -> lasso_or_scribble_prompt --+
              |                                                                              |--> 捕获到 _fusion_capture_store[series_id] (输出网格)
              +--> _lasso_interior_seeds -> [另两序列] point_prompt x N (越界跳过) ----------+
                                                                                             v
   _fuse_masks_in_roi: 每序列 SDF(各向异性模糊) * FOV覆盖 累加 / 计数 -> (sum>=0 & cnt>0) -> ROI内融合
                                                                                             v
   show_segmentation(final) -> upload_segment_to_server -> _export_to_vr(0.5)
```

## 调试输出

统一前缀 `[DEBUG fuse3]`（未确认问题解决前保留，遵守仓库约定）。

## 端到端验证

1. 加载三套已配准序列；多平面面板把横断/矢状/冠状指给 Red/Yellow/Green 并 Apply。
2. 任一视图用套索（L）圈目标一次。
3. 点 "Fuse from 3 Series (Masked by ROI)"：源序列走 lasso、另两走多点种子；当前段被 FOV 加权 SDF 融合覆盖；ROI 外原段保留；3D 视图出现平滑模型。
4. FOV：仅横断覆盖区（如胸锁关节）画套索 → 矢状/冠状种子落 FOV 外被跳过、该区不被空白序列投没。
5. 反例：未分配视图→警告；无套索→提示；<2 序列出结果→放弃并警告。
6. 撤销：经 `_record_selection_op_undo` 可回退。

## 已知权衡 / 后续可优化

- 种子派生与 FOV 覆盖、结果重采样仍在全输出网格上（仅融合 SDF 裁剪到 ROI）；如需更快可把这些也裁剪到 ROI。
- 套索填充在 RAS 轴对齐并集网格上是近 1 体素厚的平面，距离变换的"最深点"语义弱化为"任一内部点"，但等距采样仍能在套索面内均匀撒点，作为正种子有效。
- 点种子均共面（套索平面），由 nnInteractive 在各序列内生长 3D；若发现欠分割，可调大 `FUSE3_NUM_SEEDS` 或加入负种子。
