# 高分辨率输出几何与平滑

> 用途:记录"高分辨率各向同性输出网格 + 粗->细插值 + 非破坏式显示平滑"三件相关功能。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client。

## 背景

源体积常常**层内精细、层间很厚**(各向异性)。直接在源网格上存分割,层间会出现台阶状
边界。本组功能提供一个与源网格解耦的**各向同性细网格**作为分割的"标准存储网格",并在
把结果写上去时做形状感知插值,得到平滑边界;另外提供纯显示层面的平滑(不改数据)。

## 一、高分辨率输出几何

- UI:`cbEnableHighResOutput`、`sbOutputSpacing`(各向同性 mm,0=自动取源最细间距)。
- 设置键:`SETTING_HIGH_RES_ENABLED`、`SETTING_OUTPUT_SPACING`。
- 隐藏几何节点:`OUTPUT_GEOMETRY_NODE_NAME`(纯几何 scalar volume),体素预算护栏
  `OUTPUT_GEOMETRY_SOFT/HARD_VOXEL_BUDGET`,间距夹在 `OUTPUT_SPACING_MIN/MAX_MM`。
- 关键方法:`get_output_volume_node`(启用时返回细网格,否则返回源体积——这是"标准输出
  网格"的单一入口)、`_output_geometry_active`、`_output_grid_shape`、
  `_ensure_output_geometry_node` / `_build_output_geometry_node`、
  `_remove_output_geometry_node`、`_disable_high_res_output`(失败时降级关闭并提示)、
  `_rebuild_output_geometry_and_migrate`(改设置后把当前 segment 重投到新网格)。
- **重要不变量**:`get_segment_data()` 默认、分割节点参考几何(`get_segmentation_node`
  里 `SetReferenceImageGeometryParameterFromVolumeNode(get_output_volume_node())`)
  都以输出网格为基准,保证布尔/同步/融合各路 mask 形状一致。

## 二、粗->细形状感知插值

- UI:`cbSmoothInterpolate`(开启会自动启用高分辨率输出,因为它需要细网格);
  `pbSmoothCurrentSegment`(对整个当前 segment 重跑一次平滑)。
- 设置键:`SETTING_SMOOTH_INTERPOLATE_ENABLED`。
- 关键方法:`_interpolate_mask_to_output_grid`(共面网格用 SDF
  `distance_transform_edt` 做层间重建,非共面用高斯)、`_to_output_grid`、
  `_smoothing_active`、`on_smooth_current_segment_clicked`、
  `_enable_high_res_for_smoothing`、`_on_smooth_interpolate_changed`。
- server 结果落地走 `_handle_server_segmentation_result` -> 按需插值 -> 写当前 segment。

## 三、非破坏式显示平滑(2D + 3D)

- UI:`cbDisplaySmooth`、`sbDisplaySmoothStrength`(0..1)、`pbBakeDisplaySmooth`。
- 设置键:`SETTING_DISPLAY_SMOOTH_ENABLED`、`SETTING_DISPLAY_SMOOTH_STRENGTH`。
- 思路:只改**显示**——用闭合曲面表示 + windowed-sinc 平滑因子,在 2D/3D 视图里渲染得
  平滑,但**不改底层二值 labelmap**(类似 Blender 的非破坏修改器)。
- 关键方法:`_apply_display_smoothing` / `_clear_display_smoothing`、
  `_current_display_smooth_strength`、`_refresh_display_smooth_ui`、
  `_reapply_display_smoothing_if_active`、`on_bake_display_smooth_clicked`
  (显式把平滑后的闭合曲面烘焙回 segment,失败自动回滚原 labelmap)。
- 作用于当前可见分割节点(`_existing_segmentation_node`),用标准表示名
  `_closed_surface_name` / `_binary_labelmap_name`。

## 四、注意点

- 三者有依赖:平滑插值依赖高分辨率输出;关掉高分辨率会联动关掉平滑插值
  (`_on_high_res_output_changed` 里有联动)。
- 切换/重建输出网格会 `_destroy_inference_preview`(避免预览停留在旧网格),也因此
  挡住会取错网格的 sync(见 [[multi_series_fusion_and_registration]])。
- 显示平滑是"显示层";导出/上传前若要让数据真平滑,需 `pbBakeDisplaySmooth` 烘焙。
