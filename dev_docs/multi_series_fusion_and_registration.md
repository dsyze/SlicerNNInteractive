# 多序列:原生序列推理 / 自动配准 / SDF 融合

> 用途:记录围绕"多个 DICOM 序列协同推理"的三项相关功能。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client,server 单会话有状态、未改动。
> 与 [[multi_plane_display_volumes]](多平面背景显示、原生序列推理面板)配套阅读。

## 背景

一次检查常有多个序列(不同对比/方向)。本组功能允许:在某个**补充序列**上跑 AI(而不是
只在主源体积上);把不同序列的结果合并(布尔同步或 SDF 平滑融合);并在序列 DICOM 参考
坐标系不一致时**自动刚性配准**到源体积,保证空间位置正确。

## 一、原生序列推理(preview-before-merge)

- UI:`cbEnableNativeSeriesInference`、`cbInferenceWorkingVolume`(工作体积选择)、
  `cbInferenceSyncMode`(Add/Replace/Subtract/Intersect)、`pbSyncInferenceResult`、
  `pbClearInferencePreview`。
- 思路:在补充序列上的 AI 结果先进**预览**(隐藏分割
  `"NativeSeriesInferencePreviewSegmentNode (do not touch)"`),用户确认后再按 sync
  模式合并进源 segment,而不是直接覆盖。
- 关键方法:`_is_native_series_inference_active`、`get_inference_volume_node`、
  `_get_or_create_inference_preview_segmentation`、`_update_inference_preview`、
  `compute_inference_sync_mask`(纯 numpy 布尔合并)、`on_sync_inference_result_clicked`、
  `on_clear_inference_preview_clicked`、`_handle_server_segmentation_result`(server 每次
  返回后:普通推理直接写;原生序列结果暂存为可编辑预览)。
- 预览 mask 存于输出网格(`_inference_result_source_mask`,名字里的 `_source` 是历史遗留,
  实际在 output 网格);`get_segment_data()` 默认也取输出网格,故 sync 两操作数形状一致。

## 二、补充序列自动配准(BRAINSFit)

- UI:`cbAutoRegisterSupplemental`、`cbConfirmSeriesAligned`、`cbRegistrationMode`
  (Rigid/Affine)、`pbRegisterSupplemental`、`pbClearAlignment`、`lblRegistrationStatus`。
- 隐藏变换节点:`SERIES_ALIGNMENT_TRANSFORM_NODE_NAME`。
- 思路:当补充序列与源体积的 DICOM Frame-of-Reference UID 不同(或用户未确认已对齐)时,
  用 BRAINSFit CLI 异步刚性/仿射配准,把线性变换挂到补充序列上;切片视图随父变换实时
  重定位。近似单位变换(`REGISTRATION_IDENTITY_*`)视为已对齐并丢弃以免无谓重采样。
- 关键方法:`_frame_of_reference_uid`、`_series_aligned`、`_ensure_alignment`(主编排:
  查缓存/查 FoR/按设置入队或阻塞)、`_enqueue_alignment` / `_pump_alignment_queue`
  (一次跑一个)、`_start_registration` / `_on_registration_status_modified` /
  `_finish_registration`、`_attach/_drop/_remove_alignment_transforms`、
  `_prune_alignment_for_source`、`_cancel_active_registration`。
- 状态:`_alignment_transforms`(dict)、`_alignment_cli_node`、`_alignment_cli_observer`
  (**手动 observer**,非 VTKObservationMixin;`_finish_registration` 与
  `_cancel_active_registration`/`cleanup()` 负责摘除)、`_alignment_queue`、
  `_alignment_in_progress`、`_alignment_pending`。
- 配准进行中,prompt/同步会被 `@ensure_synched` 阻塞,避免空间错位的交互。

## 三、多序列 SDF 融合

- UI:`cbEnableSeriesFusion`、`pbFuseSeries`(显示已累计的序列数)。
- 思路:每跑一个序列的结果就按推理体积 id 收集到 `_fusion_results`(都在输出网格);
  Fuse 时对各 mask 求符号距离场(`distance_transform_edt` 内正外负),平均后阈值 0
  得到"平滑均值面",写回当前 segment(可撤销)。
- 关键方法:`_get_fusion_enabled`、`_maybe_collect_fusion_result`(累计,网格变了就清空)、
  `_fuse_series_results`、`on_fuse_series_clicked`、`_on_series_fusion_toggled`。
- 状态:`_fusion_results`(volume_id -> uint8 输出网格 mask)、`_fusion_grid_shape`。

## 四、注意点与坑

- 三者都以"输出网格"为统一坐标基准(见 [[output_geometry_and_smoothing]]);切换源/高分辨率
  会清空融合存储与销毁预览,避免跨网格混用。
- `_alignment_cli_observer`、推理预览相关状态较多且互相耦合(server 结果处理->预览->融合
  收集),改动这条链路时务必同时考虑三者。
- 该区域保留了较多 `[DEBUG fusion.*]` 调试输出(问题排查期),未经确认勿删。
- server 单会话有状态:多序列协同仍是同一个会话,注意交互顺序。
