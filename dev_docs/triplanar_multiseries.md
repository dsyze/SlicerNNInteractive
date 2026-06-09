# 三视图多序列统一交互 + 方向加权融合

> 用途:记录"三视图多序列模式"的设计与实现。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client,server 未改动。
> 与 [[multi_series_fusion_and_registration]]、[[output_geometry_and_smoothing]]、
> [[slice_view_behaviors]]、[[multi_plane_display_volumes]] 配套阅读(本功能在其之上整合)。

## 背景与目标

三个**同空间、已配准**的序列(横断/冠状/矢状)。原先交互只作用于单一 working volume,
多序列融合需手动切 working volume + 手动 Fuse + 均匀 SDF 平均。本模式把它整合为:
- **任一视图都可交互**,且自动作用于该视图所显示的序列(把三序列当成一个整体);
- **方向加权融合**:每个序列在其最清晰平面主导边界、抑制厚层台阶锯齿;
- **每次交互后自动融合并显示**。

## UI 与开关

- 复选框 `cbTriPlanarMode`("Tri-planar multi-series mode",在 Native-series 推理分组内)。
- 设置键 `SETTING_TRIPLANAR_ENABLED`(经 `_get_qsetting`/`_set_qsetting` 持久化;启动时只
  恢复勾选状态、**不**自动重排视图,避免在加载序列前动布局)。
- 入口:`_on_triplanar_mode_toggled` -> 开启时 `_setup_triplanar_views()`。

## 一、序列 -> 视图(手动分配)

- **用户手动选序列**:在"多平面显示"面板把 3 个序列分别指给 红=横断 / 黄=矢状 / 绿=冠状
  (Slicer 标准约定)并 Apply。**不做自动方向检测**——真实数据常含定位图等多余序列、且多为
  oblique(成角采集),自动挑选不可靠(实测教训:8 个体积里 7 个 oblique、定位图被拆成 3 个)。
- `_setup_triplanar_views()`:只"锁定 + 启用前置",不分类不扫描:
  - 调 `on_apply_plane_display_volumes_clicked()` 把用户当前红/黄/绿下拉选择应用为视图背景;
  - 启用高分辨率各向同性输出(`_enable_high_res_for_smoothing`)与序列融合(`cbEnableSeriesFusion`);
  - 用 `_view_background_volume` 校验三视图背景:**不同序列<2 个**则提示用户先去多平面面板指定
    (`[DEBUG triplanar.setup]` 打印三视图背景序列名)。
- 不依赖"横/冠/矢"标签:路由只看 `view -> _view_background_volume(view)`;oblique 序列的视图朝向
  由现有 `_align_views_to_volume_planes` 各自旋正。

## 二、按视图路由(交互 -> 该视图序列)

- **视图判定**`_view_for_ras_point(ras)`:放置点必然落在所点击视图的切片平面上;遍历
  `_iter_standard_slice_logics()`,用各 slice node 的 `GetSliceToRAS()`(原点+法向)算点到
  平面距离,取容差内最近的视图(3D 视图等无匹配返回 None)。
- **覆盖机制**:`get_inference_volume_node()` 在 `_active_inference_volume_override` 非空时
  优先返回它(override 为空时零行为变化)。`_route_prompt_to_view(ras)` 设置 override 为
  `_view_background_volume(view)` 得到的序列。
- **各交互接入**(均在计算体素坐标/栅格化**之前**路由,`finally`/出口清 override):
  - point:`on_point_placed` 用最后控制点 RAS(`_last_control_point_ras`)。
  - bbox:`on_bbox_placed` 同上(两角点同面同序列)。
  - lasso:`submit_lasso_if_present` 用曲线首点 RAS;**三视图下旁路 multi-view 累积**
    (`multiview_effective = 多视图勾选 and not 三视图`),逐视图即时提交并路由。
  - scribble:`on_scribble_finished` 用笔画质心 RAS(`_mask_centroid_ras`);diff mask 在
    源网格,`lasso_or_scribble_prompt(mask_volume_node=source)` 会自动重采样到路由序列。

## 三、方向加权融合(各向异性平滑后 SDF 平均)

- `_handle_server_segmentation_result` 顶部分支:三视图模式下走 `_handle_triplanar_result`
  -> `_resample_result_to_output`(结果重采样到输出网格)-> `_maybe_collect_fusion_result`
  (按路由序列 id 累积)-> `_maybe_autofuse`(>=2 序列时融合显示;不足时直接显示单序列)。
- `_fuse_series_results`(已升级):对每个序列的 SDF,**仅沿其"层叠/采集方向"做各向异性高斯
  平滑**(`_series_anisotropic_sigma`),抹掉该方向台阶锯齿、保留面内锐度;再平均、阈值 0。
  这样某方向上锯齿的序列被该方向平滑、让在该方向清晰的序列主导,实现"各方向取最清晰"。
  - **oblique 容忍(选项 A)**:层叠轴 = `argmax(GetSpacing())`;取其 RAS 方向投影到**最近的输出
    网格轴**(|dot| 最大),沿该轴设 sigma=`clip(0.5*(层厚/该轴输出间距-1),0,3)`,其余 0。成角序列
    也能去锯齿(近似到主轴)。`[DEBUG triplanar.obliq]` 打印层方向、所选输出轴、夹角(度),据此判断
    近似是否足够;不够可升级 C(旋转-模糊-转回,仅动此处)。
  - 注:直接用原生 spacing 当 edt `sampling` 的极性是反的(厚轴反而主导),故采用"按层方向平滑"。
- `_maybe_autofuse` 每次融合记一次 undo(与手动 `pbFuseSeries` 一致,栈上限 10)。

## 四、注意点 / 限制

- **已配准前提**:序列须同 frame-of-reference(或勾选 `cbConfirmSeriesAligned`);否则退回
  现有自动配准逻辑(`_ensure_alignment`)。本模式不强制 BRAINSFit。
- **server 单会话**:逐视图路由会在每次交互重传该序列图像并重置交互链(各序列结果独立,
  适合融合;但无法跨视图在同一 server 会话里迭代细化)。
- **scribble 分辨率**:scribble 编辑器仍绑源体积,笔画在源网格分辨率绘制,再重采样到路由
  序列(AI 会精修);未按视图序列分辨率绘制。
- **倾斜序列**:序列选定靠手动(不自动分类);方向加权融合对 oblique 用"近似到最近输出轴"
  (选项 A),强成角下可按需升级到 C。
- 启动时不自动重排视图(需用户开启模式触发 `_setup_triplanar_views`)。
- 保留该区域 `[DEBUG fusion]` 等调试输出(排查期),未经确认勿删。

## 关键符号

`cbTriPlanarMode`、`SETTING_TRIPLANAR_ENABLED`、`_triplanar_mode`、
`_active_inference_volume_override`、`_on_triplanar_mode_toggled`、`_setup_triplanar_views`、
`_view_for_ras_point`、`_route_prompt_to_view`、`_last_control_point_ras`、
`_mask_centroid_ras`、`_resample_result_to_output`、`_handle_triplanar_result`、
`_maybe_autofuse`、`_series_anisotropic_sigma`、`_fuse_series_results`。
