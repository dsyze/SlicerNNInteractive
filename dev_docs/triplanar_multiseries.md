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

## 五、修复:FOV 感知融合(2026-06-10,肱骨"切面切开、加不上"bug)

**症状**:三平面模式下某块解剖(被一个平直切面切开)用任何工具都补不进 segment,补上立刻被切掉。

**根因**:`_fuse_series_results` 对每个序列的存储掩膜做 SDF 平均。每次交互只更新当前路由
序列的掩膜(`_maybe_collect_fusion_result` 整块替换),另外两个序列仍是**交互前旧掩膜**;
在新加区域它们贡献大负 SDF,2 比 1 把改动投票否决。`_maybe_autofuse` 每次交互后全量覆盖
segment,故手动编辑也被抹。FOV 边界外的恒 0 重采样进一步制造固定平直切割面。

**第一版(A 覆盖掩膜 + B 差分传播 + n=1 合并)在真实数据上无效**。日志 `print.txt` 显示:输出网格
来自一个较大源 volume,三个斜序列各只覆盖 ~37.5%(并集 46%,53.7% 网格无任何已画序列覆盖);
每序列各存独立 mask、走各自 server 会话产生全新全量分割,SDF 平均投票让两份 mask 在重叠区互相
拉扯(在 Green 上画时反而 `removed=14423` 把 Red 加的删掉),融合结果恰等于"最后画的序列"。

**最终方案(v2):FOV 权威累积,替换 SDF 投票做实时累积**。`_handle_triplanar_result` 改为
**每个序列只在自己 FOV 内对画布有权威**:`merged = (new & cov) | (baseline & ~cov)`。
- 跨序列累积(各填各 FOV);重叠区"最后编辑者胜";无独立 mask 投票拉扯;增/减都生效
  (nnInteractive 结果已含正/负提示的累积)。n=1 自然处理(baseline 空 → `new & cov`)。
- `_maybe_autofuse`(自动 SDF 投票)删除;`_propagate_fusion_diff`(diff 传播)删除;
  `_maybe_collect_fusion_result` 回归"每序列存一份原始结果",仅供**可选的手动 Fuse** 按钮。
- 实时不再做方向加权 SDF 平滑(轻微 through-plane 阶梯);需要平滑时点手动 Fuse
  (`_fuse_series_results` 仍是覆盖掩膜版 SDF 平均)。
- **保留的辅助**:`_series_coverage_mask`/`_invalidate_fusion_coverage`(权威累积复用)、
  各失效点、`clear_current_segment` 清 store、`_log_result_resample_loss` 诊断。
- **未做(推迟 C)**:若解剖落在所有序列 FOV 外(0 覆盖),仍需把输出网格扩到各序列 bbox 并集;
  用户已确认三序列均覆盖,本次不需。

新调试前缀:`[DEBUG triplanar.merge]`、`[DEBUG fusion.cover]`。

**v3(真正修好"加不上"):输出网格覆盖三序列并集**。v2 把融合投票修对了,但用户"还是加不上"。
日志 + 确认锁定另一个独立根因:高分辨率输出网格是从 **Segment Editor 源 volume(series 29,与所画
的 19/20/21 不同)** 建的盒子(`_build_output_geometry_node`),三个所画序列伸到该盒外(覆盖仅 38%),
分割只能存在盒内,盒外部分在重采样到输出网格时被**整齐切掉** → 背景看得到骨头但分割"从来不出现",
切口是 series 29 的 FOV 平面。服务器语义已确认(`set_segment → add_initial_seg_interaction`,交互在
上传分割上累积),故跨序列累积与 v2 合并都成立,问题纯在网格盒子。
- 修法:三平面下 `_ensure_output_geometry_node` 走新分支 `_ensure_triplanar_output_geometry_node`,
  用 `GetRASBounds` 取三序列(`_triplanar_coverage_volumes`)的 **RAS 并集包围盒**,建 **RAS 轴对齐**
  等距网格(`_build_triplanar_output_geometry_node`,无父变换,32M 预算粗化),覆盖一切三视图可见解剖。
- 缓存签名 `_output_geometry_triplanar_sig=(iso, sorted series ids, bounds)`;序列重选/进出模式时
  在 `_setup_triplanar_views`/`_on_triplanar_mode_toggled` 强制重建并清 undo 栈 + 覆盖缓存。
- <2 序列回退源网格(旧行为)。新前缀 `[DEBUG triplanar.grid]`;`_log_result_resample_loss` 的 `lost`
  应显著下降。保留 v2 的 FOV 权威累积合并(并集网格下各序列覆盖率更高,合并仍需要)。

## 3D 视图切片定位线框

2D 三视图的彩色相交线由 `enable_slice_intersections()`(`SetIntersectingSlicesVisibility`)提供;
该 API 只作用于 2D 切片视图,不画到 3D。本功能在 **3D 视图**里给出对应物:进入 tri-planar 时,为
Red/Yellow/Green 各建一个**只含线段、不显示图像**的隐藏 `vtkMRMLModelNode`
(`TRIPLANAR_FRAME_NODE_NAMES`,名以 "(do not touch)" 结尾),画出该切片平面在其**自身序列** RAS 包围盒
范围内的矩形边框线,颜色取 `slice_node.GetLayoutColor()`(失败回退 `TRIPLANAR_FRAME_FALLBACK_COLORS`),
与 2D 相交线一致。
- 几何 `_make_slice_frame_polydata`:读 `GetSliceToRAS` 取归一化 u/v/n + 原点;半宽用 AABB 在 u/v 上的
  投影 `0.5*sum(|d|*extent)`;矩形中心 = 序列包围盒中心沿 n 投到当前切片平面 → 框随滚动沿法向平移、
  oblique 自动倾斜、且不随 2D 缩放变化;无背景卷退回 `GetFieldOfView`。闭合 4 点 `vtkPolyLine`。
- 显示:`SetVisibility2D(False)`(2D 已有相交线)、`SetVisibility3D(True)`、`SetLighting(False)`(线无法线)、
  `SetLineWidth(2)`、`HideFromEditorsOn`。
- 生命周期:`_enable_triplanar_slice_frames`(幂等,先 disable;建模型 + 对三 slice node 装 ModifiedEvent
  观察者,回调统一调 update)、`_update_triplanar_slice_frames`(刷新 polydata/颜色;**严禁**在此调
  `enable_slice_intersections`,否则 `Modified` 死循环)、`_disable_triplanar_slice_frames`(移除观察者 +
  删节点 + 清容器)。接线:`_setup_triplanar_views` 成功路径 enable / 序列<2 路径 disable;
  `_on_triplanar_mode_toggled` 离开分支 disable;`cleanup` disable。
- 触发为懒启动:跨会话恢复(`blockSignals` 设勾选,不触发 toggle)不建框,只在真正进入 tri-planar 时建。
  需当前布局含 3D 视图(如 Four-Up)才看得到。新前缀 `[DEBUG triplanar.frames]`。

## 关键符号

`cbTriPlanarMode`、`SETTING_TRIPLANAR_ENABLED`、`_triplanar_mode`、
`_active_inference_volume_override`、`_on_triplanar_mode_toggled`、`_setup_triplanar_views`、
`_view_for_ras_point`、`_route_prompt_to_view`、`_last_control_point_ras`、
`_mask_centroid_ras`、`_resample_result_to_output`、`_handle_triplanar_result`、
`_maybe_autofuse`、`_series_anisotropic_sigma`、`_fuse_series_results`、
`TRIPLANAR_FRAME_NODE_NAMES`、`TRIPLANAR_FRAME_FALLBACK_COLORS`、
`_make_slice_frame_polydata`、`_get_frame_color`、`_enable_triplanar_slice_frames`、
`_update_triplanar_slice_frames`、`_disable_triplanar_slice_frames`。
