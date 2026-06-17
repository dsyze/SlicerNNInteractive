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
- `_fuse_series_results`(默认走**方向性融合 + 动态等权**,见 [[directional_fusion]]):对每个序列
  的 SDF,按**该处边界法向 `n` 与序列穿层方向 `t_s`** 加权——`w_s=max(1-|n.t_s|,w_floor)*cov_s`,
  即法向落在序列清晰平面内则满权、沿穿层方向则近零权,再加权平均、阈值 0。劣势(穿层)方向**完全
  不投票**(旧法只是把它模糊后仍软投票),等权与缺序列动态适应自动涌现,不分主次。开关
  `cbDirectionalFusion`(默认开);关闭回退**旧的各向异性平滑后等权 SDF 平均**(下述)。
  - 穿层方向 `_series_through_plane_np_dir`:层叠轴 = `argmax(GetSpacing())`,取其 RAS 方向投影到
    输出网格三轴 RAS 基,得连续单位向量(比旧"snap 到最近轴"更精确,oblique 容忍)。
  - **回退路径(旧法,各向异性平滑)**:对每个序列的 SDF,仅沿其层叠方向做各向异性高斯平滑
    (`_series_anisotropic_sigma`,sigma=`clip(0.5*(层厚/该轴输出间距-1),0,3)`,把层方向 snap 到最近
    输出轴),再等权平均、阈值 0。`[DEBUG triplanar.obliq]` 打印层方向、所选输出轴、夹角(度)。
  - 注:直接用原生 spacing 当 edt `sampling` 的极性是反的(厚轴反而主导),故回退法采用"按层方向平滑"。
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

## 六、全序列融合(按提示自动融合,2026-06-16)

**动机**:第二节的"按视图路由"每次交互只用**一个**序列推理(在横断位画就只用横断序列),冠状/矢状
序列的高分辨率信息没用上、纵轴识别不精准。本节新增"全序列同时推理 + 自动融合":无论在哪个视图标注,
都用全部已分配序列分别推理,再方向加权 SDF 平均融合,一次写入当前段。

**开关**:`cbTriPlanarMode` 下方新增子复选框 `cbAllSeriesFusion`("All-Series Fusion (auto-fuse per
prompt)")。设置键 `SETTING_ALLSERIES_FUSION`(默认 True,经 `_get_allseries_fusion_enabled`/
`_on_allseries_fusion_toggled` 持久化),仅在三平面模式下可用(`_on_triplanar_mode_toggled` 里
`setEnabled(_triplanar_mode)`)。是否生效由 `_allseries_fusion_active()`(三平面开 + 子项勾选)判定。
取消勾选即退回第二节的按视图路由。

**核心例程** `_run_allseries_fusion(send_for_series)`(泛化自一键三序列 `onFuseFromThreeSeriesWithROI`):
1. `_triplanar_coverage_volumes()` 取去重后已分配序列;**<2 个** → `QMessageBox` 提示"至少需要 2 个有效
   序列"并返回 False(调用方落回按视图路由)。
2. `_fusion_capture_active=True; _fusion_capture_store={}`;循环每个序列:状态栏
   `"All-series fusion: processing series i/n..."` + `processEvents()`;设 `_active_inference_volume_override`
   = 该序列;调 `send_for_series(series)`(抛异常即跳过该序列);从 capture store 取该序列结果(非空才收集)。
   `finally` 清 `_fusion_capture_active`/override。
3. 有效结果 <2 → 状态栏提示并返回 False(回退)。
4. `self._fusion_results = {id: mask}` → `_fuse_series_results()`(自带 `_series_coverage_mask` FOV 约束 +
   `_series_anisotropic_sigma` 方向加权 + 输出网格);记 `_record_selection_op_undo`(可撤销);
   `_last_lasso_slice=None`(融合是完整 3D,不被 lasso 切片裁剪);`show_segmentation` → 清 store →
   `upload_segment_to_server`(同步服务端);返回 True。

**结果捕获**:复用 `_handle_server_segmentation_result` 既有 `if _fusion_capture_active:` 分支
(`_resample_result_to_output` 到输出网格,按推理体积 id 存入 `_fusion_capture_store`,**不显示**)。
故 prompt 方法(point/bbox/lasso/scribble)本身无需改动结果分流。

**四类交互接入**(均在原按视图路由分支**之前**插入 `if _allseries_fusion_active(): if _run_allseries_fusion(cb): return`):
- **point**:`send_for_series = 在该序列网格重算 xyz 后 point_prompt`(点在任意网格良定义,原样发三序列)。
- **bbox**:两角点先存 RAS(新增 `self.prev_bbox_ras`,首角点记下),第二角点完成时回调在每序列网格用
  `ras_to_xyz` 重算"中心点 + 一角"再 `bbox_prompt`(原样发三序列)。
- **lasso / scribble —— 混合模式**:平面提示在正交序列上退化,故**来源视图序列发真实 lasso/scribble,
  另两序列发从提示内部派生的正点种子**(`_lasso_interior_seeds` / 新 `_mask_interior_seeds` +
  `_send_point_seeds_for_series`,无 FOV 内种子则跳过)。来源序列由 `_view_for_ras_point(质心)` 定。

**辅助重构**:`_lasso_interior_seeds` 把"掩膜→距离变换→取最深内部点+若干采样→IJK→RAS"核心抽成
`_mask_interior_seeds(mask, ref_grid_node, n_seeds)`,套索(先 fill 多边形)与涂鸦(直接传掩膜)共享。

**性能**:三序列串行,每序列各自重传图像/分割(`@ensure_synched` 自动),约单次推理的 2.5-3x;状态栏
显示"processing series i/n"。**累计语义**:循环切序列会重置 server 会话,但每序列以"当前(已融合)分割段"
为初始掩膜上传,故跨多次点击的累计靠分割段种子实现(与 native-series / fuse3 一致)。

**限制**:lasso/scribble 在正交序列退化为点种子(非原始轮廓),细节略逊于来源序列;<2 序列每次提示都会
弹窗(配置问题,可接受)。新调试前缀 `[DEBUG allseries]`。

## 3D 视图切片定位面（含左上角 R/Y/G 开关）

2D 三视图的彩色相交线由 `enable_slice_intersections()`(`SetIntersectingSlicesVisibility`)提供;
该 API 只作用于 2D 切片视图,不画到 3D。本功能在 **3D 视图**里给出对应物:进入 tri-planar 时,为
Red/Yellow/Green 各建一个隐藏 `vtkMRMLModelNode`(`TRIPLANAR_FRAME_NODE_NAMES`,名以 "(do not touch)"
结尾),画出该切片平面在其**自身序列** RAS 包围盒范围内的**半透明填充面 + 彩色边框**,颜色取
`slice_node.GetLayoutColor()`(失败回退 `TRIPLANAR_FRAME_FALLBACK_COLORS`),与 2D 相交线一致。
- 几何 `_make_slice_frame_polydata`:读 `GetSliceToRAS` 取归一化 u/v/n + 原点;半宽用 AABB 在 u/v 上的
  投影 `0.5*sum(|d|*extent)`;矩形中心 = 序列包围盒中心沿 n 投到当前切片平面 → 面随滚动沿法向平移、
  oblique 自动倾斜、且不随 2D 缩放变化;无背景卷退回 `GetFieldOfView`。同一 polydata 含 `vtkPolygon`
  (填充面,`SetPolys`)+ 闭合 `vtkPolyLine`(边框,`SetLines`)。
- 显示:`SetOpacity(TRIPLANAR_FRAME_OPACITY=0.25)`、`SetBackfaceCulling(False)`(双面)、
  `SetVisibility2D(False)`、`SetVisibility3D(True)`、`SetLighting(False)`、`SetLineWidth(2)`、`HideFromEditorsOn`;
  逐视图可见性 `SetVisibility(self._get_frame_visible(view))`。
- 每面独立显隐:`_get_frame_visible`/`_set_frame_visible`(`SETTING_TRIPLANAR_FRAME_VISIBLE_PREFIX`+小写视图名,
  cast=bool,默认 True,跨会话持久化)。
- 左上角 R/Y/G 开关:`_create_triplanar_frame_buttons` 把一个透明容器 + 三个 `qt.QToolButton` 叠到
  `layoutManager.threeDWidget(0).threeDView()`(`move(8,8)`+`raise_()`);`_style_frame_button` 用 stylesheet
  亮(实色)/暗(深灰+彩字)表示显/隐;`_on_frame_button_toggled` 写设置 + 设 display 可见性 + restyle +
  `_force_render_3d_views`;`_destroy_triplanar_frame_buttons` 用 `setParent(None)+deleteLater()` 清理。
- 渲染:`_force_render_3d_views`(`threeDWidget(i).threeDView().forceRender()`)在 enable/update/toggle 后强制
  重绘——程序化设置 polydata 不会自动触发 3D 重绘(否则"要再点一下才出现")。
- 生命周期:`_enable_triplanar_slice_frames`(幂等,先 disable;建模型 + 对三 slice node 装 ModifiedEvent
  观察者,回调统一调 update;末尾建按钮)、`_update_triplanar_slice_frames`(刷新 polydata/颜色/可见性;
  **严禁**在此调 `enable_slice_intersections`,否则 `Modified` 死循环)、`_disable_triplanar_slice_frames`
  (移除观察者 + 删节点 + 销毁按钮 + 清容器)。接线:`_setup_triplanar_views` 成功路径 enable / 序列<2 路径
  disable;`_on_triplanar_mode_toggled` 离开分支 disable;`cleanup` disable。
- 触发:`_ensure_triplanar_slice_frames(reason)`(模式开 + 未建 + R/Y/G 中 >=2 个不同序列才建,已建则 no-op,
  几何由观察者实时刷新)。挂三处:`_apply_plane_display_volumes` 成功处(覆盖显式 Apply 与粘性 reapply)、
  `_handle_triplanar_result` 开头(交互兜底)、`_setup_triplanar_views` 末尾。这样跨会话**恢复勾选**的会话
  (`blockSignals` 设勾选不触发 toggle)在用户分配序列/Apply 时也能自动建面,无需手动关再开复选框。
  需当前布局含 3D 视图(如 Four-Up)才看得到面与按钮。新前缀 `[DEBUG triplanar.frames]`。
  已知限制:按钮 attach 到进入时的 `threeDWidget(0)`,进入后切布局会丢按钮,重勾选 tri-planar 重建。
- 不透明度:面板滑块 `sldPlaneOpacity`(0..100)→ `_on_plane_opacity_changed` 统一设三面 display `SetOpacity`
  并持久化 `SETTING_PLANE_OPACITY`;`_enable_triplanar_slice_frames` 建面时用 `_get_plane_opacity()`(默认
  `TRIPLANAR_FRAME_OPACITY`)。
- 几何共享:`_slice_frame_geometry(slice_node, volume)` 返回 `(center,u,v,n,hu,hv)`,被 `_make_slice_frame_polydata`
  与相机旋转共用。

## 点击序列 -> 3D 相机自动旋转正对该序列

`_rotate_camera_to_view(view_name)`:用 `_slice_frame_geometry` 取该视图定位面的 center/法向 n/半宽;相机
**沿切片平面法向**(oblique 自适应)观察,焦点=center,距离 `1.3*max(hu,hv)/tan(viewAngle/2)` 容纳整面;
法向符号按 `TRIPLANAR_CAMERA_PREFERRED_SIDE`(Red→Superior、Yellow→Anterior、Green→Left)消歧,使视点符合
"上/前/左"惯例;view-up 取切片竖直轴(偏 Superior,近轴向退偏 Anterior)。相机节点经
`slicer.modules.cameras.logic().GetViewActiveCameraNode(threeDWidget(0).mrmlViewNode())` 取得,设
`SetFocalPoint/SetPosition/SetViewUp` 后 `ResetCameraClippingRange` + `_force_render_3d_views`。
- 触发:`_route_prompt_to_view` 命中视图后,若 `_get_auto_camera_rotation()`(`SETTING_AUTO_CAMERA_ROTATION`,
  复选框 `cbAutoRotateCamera`,默认 On)则旋转——覆盖 point/bbox/lasso/scribble 四类交互(单点钩子)。
- 手动:面板按钮 `pbFaceRed/pbFaceYellow/pbFaceGreen` 直接调 `_rotate_camera_to_view`,不受总开关影响。

## tri-planar 自动 Show 3D（分割闭合曲面，防抖）

`show_segmentation` 在 tri-planar 下故意跳过每次交互的闭合曲面重建(大融合网格上 marching cubes 很重)。
改为:`with RenderBlocker()` 块后,若 `cbShow3DTriPlanar`(`SETTING_SHOW_3D_TRIPLANAR`,默认 On)开启则
`_schedule_triplanar_3d_surface()` —— 用 `qt.QTimer.singleShot(TRIPLANAR_3D_SURFACE_DEBOUNCE_MS=600, ...)` +
自增 token 防抖:连续快速交互各自排程,仅最后一个(停手 600ms 后)token 匹配,`_rebuild_triplanar_3d_surface`
才真正 `RemoveRepresentation(_closed_surface_name()) → CreateClosedSurfaceRepresentation() →
SetVisibility3D(True)/SetSegmentVisibility3D(seg_id,True)`(display-smooth 开时先设 Smoothing factor)。
开关勾上立即排一次、取消则 `SetVisibility3D(False)`;`_setup_triplanar_views` 成功路径也排一次显示已有分割。

## 关键符号

`cbTriPlanarMode`、`SETTING_TRIPLANAR_ENABLED`、`_triplanar_mode`、
`_active_inference_volume_override`、`_on_triplanar_mode_toggled`、`_setup_triplanar_views`、
`_view_for_ras_point`、`_route_prompt_to_view`、`_last_control_point_ras`、
`_mask_centroid_ras`、`_resample_result_to_output`、`_handle_triplanar_result`、
`_maybe_autofuse`、`_series_anisotropic_sigma`、`_fuse_series_results`、
`cbAllSeriesFusion`、`SETTING_ALLSERIES_FUSION`、`_get_allseries_fusion_enabled`、
`_on_allseries_fusion_toggled`、`_allseries_fusion_active`、`_run_allseries_fusion`、
`_fusion_capture_active`、`_fusion_capture_store`、`_mask_interior_seeds`、
`_lasso_interior_seeds`、`_send_point_seeds_for_series`、`prev_bbox_ras`、
`TRIPLANAR_FRAME_NODE_NAMES`、`TRIPLANAR_FRAME_FALLBACK_COLORS`、`TRIPLANAR_FRAME_OPACITY`、
`SETTING_TRIPLANAR_FRAME_VISIBLE_PREFIX`、`_make_slice_frame_polydata`、`_get_frame_color`、
`_get_frame_visible`、`_set_frame_visible`、`_ensure_triplanar_slice_frames`、`_enable_triplanar_slice_frames`、
`_update_triplanar_slice_frames`、`_disable_triplanar_slice_frames`、`_force_render_3d_views`、
`_style_frame_button`、`_create_triplanar_frame_buttons`、`_on_frame_button_toggled`、
`_destroy_triplanar_frame_buttons`、`_slice_frame_geometry`、`SETTING_PLANE_OPACITY`、`_get_plane_opacity`、
`_on_plane_opacity_changed`、`sldPlaneOpacity`、`SETTING_AUTO_CAMERA_ROTATION`、`cbAutoRotateCamera`、
`_get_auto_camera_rotation`、`_on_auto_camera_rotation_toggled`、`_rotate_camera_to_view`、
`TRIPLANAR_CAMERA_PREFERRED_SIDE`、`pbFaceRed`/`pbFaceYellow`/`pbFaceGreen`、`SETTING_SHOW_3D_TRIPLANAR`、
`cbShow3DTriPlanar`、`_get_show_3d_triplanar`、`_on_show_3d_triplanar_toggled`、`_schedule_triplanar_3d_surface`、
`_rebuild_triplanar_3d_surface`、`TRIPLANAR_3D_SURFACE_DEBOUNCE_MS`。
