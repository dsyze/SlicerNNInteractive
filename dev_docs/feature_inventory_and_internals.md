# SlicerNNInteractive 功能清单与实现原理

> 本文是插件全部已实现功能的**全局总览**：每项功能都按「能做什么 → 解决什么问题 → 实现原理（概念）→ 关键入口」组织。
> 实现原理以**概念为主**，讲清思路与设计权衡，只点到最具代表性的函数名，细节与行号请查对应设计稿（`[[文件名]]` 链接）。
>
> **阅读指引**
> - 本文 = 一眼读完的总览 + 原理；偏用户视角的逐控件说明见 [[plugin_feature_overview]]；早期接口基线见 [[capabilities_and_interfaces]]（行号已过时，仅供参考）。
> - 每个功能的完整设计与权衡见 `dev_docs/` 下同名设计稿。
> - **全局约定**：所有功能都实现在客户端单文件 `slicer_plugin/SlicerNNInteractive/SlicerNNInteractive.py`；服务端 `server/` 与上游一致、**未新增任何接口**；几何上一切以"输出网格"为统一基准（见 1.5）。

---

## 一、整体架构（后续所有功能的地基）

### 1.1 Client / Server 分离

| 组件 | 职责 | 关键事实 |
|------|------|----------|
| **服务端** `server/` | FastAPI 包裹 `nnInteractiveInferenceSession`，执行深度学习推理 | 需 NVIDIA GPU（推荐 ≥10 GB VRAM）。进程内只有一个全局单例 `PROMPT_MANAGER`，持有**单一图像 + 单一会话**——**有状态、单会话**，多客户端并发会互相覆盖 |
| **客户端** `slicer_plugin/` | 3D Slicer 扩展；承担全部 UI、交互路由、几何转换、同步、布尔编辑、融合 | 所有二次开发功能都在这里；服务端始终是"无记忆的纯推理后端" |

**设计含义**：所有"智能"都在客户端。服务端只认"当前这张图、这个会话"，因此客户端必须自己负责"何时把最新图像/分割同步过去"（见 1.4），以及"在不污染当前会话的前提下临时借用 AI"（魔棒、Lasso 3D 等都依赖这一点）。

### 1.2 HTTP 协议与数据格式

每种交互对应一个端点（`server/.../main.py`）：

| 端点 | 请求 | 用途 |
|------|------|------|
| `/upload_image` | multipart，未压缩 `.npy`（3D 数组） | 载入图像、重置交互 |
| `/upload_segment` | multipart，gzip `.npy` 掩码 | 设置/重置初始分割 |
| `/add_point_interaction` | JSON（体素坐标 + `positive_click`） | 点提示推理 |
| `/add_bbox_interaction` | JSON（两角点 + `positive_click`） | 包围盒提示推理 |
| `/add_lasso_interaction` | multipart，gzip `.npy` 3D 掩码 + `positive_click` | 套索提示推理 |
| `/add_scribble_interaction` | 同 lasso | 涂鸦提示推理 |

**结果格式**：所有推理结果统一返回 **gzip 压缩的位打包二值掩码**（服务端 `np.packbits` 把 8 个体素压成 1 字节再 gzip）。客户端 `unpack_binary_segmentation` 逆向还原，并**按当前推理图像形状 reshape**。这样跨网络传输的分割掩码体积极小。

### 1.3 坐标顺序约定（高发 bug 区）

- numpy 体积/掩码一律 `(z, y, x)` 排列；与模型交换的体素坐标是 `(x, y, z)`。
- 点/框提示在发送前用 `xyz[::-1]` 反转坐标；套索/涂鸦传的是 3D 掩码（形状与图像一致），**无需反转**。
- 交互发生在 RAS 世界坐标，`ras_to_xyz()` 负责 RAS → 体素索引转换。
- 触碰任何坐标逻辑时，务必尊重这条 zyx ↔ xyz 的换序约定。

### 1.4 自动同步：`@ensure_synched`

**它解决的问题**：服务端无记忆，客户端又允许用户用任意方式（布尔编辑、魔棒、撤销……）改动分割。如何保证下一次 AI 提示一定基于"最新的图像 + 最新的分割"？

**原理**：装饰器包住每个提示方法（`point_prompt` / `bbox_prompt` / `lasso_or_scribble_prompt`），在真正发提示前：
1. `np.array_equal` diff **当前图像** vs 上次缓存——变了就重传图像；
2. diff **当前选中段** vs 上次缓存——变了就重传分割；
3. 两者都同步成功，才执行真正的提示。

这是一种**惰性同步**：用户在 Slicer 里怎么改都行，改动会在"下一次提示"时被自动带到服务端，客户端无需显式跟踪服务端状态。此外 `request_to_server` 还有一层兜底——若服务端报 "No image uploaded"，它会**透明重传图像+分割并重试**原请求。

> 程序化的本地编辑（布尔/魔棒/融合等）会主动调用 `upload_segment_to_server()` 立即同步，**不**等到下一次提示，否则 `@ensure_synched` 可能用服务端旧掩码覆盖刚做的本地编辑。

### 1.5 输出网格 = 统一几何基准（核心不变量）

很多功能（高分辨率输出、平滑、融合、三平面）都引入了一个"与源体积解耦的几何网格"。为避免几何错位，全插件遵守一条铁律：**一切掩码运算都以"输出网格"为基准**。

- `get_output_volume_node()` 是唯一出口：关闭高分辨率时它返回源体积，开启时返回隐藏的高分辨率/并集网格。
- `get_segment_data()` 读、`show_segmentation()` 写、所有布尔与融合，都以输出网格为参考几何。
- 任何"源网格上生成的掩码"（如 ROI 栅格化、套索填充）进入运算前，必须先经 `_to_output_grid()` 桥接到输出网格，**不能裸位运算**。
- 切换源/高分辨率网格会令旧的快照几何失配，因此切换时会**清空融合存储、销毁预览、清空撤销栈**。

### 1.6 隐藏脚手架节点与命名约定

插件用若干隐藏分割节点作内部脚手架，统一命名 `"...(do not touch)"`，并集中在 `_internal_segmentation_node_names()` 排除（保证它们不会被当成用户结构）：

| 隐藏节点 | 用途 |
|----------|------|
| `ScribbleSegmentNode` | 涂鸦 Paint 的临时画布（含 fg/bg 两段） |
| `MagicWandPreviewSegmentNode` | 魔棒 AI 预览 |
| `Lasso3dInputSegmentNode` | Lasso(3D) 用户用 Scissors 画的输入区 |
| `Lasso3dPreviewSegmentNode` | Lasso(3D) AI 细化预览 |
| `NativeSeriesInferencePreviewSegmentNode` | 原生序列推理的预览 |

另有非分割类隐藏节点：高分辨率几何 `NNInteractiveOutputGeometry`、配准变换 `NNInteractiveSeriesAlignment`、三平面定位面模型 `TRIPLANAR_FRAME_NODE_NAMES`、魔棒种子 `SelectionOpWandSeeds` 等。

### 1.7 偏好持久化与观察者管理

- **跨会话偏好**集中在 `SlicerNNInteractive/` QSettings 命名空间，通过 `_get_qsetting` / `_set_qsetting` 读写，键名以 `SETTING_*` 常量声明（服务器 URL、滚轮吸附、输出间距、段不透明度、显示平滑开关/强度、套索裁剪开关/N、多视图套索、高分辨率输出、平滑插值、三平面、全序列融合、自动相机旋转、斜位相机对齐等）。
- **观察者**多数走 `VTKObservationMixin`（`_safe_add_observer` / `_safe_remove_observer`），由 `removeObservers()` 统一回收；但有三类是手动管理、必须在 `cleanup()` 里显式摘除：**scribble 的 Paint 观察者、Lasso-3D 输入观察者、配准 CLI 观察者**。

---

## 二、核心 AI 提示交互（上游能力）

### 能做什么

| 提示 | 快捷键 | 交互方式 | 说明 |
|------|--------|----------|------|
| **点（Point）** | `O` | 切片/3D 视图放点 | 正向（加入）/ 负向（排除）极性 |
| **包围盒（BBox）** | `B` | 放两个对角点 | 正/负极性；第二点放完自动提交 |
| **套索（Lasso）** | `L` | 闭合曲线圈单层 | `Shift+L` 手动提交 |
| **涂鸦（Scribble）** | `S` | 笔刷涂抹 | 发送相邻笔画**差分**；正/负极性 |

其他：`E` 新建段、`R` 清空当前段、`T` 切换正负极性。

### 实现原理

- 点/框/套索分别由 Slicer Markups 节点驱动（Fiducial / ROI / ClosedCurve），每个节点装了 `on_placed` 观察者，放点完成即触发对应提示方法。点/框取控制点坐标转体素后反转发送；套索把闭合曲线三角化/栅格化成单层 3D 掩码后整体上传。
- **涂鸦是特例**：它不用 Markups，而是一个**隐藏的 `qMRMLSegmentEditorWidget` + Paint 效果**，在隐藏的 `ScribbleSegmentNode` 上绘制（正向画到 fg 段、负向画到 bg 段）。每次笔画结束，插件计算**与上一笔的差分掩码**只把"新增部分"发给服务端的 scribble 接口——服务端会累积笔画。这样既贴合 Paint 的连续涂抹体验，又避免重复发送整片掩码。
- 正/负极性由 `is_positive`（读正/负按钮状态）统一表达，落到各提示的 `positive_click` 参数。

#### 单层不计算模式（涂鸦直接写入）

- **能做什么**：涂鸦按钮下方的复选框 `cbScribbleDirectWrite`。勾选后涂鸦**不调用模型**，笔迹直接写进当前分段——正向涂鸦把笔迹**并入**当前段、负向涂鸦把笔迹从当前段**擦除**；"涂多少算多少"，即时、不卡顿。
- **解决什么问题**：手动精修时不希望 AI 把几笔笔迹"扩散"成它推断的整片结构；需要逐体素可控、低延迟的快速增删。
- **实现原理（原生 Paint，非差分）**：勾选后点涂鸦（厚度=1），`on_scribble_clicked` 走 `_begin_direct_write_paint`——直接激活**可见的内嵌 Segment Editor**（`self.ui.editor_widget`，绑定真实分段与全局 `SegmentEditor` 节点，即用户平时手动画分割的那个编辑器）的正向 **Paint** / 负向 **Erase** 效果，**不安装** `on_scribble_finished` 差分观察者。每一笔只改画笔扫到的体素（O(笔刷)），不再有"隐藏节点+差分+整卷 `get_segment_data` 导出+布尔+`show_segmentation` 写回+每笔 `CreateClosedSurfaceRepresentation`"那条 O(整卷) 的重链路——这正是早期差分实现"明明跳过服务端却很慢、且刷 VTK `scalar type is not a valid integer type` 警告"的根因。完全**跳过** `lasso_or_scribble_prompt` / `@ensure_synched` / 服务端 / tri-planar 路由 / 全序列融合。进入时把 `previous_states["segment_data"]` 置 None，使下一次 AI 交互的 `@ensure_synched` 检测到手动改动并重新上传分段。退出涂鸦时若 `_direct_write_paint_active` 则把可见编辑器的 active effect 清空。勾选状态变化时若涂鸦正激活会自动重挂（`_rearm_scribble_if_active`）。勾选状态持久化于 `SETTING_SCRIBBLE_DIRECT_WRITE`。
  - 注：早期版本曾尝试用**隐藏** `scribble_editor_widget` 指向真实分段并设 `SetOverwriteMode(OverwriteNone)`——该枚举在本绑定下会抛 AttributeError 被静默吞掉、且双编辑器指向同一真实节点争夺交互，导致"画了没反应"；故改用可见编辑器，不再设 OverwriteMode（Paint 默认只改当前段）。
- **配套：显示平滑重应用防抖**：`on_segmentation_modified` / `on_segment_editor_node_modified` 原本每次 `SegmentModified` 都同步 `_apply_display_smoothing()` 重建平滑 closed surface（开启 `cbDisplaySmooth` 时，原生 Paint 每一笔也会触发，是警告刷屏与卡顿的另一主因）。改为 `_schedule_display_smoothing_reapply`（token + `QTimer.singleShot`，`DISPLAY_SMOOTH_REAPPLY_DEBOUNCE_MS`=400ms 防抖），连续编辑合并为暂停后一次重建。
- **笔刷大小（mm，所有涂鸦）**：`sbScribbleBrushMm`（绝对直径毫米，默认 3mm，`SETTING_SCRIBBLE_BRUSH_MM`）。`_apply_scribble_brush_params` 统一对正常涂鸦与单层 Paint/Erase 设 `BrushUseAbsoluteSize=1 / BrushAbsoluteDiameter=<mm> / BrushSphere=0`（平的 2D 笔刷）；拖动时若涂鸦激活，`_on_scribble_brush_mm_changed` 立即对当前 effect 生效。
- **板层厚度（层，仅单层模式）**：`sbScribbleThickness`（贯穿切片方向层数，默认 1，`SETTING_SCRIBBLE_THICKNESS`）。N=1 走上面的原生即时路径；**N>1** 时 `on_scribble_clicked` 改走隐藏节点+差分路径，`on_scribble_finished` 的单层分支调用 `_apply_thick_direct_write`：`_extrude_through_plane` 把本笔增量（取其最薄的栅格轴为穿层轴）沿穿层方向 OR 扩展为 N 层板坯，必要时重采样到输出网格，再 OR/擦除合并进当前段（经 `show_segmentation`）。厚度改变若涂鸦激活会自动重挂（在原生即时路径与板坯路径间切换）。
- **关键入口/跳转**：`cbScribbleDirectWrite` / `sbScribbleBrushMm` / `sbScribbleThickness`（.ui）→ `_on_scribble_direct_write_changed` / `_on_scribble_brush_mm_changed` / `_on_scribble_thickness_changed` → `on_scribble_clicked`（厚度≤1 走 `_begin_direct_write_paint`，否则隐藏节点路径 + `on_scribble_finished` 单层分支 `_apply_thick_direct_write` / `_extrude_through_plane`）；笔刷参数 `_apply_scribble_brush_params`；退出复原 `_restore_scribble_editor_target`；显示平滑防抖 `_schedule_display_smoothing_reapply` → `_reapply_display_smoothing_debounced`。

---

## 三、切片视图行为 — [[slice_view_behaviors]]

### 能做什么

- 三个 2D 视图（红/黄/绿）显示彼此切面的**交线**，提供三维位置参照（模块加载时自动开）。
- 鼠标滚轮**吸附到原始体素网格**：每滚一步恰好前进一张真实切片（复选框 `Snap slices to voxel grid`，默认开）。
- 序列为**斜采集**时，自动把 RAS 视图旋正到序列自身的采集平面。

### 实现原理

- **滚轮吸附**：在三个标准视图的 slice node 上挂 ModifiedEvent 观察者（`_on_slice_node_modified`）。每次切片偏移变化时：用递归守卫避免 `SnapSliceOffsetToIJK` 自触发的死循环 → 比对偏移判断是不是真的滚动（过滤 FOV/旋转等其它原因的 Modified）→ 检测滚动方向 → 吸附到最近真实切片；若一步太小、吸附后仍停在同一张切片，则**强制前进一个完整体素步**再吸附。消除了插值中间帧带来的"滚不动/看着模糊"。
- **斜采集自动对齐**：`_volume_is_oblique` 看体积方向矩阵——若任一 IJK 轴对最近 RAS 轴的方向余弦 < 阈值（约偏离 2.5°）即判为斜采集，则 `_align_views_to_volume_planes` 用 Slicer 原生 `RotateToVolumePlane` 把视图旋到序列采集平面，再吸附偏移。旋正后逐张滚动即真实切片、交线与序列真实轴对齐。切换背景体积时会自动重对齐。

### 3.x 手动旋转观察平面（新，单序列）

**能做什么**：折叠面板「手动旋转观察平面（单序列）」，让用户绕世界 RAS 轴（`绕 R/A/S 轴` 下拉）手动旋转红/黄/绿切片视图到任意斜面观察各向同性数据；角度滑块（联动数值框，-180~180）；`联动旋转（红→黄/绿）` 默认勾选三视图同转，取消则只转鼠标所在视图；`复位标准正位` 还原。三平面多序列模式激活时整组灰显禁用。

**解决什么问题**：0.6mm 各向同性单序列想沿任意解剖斜面观察，而不必依赖斜采集方向矩阵或动数据。

**实现原理**：
- 纯显示——只改各视图 `vtkMRMLSliceNode` 的 `SliceToRAS` 矩阵，**绝不**碰图像数据、输出网格（1.5）或分割掩码；与高分辨率输出完全解耦。
- `rotate_slice_view` 以视图中心（`SliceToRAS` 第 4 列）为锚点，用 `vtkTransform`（平移到中心→`RotateWXYZ` 绕世界轴→平移回）经 `Multiply4x4` 左乘原矩阵，in-place 改写 `GetSliceToRAS()` 后 `UpdateMatrices()` 提交（方位自动变 Reformat）。
- **绝对角语义**：每个视图记一个基准矩阵 `_rotate_base_matrices`，滑块角度始终相对基准施加，回 0 即复位到基准；切换轴或切换联动时（`_on_rotate_axis_changed`）把当前姿态烘焙为新基准并把滑块归零。
- **滚轮兼容**：旋转过的视图记入 `_manual_rotated_views`，`_on_slice_node_modified` 对其改走「沿当前倾斜法线等距步进一个层间距、跳过 `SnapSliceOffsetToIJK`」分支（任意斜面上 IJK 吸附会产生不均匀跳动）；该分支复用现有滚轮吸附观察者，仅在 `Snap slices to voxel grid` 勾选时生效。
- **不持久化**：无 QSettings；`cleanup` 与场景 `EndCloseEvent`（`_on_scene_closed_reset_rotation`）清空状态，下次为干净正位。进入三平面模式前自动 `reset_view_to_standard` 还原，避免与自动配准冲突。

#### 3.x.1 定位线拖拽旋转 + 视角锁定（新 v2，单序列）

**能做什么**：滑块/轴下拉之外，新增可勾选按钮「🔁 定位线旋转模式」（`pbLocatorRotateMode`）。开启后在 2D 切片面板（红/黄/绿）里**抓住一条彩色交叉线左右拖动**即可旋转该线对应的切片：在某操作面里看到的两条交叉线分别属于另外两个面（红视图里的黄线＝黄切片、绿线＝绿切片），抓黄线拖 → 黄切片绕「红操作面法线」转动，**操作面本身不动**（等同 Slicer 原生「旋转切片交线」）；敏感度 1px≈0.5°；抓哪条线只转哪个面。滑块/轴下拉保留为备选。

**解决什么问题**：①滑块调角不直观，直接抓线拖更自然；②手动摆好斜面后，套索/点/框等交互会触发 `_rotate_camera_to_view`/`_align_views_to_volume_planes` 把视图打回正——需要"锁死视角"。

**实现原理**：
- **2D 拾取**：模式开启时在三个 2D 切片视图的 `sliceView().interactor()` 上以**高优先级**挂 `LeftButtonPress/MouseMove/LeftButtonRelease` 观察者（`_install_locator_rotation_observers`）。按下时用操作面 `GetXYToRAS()` 把像素 (x,y) 映射到 RAS 点 P；对另两个面取面原点+法线，用「P 到候选面的垂距」`|dot(P-Oc, N̂c)|`（=操作面上到该交叉线的距离）选最近且 ≤ `LOCATOR_PICK_TOL_MM`(3mm) 的面为目标；命中则 `SetAbortFlag(1)` 拦截默认平移并武装拖拽，未命中放行默认导航。
- **旋转（精确跟手）**：角度＝**光标相对十字交点的方位角增量**，而非旧的水平像素增量。按下时在操作面 2D 单位基（`SliceToRAS` 第 0/1 列归一化 `op_xhat/op_yhat`）里记下 `start_angle=atan2(dot(d0,op_yhat),dot(d0,op_xhat))`；拖拽中操作面自身不转、其 `XYToRAS` 恒定，把当前像素映射回 RAS 求 `cur`，`angle=cur-start_angle` 收敛到 (-π,π]（线 180° 对称，取最短弧不翻半圈）。**旋转轴用 `op_axis=op_xhat×op_yhat`（该 2D 基自身的右手法线），不是 `SliceToRAS` 第 2 列法线**——因为默认轴位/矢状的 `SliceToRAS` 是左手系（第 2 列 = −(col0×col1)），若绕第 2 列转，红视图里绿/黄线会与光标反向；用 `op_axis` 保证"在 (xhat,yhat) 里 CCW 测得的方位角"＝"`rotate_slice_view` 绕 `op_axis` 的 CCW 旋转"，对所有视图（含斜位）都 1:1 跟手。目标切片绕 `op_axis` 转 Δ 会让该交叉线也精确转 Δ。复用扩展后的 `rotate_slice_view(target, axis=操作面法线, angle, base_matrix=按下时姿态, center_ras=三面公共交点)`——`center_ras` 锚点让线绕十字交点转而非绕切片自身中心；写矩阵后 `SetOrientationToOblique()` 防 Slicer 自动回正。靠近锚点（`<LOCATOR_AZIMUTH_MIN_MM`，方位角不稳）时跳过该帧防抖。目标视图记入 `_manual_rotated_views` → 自动复用 §3.x 的滚轮逐层分支。base 只存于 `_locator_drag`、不污染滑块的 `_rotate_base_matrices`。三面公共交点由三平面方程组（Cramer）解出（`_three_plane_intersection`），退化时回退到 P。
- **视角锁定 `_view_rotation_locked`**：任何手动旋转（滑块或定位线拖拽）后置 True；`_rotate_camera_to_view` 与 `_align_views_to_volume_planes` **函数开头**早退 → 套索/点/框、装吸附观察者、应用平面背景都不再重定向视图。`reset_view_to_standard` 把视图还原标准 Ax/Cor/Sag 并 `JumpSliceByCentering` 居中、按剩余旋转视图重算锁标志（全复位即解锁）。进入三平面模式：自动取消勾选并禁用按钮（`pbLocatorRotateMode`）、清锁、复位。`cleanup`/场景关闭移除观察者并清锁。

---

## 四、3D 相机斜位对齐（新） — [[oblique_camera_alignment]]

### 能做什么

- 交互后（或点 Face Red/Yellow/Green 按钮）自动把 **3D 相机**转到当前激活序列的真实采集平面正交观察。
- 复选框 `Align 3D camera to oblique series` 在"斜位对齐 / 标准方向"间切换（默认开）。

### 实现原理

- **根因修复**：旧逻辑依赖 2D slice 已被旋到斜位，但当斜位**来自父级配准变换**（序列自身方向矩阵近单位）时，原斜位判定漏判，相机只能回到正横/冠/矢。新方案直接从**序列体积的世界方向矩阵**（含父级配准变换）算相机帧——对方向向量用"两点差法"（`TransformPoint(v) - TransformPoint(0)`）旋到世界，避免混入平移；`_has_oblique_world_frame` 因此能命中"配准造成的斜位"。
- **朝向**：相机沿**另外两个视图平面法线的叉乘**方向观察（正交三平面下等价于"正对该视图"，斜位/三序列下才不同）；退化时回退到该视图自身法线。
- **稳定性**：法线符号按固定参考侧（Red→+S、Yellow→+A、Green→−R）消歧，**反复点不会 180° 翻转**；view-up 经 Gram-Schmidt 正交化避免滚转；只改相机朝向、**保持当前焦点与距离**（不重新取景）。
- 触发入口（交互自动旋转 `_route_prompt_to_view`、手动 Face 按钮）都走 `_rotate_camera_to_view`，改造后自动生效；关闭复选框/非斜位时回退原标准方向逻辑，无回归。

---

## 五、多平面显示体积 — [[multi_plane_display_volumes]]

### 能做什么

- 红/黄/绿三视图的**背景**可分别指定不同的（已配准）序列：`Apply view volumes` 应用、`Use source volume for all` 还原。
- 解决"一次检查多个序列各自采集面清晰、其它面插值模糊"的问题——每个面看它最清晰的那个序列。

### 实现原理

- **背景与分割源体积彻底解耦**：显示用哪个序列，与分割几何基准、提示坐标基准无关（后者始终是源体积/输出网格）。
- **快照锁定**：点 Apply 时把三个选择器的节点 ID 快照进 `_plane_display_snapshot`，之后改下拉框不会悄悄生效，必须再点 Apply（避免误操作）。
- **粘性重应用**：隐藏的 Scribble / Lasso(3D) 编辑器激活时会把切片背景重置回源体积，插件检测到后自动用快照**静默重应用一次**，保证背景不被覆盖。
- 应用时对每个非源背景序列触发自动配准（见第七节）。

---

## 六、原生序列推理（preview-before-merge） — [[multi_series_fusion_and_registration]]

### 能做什么

- 在**补充序列**（而非主源体积）上跑 AI：`Analyze a supplemental series` 选工作体积。
- 结果先进**紫色预览**（不动主分割），确认后按 `Sync mode` 合并：**Add（默认）/ Replace / Subtract / Intersect**。

### 实现原理

- `get_inference_volume_node()` 决定"这次推理用哪个体积"；当推理体积 ID ≠ 源体积 ID 即判为原生序列推理。
- 服务端结果不直接写主分割，而是经 `_handle_server_segmentation_result` 暂存到隐藏预览节点（重采样到输出网格）；用户点 `Sync preview to source` 时，`compute_inference_sync_mask` 用纯 numpy 布尔按所选模式与主分割合并。
- 预览始终落在输出网格上，保证后续可平滑、可融合、可布尔。

---

## 七、补充序列自动配准（BRAINSFit） — [[multi_series_fusion_and_registration]]

### 能做什么

- 当两序列 DICOM 参考坐标系（Frame-of-Reference UID）不同时，自动把补充序列配准到源序列。
- 控件：`Auto-register...`（默认开）、`I confirm these series are already aligned`（跳过）、`Registration mode`（刚性/仿射）、`Register now`、`Clear alignment`。

### 实现原理

- `_ensure_alignment` 是总编排：先查缓存变换、再比 Frame-of-Reference；需要配准则入 **FIFO 队列**串行执行（一次只跑一个 BRAINSFit CLI），通过 CLI 观察者监控完成。
- **近似单位变换丢弃**：配准结果若平移/旋转都小于阈值（≈1 mm / 1°），视为"本就对齐"，丢弃变换避免无谓重采样。
- 配准期间，提示与同步会被阻塞，防止在空间错位状态下交互。
- 变换以"挂父变换"的方式作用于补充序列，切片视图实时跟随；变换按 (补充序列, 源体积) 对**缓存复用**。

---

## 八、高分辨率各向同性输出网格 — [[output_geometry_and_smoothing]]

### 能做什么

- 可选的各向同性细网格作为分割的标准存储基准：`Store masks on a high-resolution isotropic grid` + `Isotropic spacing (mm, 0=auto)`。
- 解决"层内细（如 0.5 mm）/ 层间厚（如 5 mm）导致分割在层间呈台阶状"的问题。

### 实现原理

- 启用后，所有掩码（AI 结果、布尔、上传同步）都以此网格为基准（即 1.5 的"输出网格"切换到细网格）。
- 网格与源体积**同向**建立（保证后续 SDF 平滑的"共面"前提），覆盖源 FOV，间距取目标值（0=自动取源最细间距）。
- **体素预算护栏**：软/硬上限，超限自动粗化间距并提示，避免内存爆炸。
- 切换间距/开关时，当前分割自动重投影到新网格，同时清空撤销栈（旧快照几何失配）。
- 三平面模式下用专门的"三序列 RAS 并集"包围盒建网格（而非源体积盒子），保证覆盖三序列全 FOV。

---

## 九、平滑插值（SDF） — [[output_geometry_and_smoothing]]

### 能做什么

- `Smooth (interpolate) result between slices`：把每次 AI 结果写入细网格时做形状感知插值，重建层间平滑边界（开启会自动开高分辨率输出）。
- `Smooth current segment`：对整个当前段手动重跑一次平滑。

### 实现原理

- **共面情形**（源与输出同轴）：对粗掩码计算**符号距离场（SDF，内正外负）**，在细网格上插值重建，零值面即平滑边界——比直接重采样更接近真实解剖的连续过渡。
- **非共面情形**（如补充序列推理结果）：退回最近邻重采样 + 高斯平滑（质量较低但稳健）。
- **作用域**：只平滑 AI 提示返回的结果与原生序列预览；手动布尔、魔棒、ROI 编辑**不经过**平滑，保持精确。
- scipy 不可用或网格过大时优雅降级到普通重采样，结果绝不丢失。

---

## 十、非破坏式显示平滑 — [[output_geometry_and_smoothing]]

### 能做什么

- `Display smooth` + 强度滑块：让分割在 2D/3D 视图里渲染得平滑，**不改底层二值掩码**。
- `Bake display smooth`：把平滑后的闭合曲面表示烘焙回 labelmap（导出前用；失败自动回滚）。

### 实现原理

- 类比 Blender 的非破坏式修改器：通过给闭合曲面表示设置 windowed-sinc 平滑因子，只影响**显示**；底层数据不变，所以可随时调强度或关掉。
- 切换段时会重新应用当前平滑设置。需要数据真实平滑（如导出）时才烘焙——烘焙把平滑曲面转回 labelmap，并在失败时回滚到原始 labelmap。

---

## 十一、多序列 SDF 融合 — [[multi_series_fusion_and_registration]]

### 能做什么

- `Enable series fusion`：每次 AI 结果自动按序列收集；`Fuse N series`：把已收集的多序列结果融合成一份平滑联合掩码（可撤销）。

### 实现原理

- 对每个序列的二值掩码分别求**符号距离场**，平均后阈值 0，得到"SDF 均值面"——比简单求并/求交更平滑、更鲁棒。
- 用 `_series_coverage_mask`（把序列网格的全 1 体积重采样到输出网格）得到各序列的 **FOV 覆盖区**，只在覆盖区内参与投票，避免某序列没拍到的区域被"投没"。
- 这是三平面方向性融合（见 14.4）的基础版本，各序列权重相等；14.4 在此之上按边界法向做方向加权。

---

## 十二、选区布尔操作 — [[semantic_selection_boolean_operations]]

### 能做什么

对已生成的分割做局部精修，无需重跑整段推理。三种布尔运算 × 四种操作数来源：

- 运算：**Add = OR / Subtract = AND NOT / Intersect = AND**（`compute_boolean_mask`）。
- 操作数来源（`cbOperandSource`）：ROI / Magic Wand / Segment / Lasso(3D)。
- 预览-再-应用、私有撤销（最多 10 步）、应用后自动同步服务端。

### 实现原理

- **网格桥接**：操作数掩码常在源网格生成，运算前一律 `_to_output_grid()` 桥接到输出网格再与当前段位运算（见 1.5），保证高分辨率/三平面/斜位下都正确。
- **ROI 操作数**（盒/球/椭球，`cbRoiShape`）：`roi_node_to_mask` 用 **OBB（有向包围盒）包含判定**栅格化——把候选体素变换到 ROI 本地坐标后按形状判定（盒：各轴绝对值 ≤ 半径；球：半径平方；椭球：归一化平方和 ≤ 1）。因此对**斜采集体积和旋转过的 ROI 都正确**，而非简单世界 AABB。
- **Magic Wand 操作数** — [[selection_operand_magic_wand]]：放多点正/负种子后，插件**临时备份服务端会话状态 → 重置 → 把种子当点提示发给 AI 取区域掩码 → 恢复服务端原状态**（不污染当前推理会话）。支持形态学生长/收缩（±体素）、预览。三平面模式下对每个分配序列各跑一次再取并集。
- **Segment 操作数**：直接读另一个分割段为掩码做布尔，自动处理参考几何对齐。
- **Lasso(3D) 操作数** — [[lasso_3d_selection]]：**双层设计**——隐藏 Segment Editor 跑原生 Scissors 让用户在 3D 视图画出实心输入块（黄色输入层），再把该输入区当 AI lasso 接口的提示，细化成预览（青色预览层），同样备份/恢复服务端状态。
- **私有撤销栈** `_sel_op_undo_stack`：因为嵌入式 Segment Editor 的历史**不记录程序化写入**（`show_segmentation` 改的掩码），所以布尔/裁剪等程序化编辑用独立的位打包快照栈（上限 10）撤销；输出网格变化时清空。

---

## 十三、几何裁剪当前分割（新） — [[crop_segment_by_box]]、[[draw_crop_region]]

两种"几何方式裁剪当前段"的工具，共享同一条安全链路：**输出网格上的纯布尔运算 → 私有撤销栈（`_record_selection_op_undo`）→ `show_segmentation` → `setup_prompts` → 立即同步服务端**；因此高分辨率/三平面/斜位/配准序列下均自动正确。区别只在"区域掩码"怎么来。两者都只删当前段体素、从不新增，也不新增 server 端点。

### 13.1 盒子裁剪（Crop Segment by Box）

**能做什么**：像 CT 三维重建软件那样转动 3D 视角、拖一个盒子框住要保留的区域，一键去除盒外部分。控件 `Place Crop ROI`、`Crop Segment by Box`、`Keep crop box after cropping`（默认裁后删盒）。

**实现原理**：本质是"当前段 ∩ 盒子掩码"，**完全复用布尔 Intersect 的网格桥接**（盒子在源网格栅格化 → `_to_output_grid` → 与输出网格当前段求交）。用**独立于 Selection Ops ROI 的专用 crop ROI 节点**（固定盒形，按当前段世界包围盒初始化，便于一拖即框住目标）。选择**清除盒外体素**而非物理裁剪参考体积——后者会破坏输出网格不变量、与三平面/高分辨率几何系统冲突、且难撤销。

### 13.2 闭合曲线 3D 裁剪（Draw Crop Region）

**能做什么**：在 **3D 视图**里左键依次放控制点画一条闭合曲线（双击/右键闭合，亮黄线 + 半透明填充），点 `Crop Inside` 或 `Crop Outside`，把当前段裁成"闭合曲线沿当前视线方向拉伸成的无限棱柱"的内部或外部——比盒子更贴合不规则轮廓。控件 `Draw Crop Region`（进入绘制）、`Cancel Drawing`、`Crop Inside`、`Crop Outside`（画好后才启用）。纯几何、不调用 AI。

**实现原理**：
- **专用闭合曲线节点** `vtkMRMLMarkupsClosedCurveNode`（限制只在第一个 3D 视图显示/绘制），独立于套索提示节点，二者互不干扰。
- **冻结视线方向**：绘制结束的瞬间（再次点 `Draw Crop Region` 收尾，或双击闭合使交互模式回到 ViewTransform 触发的自动收尾）一次性快照 3D 相机正交基 `{forward, right, up, cam_pos, parallel}` 存入 `_crop_region_camera_basis`；之后旋转视角不再影响裁剪方向（与"实时读相机"的本质区别）。
- **区域掩码 = 屏幕空间点在多边形内**（等价 `vtkImplicitSelectionLoop`，法线 = 冻结视线）：直接遍历**当前段前景体素**，用输出网格 IJK→世界矩阵（线性父变换并入矩阵、非线性逐点）把体素中心投到冻结相机的屏幕平面，再用向量化偶数-奇数射线法判定是否落在闭合曲线（稠密曲线点）多边形内。只测前景体素 → 高效；用真实 IJK→世界 → 斜位/高分辨率自动正确；平行/透视投影分别处理。
- `Crop Inside = 当前段 & 区域`，`Crop Outside = 当前段 & ~区域`。**零重叠**（区域不覆盖任何前景体素）按需求**取消不写入**并提示；无变化则提示"未改动"。裁剪后**自动退出绘制态并清空曲线**（`_exit_crop_region_drawing`）。
- 关键入口：`on_draw_crop_region_clicked` / `_finalize_crop_region_drawing` / `_capture_3d_camera_basis` / `_crop_region_to_mask`（+ `_output_ijk_to_world_matrix` / `_project_to_screen` / `_points_in_polygon`）/ `_apply_crop_region`。

---

## 十四、三平面多序列模式（新整合） — [[triplanar_multiseries]]

把前述能力串起来，服务三个**共配准**序列（如横断/矢状/冠状）。有两种交互→推理映射：默认**全序列融合**（14.6，每次交互用全部序列推理再融合）；关掉子开关则退回**按视图路由**（14.2，只用所点视图的序列）。

### 14.1 启用前提

- **先**在"多平面显示"面板把三个序列手动指给红/黄/绿并 Apply（**无自动检测**——真实数据有定位像、斜位序列，自动猜不可靠），再勾 `Tri-planar multi-series mode`。
- `_setup_triplanar_views` 据此自动开启**高分辨率输出 + 序列融合 + 三序列并集输出网格**；不同序列 < 2 个则提示用户先去多平面面板分配。

### 14.2 按视图路由交互

- `_view_for_ras_point` 判断交互控制点落在哪个切片平面，把该次推理路由到对应视图的序列：通过设置 `_active_inference_volume_override`（`get_inference_volume_node` 会优先返回它），交互结束即清除（只对该次交互有效）。
- 取点规则：点/框取末控制点 RAS、套索取首点 RAS、涂鸦取笔画质心 RAS。

### 14.3 FOV 权威累积

- 每次结果按 `merged = (new & cov) | (base & ~cov)` 合并：`new` 是新结果、`cov` 是路由序列的 FOV 覆盖区、`base` 是之前分割。含义是**新序列的改动只在自己 FOV 内有权威，重叠区"最后编辑者胜"**，避免空白序列把已有结果投没。

### 14.4 方向性融合 + 动态等权（`cbDirectionalFusion`，默认开） — [[directional_fusion]]

- **默认算法**：每个序列对融合边界的贡献，按**该处边界法向 `n` 与序列穿层方向 `t_s`** 加权：`w_s = max(1-|n·t_s|, w_floor) * cov_s`（`_directional_weighted_sdf`）。法向落在序列清晰平面内 → 满权；法向沿其穿层方向 → 近零权。于是**劣势（穿层）方向完全不投票**，而非旧法那样被模糊后仍软投票。等权与缺序列动态适应（不分主次、缺序列时剩余序列在共同方向各占等份）**自动涌现**。穿层方向由 `_series_through_plane_np_dir` 取最厚 IJK 轴投影到输出三轴 RAS 基的连续单位向量（oblique 容忍，比 snap 到单轴更准）。
- **回退**：关掉 `cbDirectionalFusion`（`SETTING_DIRECTIONAL_FUSION`）即走旧法——每个序列掩码**只沿自己的切片/层叠方向**做各向异性高斯模糊（`_series_anisotropic_sigma`，sigma 由层厚/输出间距推算，层叠方向 snap 到最近输出轴）再等权 SDF 平均、阈值 0。
- 两条算法都复用 `_series_coverage_mask`（FOV 覆盖，无数据处权重为 0）。`Fuse`、全序列融合（14.6）、一键三序列 ROI 融合（15）共享同一开关。

### 14.5 配套

- **3D 定位面 / 参考面（单+多序列通用）**：半透明矩形标出切片范围（`TRIPLANAR_FRAME_NODE_NAMES`），不透明度由 `sldPlaneOpacity` 滑块调（`SETTING_PLANE_OPACITY`），可逐视图显隐（`SETTING_TRIPLANAR_FRAME_VISIBLE_PREFIX` + 红/黄/绿三个 R/Y/G 切换）。**此功能不再绑定三平面模式**：`_ensure_triplanar_slice_frames` 已去掉 `_triplanar_mode` 门控，任一已实现视图有背景体即建；单序列下三视图同为源体、各方向各画一张，tri-planar 下各面按自身分配序列。单序列触发点：`setup()` 末尾、`get_volume_node` 自动选源体后、上传图像成功后（均经 `_schedule_ensure_slice_frames` 延迟一拍、幂等）。
- **显示分割 3D 闭合曲面**（`cbShow3DTriPlanar`，`SETTING_SHOW_3D_TRIPLANAR`）：开启后每次交互在 3D 视图重建并显示当前段的封闭曲面。重建是**防抖**的（`_schedule_triplanar_3d_surface` → `QTimer.singleShot(TRIPLANAR_3D_SURFACE_DEBOUNCE_MS=600ms)` + 自增 token 去重）——一连串快速交互只在用户停手后跑**一次**（重的 marching-cubes），避免每点一下都卡顿；若显示平滑开启则把平滑因子并入曲面转换参数。
- **交互后相机自动旋转**到激活序列采集平面（见第四节）。
- 此模式下**旁路多视图套索累积**（各视图本就是不同序列，逐视图即时路由提交即可）。

### 14.6 全序列融合（按提示自动融合，子开关 `cbAllSeriesFusion`，默认开）

- **解决什么**：14.2 的按视图路由每次只用一个序列，其他序列的高分辨率信息浪费、纵轴不准。本子模式让**每次交互都用全部已分配序列分别推理再融合**，一步写入当前段。
- **开关**：`Tri-planar multi-series mode` 下方子复选框 `cbAllSeriesFusion`（`SETTING_ALLSERIES_FUSION`，默认 True，仅三平面下可用）。取消即退回 14.2 按视图路由。`_allseries_fusion_active()` 判定生效。
- **核心** `_run_allseries_fusion(send_for_series)`（泛化自一键三序列融合）：取 `_triplanar_coverage_volumes()`（**<2 个弹窗"至少需要 2 个有效序列"并回退**）→ 进入 capture 模式（`_fusion_capture_active`，结果只存 `_fusion_capture_store` 不显示）逐序列推理（状态栏 "processing series i/n"，单序列失败跳过）→ 有效 <2 回退 → `_fusion_results` 喂给 `_fuse_series_results()`（FOV 约束 + 方向加权 SDF，落输出网格，见 1.5）→ `_record_selection_op_undo`（可撤销）→ `show_segmentation` → `upload_segment_to_server`。
- **各交互**：点/框由 RAS 良定义 → 在每序列网格重算坐标原样发；**套索/涂鸦走混合通路**（同 15 节）——来源视图序列发真提示，另两序列发从提示内部派生的点种子（`_lasso_interior_seeds` / `_mask_interior_seeds` + `_send_point_seeds_for_series`）。
- **代价**：三序列串行、各自重传，约单次 2.5-3x；累计靠"上传当前已融合段作初始掩码"实现（与 native-series 一致）。

---

## 十五、一键三序列融合（新） — [[one_click_three_series_fusion]]

### 能做什么

- 只画**一次**套索 → 点 `Fuse from 3 Series (Masked by ROI)` → 三序列分别推理 → 按 FOV 加权融合成一份平滑高分辨率分割 → 覆盖当前段、同步服务端、自动生成平滑 3D（VR）模型。
- 面向"三正交序列、面内细层间厚、FOV 不完全相同"的肩关节等场景。

### 实现原理

- **混合通路**：套索是单平面 2D 轮廓，原样发给正交序列会退化成薄片。所以——
  - **源序列**（套索所在视图对应的序列，由套索质心判定）→ 发**真套索**（精确边界）；
  - **另两正交序列** → 从套索内部派生**多个 3D 正种子点**（RAS 世界点对任意朝向都干净映射），走点提示让 AI 在各序列内生长 3D；
  - 斜位致源视图判不出时优雅退化为三序列全走点种子。
- **FOV 即有效区**：复用 `_series_coverage_mask`，不做强度阈值（避免误删压脂暗组织）；落某序列体素外的种子自动跳过（= 该序列无此处 FOV）。
- **性能**：融合的 SDF/高斯只裁剪到套索包围盒 +5 mm 的输出网格子框内计算，ROI 外保留现有段。
- **自动接管**：内部自动开启 tri-planar 模式以拿到三序列并集输出网格 + 高分辨率 + 融合收集；结果自动导出为封闭表面 3D 模型（平滑因子 0.5）。
- 操作可经私有撤销栈回退。

---

## 十六、其他增强

| 功能 | 能做什么 | 实现原理（概念） |
|------|----------|------------------|
| **套索切片范围裁剪** | `Enable lasso slice-range clipping` + 宽度 N：套索结果只保留套索平面 ±N 张切片，消除平面外溢出 | 提交时记录套索的常数切片平面（`_last_lasso_slice`），在 `show_segmentation` 中消费一次；**仅裁套索结果**，非套索提示不受影响。高分辨率输出开启时跳过（源切片索引不映射到细网格）；多视图套索下跳过（各平面不同，单平面裁剪无意义） |
| **多视图套索** | `Multi-view lasso`：跨多个视图分别画套索，各作为独立 lasso 交互提交，从多平面约束 AI | 累积每个绘制平面的套索掩码，逐个作为独立 lasso interaction 提交，让服务端会话被多视图同时约束。**三平面模式下自动旁路**（各视图已各自路由到对应序列） |
| **每段不透明度** | `Segment opacity` 滑块：调当前段不透明度 | 默认值跨会话持久化（QSettings），**新建段时自动套用**该默认值 |

---

## 十七、功能依赖关系

```
三平面多序列模式
  ├─ 依赖：多平面显示体积（视图 -> 序列映射，手动分配）
  ├─ 自动启用：高分辨率输出 + 序列融合 + 三序列并集网格
  ├─ 内部：按视图路由 + FOV 权威累积 + 方向性融合（cbDirectionalFusion）
  └─ 配套：3D 定位面/按钮 + 3D 相机斜位对齐

方向性融合 + 动态等权（cbDirectionalFusion，默认开）
  ├─ 复用：FOV 覆盖掩码 + SDF + 穿层方向（_series_through_plane_np_dir）
  ├─ 服务：_fuse_series_results（手动 Fuse / 全序列融合）+ _fuse_masks_in_roi（一键三序列）
  └─ 回退：关掉即走旧的各向异性平滑 + 等权 SDF 平均

一键三序列融合
  ├─ 自动开启：三平面模式（拿并集网格）
  ├─ 复用：FOV 覆盖掩码 + 方向性 / 方向加权 SDF 融合
  └─ 产出：覆盖当前段 + 平滑 VR 模型

手动旋转观察平面（单序列，纯显示）
  ├─ 入口：滑块/轴下拉 或 定位线拖拽模式（pbLocatorRotateMode，2D 面板抓交叉线）
  ├─ 复用：滚轮吸附观察者（旋转视图走等距步进、跳过 IJK 吸附）+ rotate_slice_view（加 center_ras 锚点）
  ├─ 锁定：手动旋转后 _view_rotation_locked=True → _rotate_camera_to_view / _align_views_to_volume_planes 早退（套索/点/框不再回正）；复位解锁
  ├─ 互斥：三平面模式激活时灰显禁用（含定位线按钮）、清锁，并先复位手动旋转
  └─ 解耦：只改 SliceToRAS，不碰图像/输出网格/掩码；不持久化（场景关闭即复位）

3D 相机斜位对齐
  └─ 增强：三平面"自动旋转 3D" + 容忍配准造成的斜位

高分辨率输出网格（统一基准）
  ├─ 前提：平滑插值（SDF 需要细网格，开平滑自动开高分辨率）
  ├─ 影响：撤销栈（网格变化即清空）
  └─ 影响：套索切片裁剪（高分辨率开启时跳过）

原生序列推理
  ├─ 依赖：多平面显示（共用工作体积面板）
  ├─ 前提：自动配准（坐标系不同时先配准）
  └─ 可选：序列融合（收集每序列结果）

选区布尔操作 / 几何裁剪（盒子 / 闭合曲线）
  ├─ 网格桥接到输出网格后位运算（不能裸 &）
  ├─ 魔棒 / Lasso(3D)：临时借用 AI（备份/恢复服务端状态）
  ├─ 闭合曲线裁剪：前景体素投到冻结相机屏幕平面 + 点在多边形内（不调用 AI）
  └─ 结果经 show_segmentation -> 立即上传同步

涂鸦「单层不计算模式」（cbScribbleDirectWrite，默认关，原生 Paint）
  ├─ 直接：隐藏编辑器指向真实分段+当前段，正向 Paint / 负向 Erase，OverwriteNone
  ├─ 不装差分观察者：每笔只改画笔体素（O(笔刷)），无整卷导出/布尔/show_segmentation
  ├─ 跳过：服务端 / @ensure_synched / tri-planar 路由 / 全序列融合
  ├─ 退出复原：_restore_scribble_editor_target 指回 scratch 节点
  └─ 同步：进入时 previous_states["segment_data"]=None → 下次 AI 交互重传分段

显示平滑重应用防抖（DISPLAY_SMOOTH_REAPPLY_DEBOUNCE_MS=400ms）
  ├─ on_segmentation_modified / on_segment_editor_node_modified 不再每笔同步重建
  └─ token + QTimer 合并连续编辑为暂停后一次 _apply_display_smoothing

逐层编辑（跨层复制 / 层间插值）
  ├─ 依赖：输出网格（取整面 / 求层号 / 轴向）+ 当前作用视图轴对齐判定
  ├─ 复用：选区运算撤销栈 + 写回链（_commit_layer_edit），共用「撤销」按钮
  └─ 层间插值复用 SDF 范式（distance_transform_edt）
```

---

## 十八、已知限制

| 限制 | 说明 |
|------|------|
| 服务端单会话 | 多用户/多客户端并发会互相覆盖 |
| 三平面逐视图重传 | 每路由到不同序列都要重传图像+分割，无法在跨视图同一会话内迭代细化 |
| 全序列融合耗时 | 每次交互对全部序列串行推理（各自重传图像+分割），约单次 2.5-3x；套索/涂鸦在正交序列退化为点种子，细节略逊于来源序列 |
| Scribble 分辨率 | 涂鸦在源体积分辨率绘制，再重采样到路由序列 |
| 涂鸦直接写入不参与融合 | 「单层不计算模式」是对当前选中段的纯手动 2D 笔刷编辑，跳过服务端与 tri-planar 路由/全序列融合；仅作用于当前选中段，且依赖正/负极性区分加/擦 |
| 板层厚度按栅格轴近似 | 厚度 N>1 时把笔迹沿「最薄的栅格轴」扩展为 N 层；强斜位/手动旋转视图下该轴只是真实穿层方向的最近栅格轴近似，板坯会贴栅格而非贴倾斜采集面。且 N>1 走差分+`show_segmentation` 整卷合并，比 N=1 的原生即时路径慢（仅在用厚板时） |
| 斜采集融合近似 | 方向性融合的穿层方向 `t_s` 由最厚 IJK 轴投影到输出三轴 RAS 基（连续向量，优于旧法 snap 单轴），但强成角下仍是近似 |
| 方向性融合法向估计 | 边界法向取各序列等权 SDF 均值的梯度；平坦区（远离边界）法向不稳，此处由 `w_floor` 兜底退化为等权，不影响内外 sign |
| BRAINSFit 依赖 | 自动配准需 Slicer 内置 BRAINSFit；非 DICOM 导入无 Frame-of-Reference UID，需手动确认对齐 |
| 显示平滑纯视图 | 不烘焙时导出的数据仍为原始 labelmap |
| 高分辨率内存 | 大视野细间距网格内存消耗大；超预算自动粗化间距 |
| 闭合曲线裁剪沿视线 | Draw Crop Region 按"绘制结束瞬间"的视线方向拉伸成无限棱柱；该方向冻结后旋转视角不改变结果。透视相机用近似投影，平行相机精确 |
| 手动旋转仅单序列 | 手动旋转观察平面只对单序列有意义，三平面模式下禁用；滚轮逐层步进依赖 `Snap slices to voxel grid` 勾选（关掉则旋转后滚轮回到 Slicer 默认连续滚动）；非联动时以鼠标所在视图为目标，鼠标不在任一切片视图上时回退红视图 |
| 视角锁定后自动回正/Face 失效 | 手动旋转（滑块或定位线拖拽）后 `_view_rotation_locked=True`，`_rotate_camera_to_view`/`_align_views_to_volume_planes` 一律早退——副作用是锁定期间「Face 红/黄/绿」按钮与三平面自动相机旋转也不生效；点「复位标准正位」解锁后恢复（设计取舍：彻底锁死视角优先） |
| 定位线拾取为近似 | 拾取用「点到候选面的垂距」近似「点到交叉线距离」，强斜面下二者有 1/sin(二面角) 的偏差（容差实际略放宽，不影响命中最近线）；阈值 `LOCATOR_PICK_TOL_MM`(3mm) 可调。注：仅**拾取判定**为近似；命中后的**旋转角已是光标相对十字交点的方位角**，被拖的线精确跟手（非旧的像素增量灵敏度） |
| 内嵌分段编辑器不随插件本地化 | 嵌入的 `qMRMLSegmentEditorWidget`（各 Effect 名、分段表头、Add/Remove 等）与 `qMRMLNodeComboBox` 的 None/节点名由 Slicer 本体提供，**跟随 Slicer 应用语言**，本插件的 `.ui` 直译与 `zh_CN.json` 改不到它们；要它们中文需把 Slicer 应用语言设为中文（见第十九节） |
| 逐层编辑仅轴对齐视图 | 跨层复制/层间插值把「层」当作输出网格上的一个 numpy 整面，故只在作用视图的切片法向贴近某输出轴（`OBLIQUE_COS_THRESHOLD`）且非手动旋转时可用，否则提示并拒绝；三平面下若序列采集面与输出网格斜交多半判为未对齐；作用视图取「鼠标所在视图→最近滚动视图→红」，滚轮吸附关闭时无滚动观察者，回退红视图（见第二十节） |

---

## 十九、界面中文本地化（i18n）

### 能做什么

- 插件自有界面（两个 Tab、各分组框、按钮、标签、下拉项、悬浮提示）以及运行时弹窗（`QMessageBox`）、状态栏消息、动态拼接的按钮文字，**无条件以中文显示**（不依赖 Slicer 应用语言设置）。

### 解决什么问题

- 团队二次开发自用，需要全中文界面；同时受 CI 约束：`check-utf8.yml` 禁止 `.py` 源码出现任何非 ASCII 字符。

### 实现原理

- **两条通路，规避 ASCII 约束**：
  - **静态界面文本**：全部在 `slicer_plugin/.../Resources/UI/SlicerNNInteractive.ui`（XML/UTF-8，不被 CI 扫描）里**直接写中文**。这是仓库既有做法（早期 `rotateViewGroup` 手动旋转那组即如此）。术语风格：中文为主，保留 `ROI/BBox/SDF/nnInteractive/RAS/DICOM` 等缩写，必要时「中文(英文)」并列。
  - **Python 运行时文本**：`.py` 不能写中文，改为「出口处统一翻译」。模块级 `_cn(text)` 按英文原文查外置表 `Resources/Strings/zh_CN.json`（UTF-8，不被 CI 扫描），缺条目则回退英文。两组包装函数把英文字面量留在原地、在显示前翻译：`_status(...)` 包 `slicer.util.showStatusMessage`；`_mb_warning/_mb_critical/_mb_information(...)` 包对应 `QMessageBox.*`。调用点由全局替换一次性接入（`slicer.util.showStatusMessage(` → `_status(` 等）；少量动态拼接（`Fuse & apply ({})`、`Submit ({})`、定位面按钮 tooltip、helpText）直接用 `_cn("...").format(...)`/`% ...` 包裹。
- **不变量**：`.py` 始终纯 ASCII（CI 通过）；中文只存在于 `.ui` 与 `zh_CN.json`。**新增任何 Python 动态用户文本时，必须同步在 `zh_CN.json` 补一条精确匹配英文原文的 key**，否则显示英文。

### 关键入口/跳转

- `_cn` / `_status` / `_mb_warning` / `_mb_critical` / `_mb_information`（`SlicerNNInteractive.py` 模块级，class 之前）
- 字符串表：`slicer_plugin/SlicerNNInteractive/Resources/Strings/zh_CN.json`
- 设计稿：[[chinese_localization]]

---

## 二十、逐层编辑：跨层复制 + 层间插值（新） — [[slice_by_slice_editing]]

### 能做什么

- **跨层复制**：「从上一层复制 / 从下一层复制」两个按钮，把当前段在相邻切片层的掩码**整面替换**到当前层（移植自 label_client 的「从上一张/下一张」继承）。
- **层间插值填充**：「层间插值填充」按钮，沿当前视图轴在所有「已画分段的关键层」之间，用形状（距离场）插值一键补齐中间的空层。

### 解决什么问题

- 插件已有 Paint/Scribble 逐层手绘，但缺「把相邻层标注拷来当起点再微调」和「只画几张关键层、中间自动补齐」这两类逐层标注的常见提速操作。

### 实现原理

- **作用视图/层/轴解析**：`_active_layer_axis()` 取作用视图（鼠标所在 `_active_slice_view_name` → 最近滚动的 `_last_active_slice_view` → 红），由 `GetSliceToRAS` 取切片法向，投影到**输出网格** IJK-to-RAS 三列（idiom 同 `_series_anisotropic_sigma`）求最近 numpy 轴 `np_axis`；切片原点经 `ras_to_xyz(..., get_output_volume_node())` 得整数层号 `layer_index`；`best_dot ≥ OBLIQUE_COS_THRESHOLD` 且非手动旋转才 `aligned`。
- **跨层复制**：读 `get_segment_data()`（输出网格 `(z,y,x)`），`new[当前层面] = pre[相邻层面]`（`_slice_plane_index` 构造沿 `np_axis` 的整面索引），替换语义。
- **层间插值**：`_fill_between_key_slices` 把轴 `np_axis` 搬到 0 轴，找出有前景的关键层，对每对相邻关键层各做 2D 有符号距离场（`scipy.ndimage.distance_transform_edt` 内-外，范式同 `_interpolate_mask_to_output_grid`），按位置 `alpha` 线性混合后阈值 `≥0` 填满中间空层。
- **统一写回**：`_commit_layer_edit` 复用选区运算同一条链路 `_record_selection_op_undo`（写前 bit-packed 整段快照）→ `show_segmentation` → `setup_prompts` → `upload_segment_to_server`，因此**共用「撤销」按钮** `on_undo_selection_op_clicked`（含输出网格形状校验），且自动满足「一切以输出网格为基准」。不新增隐藏节点、不新增服务端端点。
- **作用视图跟踪**：`_on_slice_node_modified` 在偏移真正变化时记录 `_last_active_slice_view`（普通与手动旋转两分支都记）。

### 关键入口/跳转

- `_active_layer_axis` / `_slice_plane_index` / `_commit_layer_edit` / `_copy_adjacent_layer` / `on_copy_from_prev_slice_clicked` / `on_copy_from_next_slice_clicked` / `_fill_between_key_slices` / `on_fill_between_slices_clicked`（`SlicerNNInteractive.py`）
- UI：`pbCopyFromPrevSlice` / `pbCopyFromNextSlice` / `pbFillBetweenSlices`（`Resources/UI/SlicerNNInteractive.ui`，独立的「逐层编辑」分组 `sliceEditGroup`，位于选区运算组与上传进度组之间）
- 设计稿：[[slice_by_slice_editing]]

---

> 维护提示：本文不写死行号（行号随代码漂移）。新增功能时，按"能做什么 → 解决什么问题 → 实现原理 → 关键入口/跳转"补一节，并在第十七节依赖图与第十八节限制中相应更新；逐功能细节请同步到对应 `dev_docs/*.md` 设计稿。
