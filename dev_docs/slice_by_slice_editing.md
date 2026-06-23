# 逐层编辑：跨层复制 + 层间插值

> 从 `label_client`（Godot 逐层标注客户端）移植「跨层复制/继承」，并新增其本身没有的「层间插值填充」。两者都作为当前段的逐层编辑辅助，落在插件已有的「输出网格 + 撤销栈 + 写回链 + 服务端同步」基建上。

## 能做什么

- **从上一层复制 / 从下一层复制**：把当前段在相邻切片层（沿当前视图轴 ±1）的掩码**整面替换**到当前层。典型用法：在某层画好轮廓，滚到相邻空层，一键拷过来当起点再微调。
- **层间插值填充**：沿当前视图轴，找出所有「已画分段的关键层」，在每一对相邻关键层之间用形状（距离场）插值**自动补齐中间的空层**。典型用法：只画第 A 层和第 A+10 层，一键把中间 9 层平滑补出来。

## 解决什么问题

- 插件已有隐藏 Segment Editor 的 Paint/Scribble 逐层手绘，但缺两类逐层标注常见提速操作：相邻层「拷来当起点」、关键层之间「自动补齐」。label_client 有前者（`Main.gd._copy_slice_annotations`，「从上一张/下一张」按钮），无后者。

## 实现原理

### 作用视图 / 层号 / 轴向解析（`_active_layer_axis`）
- **作用视图**：鼠标所在视图（`_active_slice_view_name`）→ 最近滚动的视图（`_last_active_slice_view`，在 `_on_slice_node_modified` 偏移变化时记录）→ 回退 `Red`。因为点击面板按钮时鼠标通常不在任何切片视图上，需要「最近滚动视图」兜底。
- **轴向 `np_axis`**：由 `slice_node.GetSliceToRAS()` 取切片法向，投影到**输出网格** `GetIJKToRASDirectionMatrix` 的三列（numpy z,y,x → IJK 列 K,J,I，`out_col_for_np_axis=(2,1,0)`），取 `abs(dot)` 最大者。此 idiom 与 `_series_anisotropic_sigma` 一致。
- **层号 `layer_index`**：切片原点（SliceToRAS 平移列）经 `ras_to_xyz(..., get_output_volume_node())` 得 `[i,j,k]`，沿轴取分量 `[i,j,k][2-np_axis]`。
- **`aligned`**：`best_dot ≥ OBLIQUE_COS_THRESHOLD`（≈cos2.5°）且视图不在 `_manual_rotated_views`。只有轴对齐时，「层」才等价于输出网格上的一个 numpy 整面，复制/插值才几何正确；否则提示「需轴对齐视图」并放弃。

### 跨层复制（替换语义，`_copy_adjacent_layer`）
- 读 `get_segment_data()`（输出网格 `(z,y,x)` bool）→ `new`；`src = layer_index ± 1`，越界则提示返回。
- `new[当前层面] = pre[源层面]`，整面替换（`_slice_plane_index` 构造沿 `np_axis` 的整面索引元组）。本层原内容由「撤销」恢复。

### 层间插值（形状插值，`_fill_between_key_slices`）
- `np.moveaxis(mask, np_axis, 0)` 把作用轴搬到 0 轴；关键层 = 该轴上有任一前景的层；< 2 个则提示「至少画两层」。
- 每对相邻关键层 `(a,b)`（`b-a>1`）：各做 2D 有符号距离场 `distance_transform_edt(m) - distance_transform_edt(~m)`（范式同 `_interpolate_mask_to_output_grid` 的 SDF 插值）；中间层 `t` 按 `alpha=(t-a)/(b-a)` 线性混合两端 SDF，阈值 `≥0` 得二值。形状插值能在两层间重建平滑过渡轮廓，而非阶梯复制。

### 统一写回（`_commit_layer_edit`）
- 复用选区运算同一条链路：`_record_selection_op_undo(target_id, pre_uint8)`（写前 bit-packed 整段快照）→ `show_segmentation(new)` → `setup_prompts()` → `upload_segment_to_server()`。
- 因此**共用现有「撤销」按钮** `on_undo_selection_op_clicked`（带输出网格形状校验），自动满足「一切以输出网格为基准」的不变量。**不新增隐藏节点、不新增服务端端点**。

## 关键入口/跳转

- 逻辑：`_active_layer_axis` / `_slice_plane_index` / `_commit_layer_edit` / `_copy_adjacent_layer` / `on_copy_from_prev_slice_clicked` / `on_copy_from_next_slice_clicked` / `_fill_between_key_slices` / `on_fill_between_slices_clicked`（`slicer_plugin/SlicerNNInteractive/SlicerNNInteractive.py`）。
- 跟踪：`_last_active_slice_view`（`__init__` 初始化、`_on_slice_node_modified` 记录）。
- UI：`pbCopyFromPrevSlice` / `pbCopyFromNextSlice` / `pbFillBetweenSlices` + `lblSliceEditHint`（`Resources/UI/SlicerNNInteractive.ui`，独立的「逐层编辑」分组 `sliceEditGroup`，位于选区运算组与上传进度组之间）；运行时中文走 `_cn` + `Resources/Strings/zh_CN.json`。

## 已知限制

- **仅轴对齐视图**：斜位 / 手动旋转视图、或三平面下序列采集面与输出网格斜交时，多半判为未对齐而拒绝（保证 numpy 整面切片几何正确）。
- **作用视图歧义**：滚轮吸附（`Snap slices to voxel grid`）关闭时无滚动观察者，`_last_active_slice_view` 不更新，未悬停切片视图直接点按钮会回退红视图。
- **替换语义**：跨层复制会覆盖本层已有标注（按用户选择；可「撤销」还原）。
- **插值仅填空隙**：只在相邻关键层之间填**空**层，不改动任何已画层；关键层本身即边界。
