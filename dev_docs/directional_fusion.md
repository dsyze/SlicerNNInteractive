# 方向性融合 + 动态等权（法向对齐置信加权）— directional_fusion

> 三序列（横断/冠状/矢状，co-registered）SDF 融合的核心算法升级。开关 `cbDirectionalFusion`（`SETTING_DIRECTIONAL_FUSION`，默认开），关闭即回退到旧的「各向异性模糊 + 等权 SDF 平均」。

## 能做什么

- 三个（或两个、一个）已配准序列分别推理后，把各序列掩膜融合成一份平滑的高分辨率分割。
- **每个序列只在自己「清晰」的方向上影响最终边界**：它的模糊（穿层/层叠）方向不再参与该处投票，避免被低分辨率轴拉偏。
- 等权、且随序列数量**动态自适应**：不分主次，缺序列时剩余序列在它们共同负责的方向上自动各占等份，不会人为产生主次关系。

## 解决什么问题

旧 `_fuse_series_results` 是「各序列 SDF 沿其厚层方向各向异性高斯模糊 → 等权累加 → 阈值 0」。问题在于：**被模糊的劣势方向 SDF 仍然参与所有方向的累加投票**（票变软但没消失），会把边界往各序列的劣势方向拉偏。

典型反例（合成验证）：三序列各自只沿自身穿层方向过分割 3 voxel。
- 旧等权 SDF 平均：Dice 0.85、体素 9603（真值 7153）——每个序列的穿层过分割都在所有方向投票，边界被撑大。
- 方向性融合：Dice 0.93、体素 8217——各序列穿层过分割被抑制，边界回弹贴近真值。

## 实现原理

核心洞察：一个序列在某处边界是否可信，由**该处边界法向 `n`** 与**该序列穿层方向 `t_s`** 的关系决定——法向落在序列清晰平面内（`n·t_s≈0`）→ 该序列看得清 → 满权；法向沿穿层方向（`|n·t_s|≈1`）→ 该处是台阶锯齿 → 近零权。这把「按方向选择序列」直接做在**标量 SDF 域**，无需梯度分解 + 泊松重建，因而没有重建的全局常数偏移歧义、无需调参。

共享核心 `_directional_weighted_sdf(items, shape, samp, w_floor=0.05)`（whole-grid 与 ROI 两条路径共用）：

1. **每序列 SDF**：`phi_s = edt(mask) - edt(~mask)`（sampling 用输出网格物理间距，内正外负）。
2. **穿层方向 `t_s`**（`_series_through_plane_np_dir`）：取序列最厚 IJK 轴（`argmax(GetSpacing())`）的 RAS 方向，投影到输出网格三个轴的 RAS 方向（`GetIJKToRASDirectionMatrix` 各列，单位正交基），得到输出 numpy 轴 (z,y,x) 基下的连续单位向量。**比旧 `_series_anisotropic_sigma` 的「snap 到最近轴」更精确**——保留连续方向，oblique 容忍。
3. **共识法向 `n_hat`**：对覆盖加权的各序列 SDF 均值 `mean_sdf` 求 `np.gradient(..., *samp)`，逐体素归一化；`|grad|≈0`（远离边界的平坦区）退化为等权。
4. **逐序列权重场**：`w_s = max(1 - |n_hat·t_s|, w_floor) * cov_s`。`w_floor` 保证每个被覆盖体素总权恒 >0（不会 0/0），并让法向模糊处优雅退化为近等权。
5. **加权平均 + 阈值**：`fused_sdf = Σ(w_s·phi_s) / Σw_s`；`fused = (fused_sdf>=0) & covered`，`covered = 任一序列 FOV 覆盖`。
6. **极端情况**：某体素所有序列均无覆盖 → `covered=False` → 保持背景 0；调用方再叠加 baseline 保留（FOV 外的手动编辑）。

**等权与动态适应是自动涌现的**，不靠手设权重：三序列齐全时，某方向有 2 个序列优势 → 各占约 50%；缺横断位时，L/R 仅冠状优势（占满）、A/P 仅矢状优势（占满）、S/I 冠状+矢状各半；只剩一个序列时它在所有方向占满。

**FOV 覆盖**复用 `_series_coverage_mask`（把序列全 1 体积重采样到输出网格，bit-packed 缓存）：某序列在某体素无数据 → 权重直接乘 0，不参与任何方向投票。

调试前缀 `[DEBUG fusion.dir]`：打印每序列 `t_np(z,y,x)`、保留体素数等。排查期勿删。

## 关键入口 / 跳转

- 核心：`_directional_weighted_sdf`、`_series_through_plane_np_dir`。
- 接线：`_fuse_series_results`（手动 `pbFuseSeries` / 全序列融合 `_run_allseries_fusion` 都经此）与 `_fuse_masks_in_roi`（一键三序列 ROI 融合 `onFuseFromThreeSeriesWithROI`）开头按开关分支；开关关闭则走各自原有的「各向异性模糊 + 等权 SDF 平均」回退路径。
- 开关：`cbDirectionalFusion` / `SETTING_DIRECTIONAL_FUSION` / `_get_directional_fusion_enabled` / `_on_directional_fusion_toggled`。
- 写回/撤销/同步：调用链不变——`on_fuse_series_clicked` 与 `_run_allseries_fusion` 已分别做 `_record_selection_op_undo` → `show_segmentation`（写当前输出网格）→ `upload_segment_to_server`。
- 相关：[[triplanar_multiseries]]、[[multi_series_fusion_and_registration]]。

## 已知限制

- **法向估计依赖共识均值**：`n_hat` 取各序列 SDF 等权均值的梯度，平坦区（远离边界）法向不稳，此处退化为等权（由 `w_floor` 兜底，不影响 sign）。
- **oblique 近似**：`t_s` 用最厚 IJK 轴投影到输出轴基；强成角下仍是近似（与旧算法同源，但保留连续方向，优于 snap）。
- 与旧算法相比每序列多一次 `np.gradient` 与权重场，内存/时间略增；ROI 路径在子框内计算，开销可控。
