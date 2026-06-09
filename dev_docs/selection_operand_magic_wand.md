# Magic Wand 操作数(Selection Operations)

> 用途:记录"魔棒"操作数的设计与实现,供后续维护。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client
> `slicer_plugin/SlicerNNInteractive/SlicerNNInteractive.py`,server 未改动。

## 一、它是什么

Magic Wand 是 [[semantic_selection_boolean_operations]] 里 Selection Operations 的四种
操作数来源之一(`cbOperandSource` 选中 `OPERAND_SOURCE_WAND`)。用户在视图里放若干
**种子点**(正/负),插件调用 nnInteractive 的 point 交互在当前推理图像上生成一个区域
mask,作为布尔操作(Add/Subtract/Intersect)的 operand。它本质是"用 AI 点提示快速圈出
一块区域,再并/减/交到当前 segment"。

## 二、UI

- `pbPlaceWandSeed` / `pbClearWandSeed`:进入放置模式 / 清空种子。
- `pbPreviewWand` / `pbClearPreviewWand`:按需预览 / 清预览。
- `sbGrowShrinkWand`:对结果做形态学生长(正)/收缩(负),单位体素,范围约 -20..20。
- 种子的正负由全局 prompt 极性(`is_positive`,Positive/Negative 按钮)决定。

## 三、关键方法

| 方法 | 作用 |
|------|------|
| `_get_or_create_wand_seed` | 取/建种子 Fiducial 节点 `SelectionOpWandSeeds`,挂放置 observer |
| `_collect_wand_seeds` | 读取种子,RAS->IJK(`ras_to_xyz`),过滤越界,返回 `(voxel_coord, is_positive)` |
| `_compute_magic_wand_mask` | 核心:备份当前 server 交互状态 -> 重置 -> 逐个发送种子 point 交互 -> 取回累计 mask -> 恢复;失败返回 None |
| `_postprocess_wand_mask` | 用 `scipy.ndimage` 做 grow/shrink(`sbGrowShrinkWand`);scipy 不可用时优雅跳过 |
| `_get_or_create_wand_preview_segmentation` | 取/建隐藏预览分割(委托共享工厂 `_get_or_create_hidden_segmentation`) |
| `_update_magic_wand_preview` / `on_preview_wand_clicked` | 计算并写入预览 segment;非魔棒来源时仅清理 |
| `_destroy_wand_seed` | 删当前及历史命名的种子节点(`_WAND_SEED_NODE_NAMES`),摘 observer,退出放置模式 |

## 四、隐藏节点与状态

- 种子节点:`SelectionOpWandSeeds`(Fiducial)。历史命名集中在类常量
  `_WAND_SEED_NODE_NAMES`,`setup()` 与 Clear 时清扫遗留,避免重载泄漏。
- 预览分割:`"MagicWandPreviewSegmentNode (do not touch)"`(隐藏,
  `_internal_segmentation_node_names()` 排除),紫红色、2D 填充 0.35 / 轮廓 0.9。
- 实例状态:`_sel_op_wand_seed_node`、`_sel_op_wand_preview_segment_node`、
  `_wand_preview_segment_id`。

## 五、数据流

1. 用户放种子 -> `_on_wand_seed_placed` -> 可触发预览更新。
2. Apply(`on_apply_selection_op_clicked`,来源==`OPERAND_SOURCE_WAND`):
   `_compute_magic_wand_mask` 得到 operand mask -> `apply_boolean_operation`
   -> `show_segmentation` -> 记 undo(`_record_selection_op_undo`)-> 上传 server。
3. Apply 后清理预览(`_clear_wand_preview_segment`)。

## 六、注意点

- `_compute_magic_wand_mask` 会临时改写 server 的交互链(备份->重置->发种子->恢复),
  与正常 prompt 共用同一 stateful、单会话的 server,期间不要并发其他 prompt。
- 预览/种子节点跟随当前推理体积(`get_inference_volume_node`)的几何;切换序列后
  应重算。
- 预览节点目前在切换操作数来源时只 **清空**(`_clear_wand_preview_segment`),不销毁;
  与 lasso-3D 的"切换即销毁"不同(历史差异,行为无害)。
