# 界面中文本地化（Chinese localization）设计稿

## 目标

把 nnInteractive 插件**自有界面**的全部用户可见文字改为中文显示，供团队二次开发自用。覆盖：控件文字、分组框/Tab 标题、下拉项、悬浮提示(tooltip)、运行时弹窗(`QMessageBox`)、状态栏消息、运行时动态拼接的按钮文字、模块 Help 文本。

风格：**中文为主 + 保留关键缩写**（`ROI / BBox / SDF / nnInteractive / RAS / DICOM` 等保留原文；`Axial/Sagittal/Coronal`→轴位/矢状/冠状；`Rigid/Affine`→刚性/仿射），必要时「中文(英文)」并列。

## 关键约束

- **CI `check-utf8.yml` 只扫描 `*.py`，禁止任何非 ASCII 字符**（`grep -n '[^ -~]'`）。所以 **`.py` 源码里绝不能直接写中文**。
- `.ui`（Qt Designer XML，UTF-8）**不被该检查扫描**，可直接写中文 —— 仓库早期 `rotateViewGroup`（手动旋转观察平面）那组就是直接写中文的，本方案沿用统一。
- 不采用 Qt 标准 `tr()/.ts/.qm`：生成的 `.qm` 只在「Slicer 应用语言设为中文」时才生效，英文版 Slicer 下不显示中文，且需要 `lrelease`。直接改 `.ui` + 外置 JSON 表可**无条件显示中文**。

## 方案：两条通路

### 1. 静态界面文本 → 直接改 `.ui`

文件：`slicer_plugin/SlicerNNInteractive/Resources/UI/SlicerNNInteractive.ui`

逐条把用户可见的英文 `<string>`（按钮 text、QLabel、QGroupBox/ctkCollapsibleButton title、QComboBox `<item>`、QTabWidget tab title、`toolTip`）改为中文。只改文本内容，**不动** `name=`（objectName）、布局、数值属性、`notr="true"`、默认服务器地址。Alt 助记符 `&X` 写作「重置分段(&R)」形式保留快捷键（XML 中写 `&amp;`）。

### 2. Python 运行时文本 → 外置 JSON 表 + 出口包装

`.py` 保持纯 ASCII：英文原文作为 key 留在代码里，中文存放在不被 CI 扫描的 `Resources/Strings/zh_CN.json`（UTF-8），运行时查表，缺条目回退英文。

- `_cn(text)`（模块级）：懒加载 `zh_CN.json`（按 `__file__` 定位），返回 `table.get(text, text)`；加载失败/缺条目均回退英文，**永不抛异常打断插件**。
- `_status(message, *a, **k)`：包 `slicer.util.showStatusMessage`，显示前 `_cn(message)`。
- `_mb_warning / _mb_critical / _mb_information(parent, title, text, *a, **k)`：包对应 `QMessageBox.*`，`title`、`text` 均经 `_cn`。
- **接入方式**：对调用点做一次性全局替换 `slicer.util.showStatusMessage(` → `_status(`、`QMessageBox.warning(` → `_mb_warning(`（critical/information 同理）。包装函数自身的实现刻意写成无尾括号形式（`show = slicer.util.showStatusMessage` / `fn = QMessageBox.warning`），从而**不被全局替换命中**，避免递归。
- 少量动态拼接直接包：`_cn("Fuse & apply ({})").format(n)`、`_cn("Submit ({})").format(count)`、`_cn("Show/hide the %s locator plane in 3D") % view_name`、helpText `_cn("...%s...") % PLUGIN_VERSION`。

## 维护规则（重要）

- **新增任何 Python 动态用户文本时，必须同步在 `zh_CN.json` 补一条 key**，且 key 与源码英文字面量**逐字节一致**（含标点、空格、`{}`/`%s`/`%d`/`\n` 等占位符）。漏补 = 回退英文，不报错。
- 新增静态控件 → 直接在 `.ui` 写中文。
- 翻译 value 必须**原样保留**所有占位符与格式符。

## 已知限制

- **内嵌 `qMRMLSegmentEditorWidget`**（各 Effect 名、分段表头、Add/Remove 等）与 `qMRMLNodeComboBox` 的 None/节点名由 **Slicer 本体**提供，跟随 Slicer 应用语言，本方案改不到。若需其中文：把 Slicer 应用语言设为中文（Edit → Application Settings → General → Language），与本插件本地化相互独立、可叠加。

## 关键文件

- `Resources/UI/SlicerNNInteractive.ui` —— 静态界面中文
- `Resources/Strings/zh_CN.json` —— Python 动态文本英文→中文表
- `SlicerNNInteractive.py` —— `_cn` / `_status` / `_mb_*`（模块级，class 之前）+ 全局替换接入
- 总览：`dev_docs/feature_inventory_and_internals.md` 第十九节
