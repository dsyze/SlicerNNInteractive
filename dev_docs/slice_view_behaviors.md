# 切片视图行为(交叉线 / 网格吸附 / 斜面对齐)

> 用途:记录三项与 2D 切片视图相关的二次开发行为。
> 行号引用基于撰写时代码,改动后需复核;实现全部在 client。

这三项功能解决同一类痛点:让 2D 切片视图在斜采集(oblique)序列与各向异性体素上
**对齐真实采集平面、逐张落在真实切片上**,而不是显示插值出来的中间帧。

## 一、切片交叉线(slice intersections)

- `enable_slice_intersections`(静态):在每个 2D 视图显示其余切片平面的交线,
  提供 3D 位置参照。`setup()` 内调用一次。属轻量增强。

## 二、滚轮吸附原始体素网格

- UI:`cbSnapSlicesToGrid`;设置键 `SETTING_SNAP_SLICES`(默认 True)。
- 目的:鼠标滚轮每次**恰好前进一张真实切片**,消除"滚动步长小于体素间距"导致的
  回弹/插值中间帧。
- 关键方法:
  - `_install_slice_snap_observers` / `_remove_slice_snap_observers`:在三个标准视图的
    slice node 上挂/摘 `ModifiedEvent` observer(裸 `AddObserver` tag 模式,存于
    `_slice_snap_observers`;**不**走 VTKObservationMixin,故 `cleanup()` 显式
    `_remove_slice_snap_observers`)。
  - `_on_slice_node_modified`:吸附回调,含递归守卫(`_snapping_in_progress`)与
    方向跟踪(`_snap_last_offset`),把 offset 归到最近的真实切片。
  - `_refresh_slice_snap` / `_on_snap_slices_changed`:按设置启停;`__init__`/`setup`
    据持久化设置安装,`init_ui_functionality` 只用 blockSignals 同步勾选框。

## 三、斜采集面自动对齐

- 目的:斜采集序列在 RAS 轴视图里是歪的;把视图旋到序列自身的采集平面,图像被"摆正",
  交叉线与序列真实轴对齐,逐张滚动即真实切片。
- 关键方法:
  - `_volume_is_oblique`:用方向余弦判定是否倾斜(阈值常量 `OBLIQUE_COS_THRESHOLD`,
    约偏离最近 RAS 轴 2.5 度)。
  - `_align_views_to_volume_planes`:把三视图重定向到体积的 IJK 轴。
- 触发点:网格吸附启用时,`_apply_plane_display_volumes` 在换背景后会调用它(换背景体
  可能是斜序列);`_install_slice_snap_observers` 安装时也对齐一次。

## 四、注意点

- 吸附 observer 是手动 tag 模式,新增/移除务必成对,并在 `cleanup()` 兜底移除。
- 斜面对齐与网格吸附**耦合**:吸附关闭时,换背景后不会自动重对齐(见
  `_apply_plane_display_volumes` 中 `if self._get_snap_slices_setting():` 分支)。
- 与 [[multi_plane_display_volumes]] 的多平面背景相互配合(背景体几何决定对齐目标)。
