# Divere 内存优化修复方案

## 问题诊断总结

根据代码分析，divere 系统在操作中、更新 preview、换照片过程中内存单调递增的主要原因：

### 核心问题点

1. **QPixmap 对象累积**（最主要）
   - 每次 `_update_display()` 都创建新的 QPixmap（~17MB/次）
   - 转换链：numpy array → QImage → qimage.copy() → QPixmap（多次拷贝）
   - Qt 的 QPixmap 在 GPU 内存中缓存，即使 Python 对象删除，Qt 也可能延迟释放
   - 旧 pixmap 在 `set_source_pixmap()` 中只是 `= None`，但没有强制 Qt 释放 GPU 资源

2. **ImageData 在信号槽中的生命周期**
   - `preview_updated.emit(result_image)` 中的 ImageData 对象被 Qt 信号槽机制持有
   - 多个槽（`_on_preview_updated`, `_on_preview_updated_for_contactsheet`）都连接，延长对象生命周期
   - 过时的 preview 结果虽然被丢弃，但 ImageData 对象可能已创建

3. **切换图像时的清理不彻底**
   - `reset()` 方法存在，但 QPixmap 的 GPU 资源释放依赖 Qt 的事件循环
   - 切换图像时，旧的 QPixmap 可能还没完全释放，新的就已经创建

4. **高频预览更新时的对象堆积**
   - 参数快速调整时，多个 preview 任务可能同时在队列中
   - 虽然 generation 机制丢弃过时结果，但已创建的中间对象可能堆积

---

## 修复方案（基于现有代码架构）

### 方案 1: 强制释放 QPixmap GPU 资源 ⭐⭐⭐

**位置：** `divere/ui/preview_widget.py`

**问题：** `set_source_pixmap()` 只是设置 `= None`，但没有强制 Qt 释放 GPU 纹理缓存

**修复：**
```python
def set_source_pixmap(self, pixmap: QPixmap) -> None:
    # 释放旧的 source pixmap 引用
    if self._source_pixmap is not None:
        # 关键修复：强制 Qt 释放 GPU 资源
        # detach() 断开与底层数据源的连接，触发资源释放
        self._source_pixmap.detach()
        self._source_pixmap = None

    # 释放旧的 scaled pixmap 缓存
    if self._scaled_pixmap is not None:
        self._scaled_pixmap.detach()
        self._scaled_pixmap = None

    # 设置新的 source pixmap
    self._source_pixmap = pixmap
    # ... 其余代码不变
```

**在 `reset()` 中也添加：**
```python
def reset(self):
    # 释放 ImageData 对象
    if self.current_image is not None:
        del self.current_image
    self.current_image = None

    # 强制释放 pixmap 的 GPU 资源
    if self.image_label._source_pixmap is not None:
        self.image_label._source_pixmap.detach()
        self.image_label._source_pixmap = None
    if self.image_label._scaled_pixmap is not None:
        self.image_label._scaled_pixmap.detach()
        self.image_label._scaled_pixmap = None
    
    self.image_label.set_source_pixmap(QPixmap())
    self.image_label.setText("")
```

---

### 方案 2: 优化 QPixmap 创建，减少中间对象 ⭐⭐

**位置：** `divere/ui/preview_widget.py` 的 `_array_to_pixmap()`

**问题：** 创建了太多中间对象：QImage → qimage.copy() → QPixmap，每次都是全量拷贝

**修复：** 考虑缓存或复用机制（但要注意线程安全）

**简化版修复（先不缓存，但优化拷贝）：**
```python
def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
    # ... 现有的类型转换代码不变 ...
    
    # 创建 QImage（引用 numpy array）
    if len(array.shape) == 3:
        channels = array.shape[2]
        if channels == 3:
            qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
        elif channels == 4:
            qimage = QImage(array.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        # ...
    else:
        qimage = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)

    # 应用色彩空间
    if hasattr(self, 'current_image') and self.current_image and self.current_image.color_space == "DisplayP3":
        from PySide6.QtGui import QColorSpace
        displayp3_space = QColorSpace(QColorSpace.NamedColorSpace.DisplayP3)
        qimage.setColorSpace(displayp3_space)

    # 关键修复：直接创建 QPixmap，不经过 copy()
    # 但需要确保 QImage 的数据在 QPixmap 创建后仍然有效
    # 由于我们在同一个函数内完成，array 还在作用域内，所以是安全的
    pixmap = QPixmap.fromImage(qimage.copy())  # 仍需 copy() 确保数据独立
    
    # 修复：立即释放临时的 QImage copy（如果 Qt 允许）
    # 实际上 QImage 会被 GC 回收，但我们可以帮助一下
    del qimage  # 显式释放引用
    
    return pixmap
```

**注意：** 这个方案可能影响很小，因为 `qimage.copy()` 和 `QPixmap.fromImage()` 都是必要的。真正的优化可能需要引入 pixmap 缓存机制。

---

### 方案 3: 在切换图像时强制清理和等待 ⭐⭐⭐

**位置：** `divere/ui/main_window.py` 的 `_on_image_loading_started()`

**问题：** `reset()` 被调用了，但可能还没等旧资源释放完就开始加载新图

**修复：**
```python
def _on_image_loading_started(self):
    """图像开始加载时的处理：清理预览控件资源"""
    # 重置预览控件
    self.preview_widget.reset()
    
    # 关键修复：强制 Qt 处理 pending 的删除操作
    # 这确保 QPixmap 的 deleteLater() 被立即处理
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.processEvents(QCoreApplication.ProcessEventsFlag.DeferredDeletions)
    
    # 触发垃圾回收，确保 Python 对象被释放
    import gc
    gc.collect()
```

---

### 方案 4: 在预览更新时显式释放旧 ImageData ⭐⭐

**位置：** `divere/ui/preview_widget.py` 的 `set_image()`

**问题：** 虽然已经 `del self.current_image`，但在高频更新时可能不够及时

**修复：**
```python
def set_image(self, image_data: ImageData):
    """设置显示的图像"""
    # 保存当前cut-off显示状态
    was_showing_cutoff = self._show_black_cutoff
    current_compensation = self._cutoff_compensation

    # 显式释放旧的ImageData对象以防止内存泄漏
    if self.current_image is not None:
        # 关键修复：先释放图像数组（最大内存占用）
        if hasattr(self.current_image, 'array') and self.current_image.array is not None:
            self.current_image.array = None  # 释放 numpy 数组引用
        del self.current_image

    self.current_image = image_data
    
    # ... 其余代码不变 ...
    
    # 修复：更新显示后，如果旧 pixmap 还存在，强制释放
    self._update_display()
    
    # 如果正在显示cut-off，重新检测像素以同步最新图像数据
    if was_showing_cutoff:
        self._show_black_cutoff = True
        self._cutoff_compensation = current_compensation
        self._detect_black_cutoff_pixels()
        
    self.image_label.update()
```

---

### 方案 5: 在预览结果回调中立即释放过时结果 ⭐⭐⭐

**位置：** `divere/core/app_context.py` 的 `_on_preview_result_from_signals()`

**问题：** 过时的 `result_image` 虽然被丢弃，但对象可能还在内存中

**修复：**
```python
@Slot(ImageData)
def _on_preview_result_from_signals(self, result_image: ImageData):
    """新的preview结果回调"""
    sig = self.sender()
    if not sig:
        return

    gen = getattr(sig, "generation", -1)

    # 关键检查：如果不是最新的generation，立即丢弃并释放
    if gen != self._preview_generation:
        # 过时结果，立即释放资源
        if result_image is not None and hasattr(result_image, 'array'):
            result_image.array = None  # 释放 numpy 数组（最大内存占用）
        return  # 不发送到UI

    # 最新结果，发送到UI并触发后续处理
    self.preview_updated.emit(result_image)
    
    # ... 其余代码不变 ...
```

---

### 方案 6: 在 load_image 中增强清理 ⭐⭐

**位置：** `divere/core/app_context.py` 的 `load_image()`

**问题：** 虽然已有清理逻辑，但可能不够彻底

**修复：**
```python
def load_image(self, file_path: str):
    try:
        # 通知 UI 层图像加载开始（用于清理旧资源）
        self.image_loading_started.emit()

        # === 内存优化：切图时等待并清理在途任务 ===
        # 1) 等待在途任务收尾
        try:
            self.thread_pool.waitForDone(2000)
        except Exception:
            pass

        # 2) 停止自动校色/中性点迭代调度
self._auto_color_iterations = 0
self._get_preview_for_auto_color_callback = None
self._neutral_point_iterations = 0
self._neutral_point_callback = None

        # 3) 显式释放大对象
        if self._current_proxy is not None:
            if hasattr(self._current_proxy, 'array') and self._current_proxy.array is not None:
                self._current_proxy.array = None  # 关键修复：释放数组
            self._current_proxy = None
            
        if self._current_image is not None:
            if hasattr(self._current_image, 'array') and self._current_image.array is not None:
                self._current_image.array = None  # 关键修复：释放数组
            self._current_image = None
            
        # 4) 处理 Qt 事件，确保 deleteLater() 被处理
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents(QCoreApplication.ProcessEventsFlag.DeferredDeletions)
        
        # 5) 强制垃圾回收
        import gc
        gc.collect()
        # === 内存优化结束 ===

        # ... 其余加载代码不变 ...
```

---

## 修复优先级和建议实施顺序

### 第一批（最关键，立即实施）：
1. **方案 1**: 强制释放 QPixmap GPU 资源（`detach()`）
2. **方案 3**: 在切换图像时强制清理和等待
3. **方案 5**: 在预览结果回调中立即释放过时结果

### 第二批（重要，观察第一批效果后决定）：
4. **方案 4**: 在预览更新时显式释放旧 ImageData
5. **方案 6**: 在 load_image 中增强清理

### 第三批（可选，如果前两批效果不够）：
6. **方案 2**: 优化 QPixmap 创建（可能需要引入缓存机制，改动较大）

---

## 验证方法

### 1. 内存监控
在关键位置添加内存使用日志：
```python
import psutil
import os

def log_memory(prefix=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] {prefix}: {mem_mb:.1f} MB")

# 在以下位置调用：
# - _trigger_preview_update() 开始和结束
# - _on_preview_result_from_signals() 开始和结束
# - load_image() 开始和结束
# - set_image() 开始和结束
```

### 2. 观察指标
- **正常操作**：内存应该在小范围内波动，不再单调递增
- **切换图像**：内存峰值应该能快速回落（1-2秒内）
- **高频预览**：内存增长应该被抑制，不会无限累积

### 3. 压力测试
- 连续切换 50 张图像，观察内存是否持续增长
- 快速拖拽参数滑块 5 分钟，观察内存是否稳定
- 长时间运行（30分钟+），观察是否有缓慢泄漏

---

## 注意事项

1. **`detach()` 方法**：确保在不同 Qt 版本中可用，如果不可用，可能需要使用其他方法（如创建空 QPixmap 替换）

2. **`processEvents()` 调用**：要谨慎使用，避免在关键路径造成性能问题。只在切换图像时调用。

3. **线程安全**：所有修改都应在主线程中进行，确保 Qt 对象操作的安全性。

4. **向后兼容**：修改要确保不影响现有功能，特别是预览更新的实时性。

---

## 预期效果

实施第一批修复后：
- ✅ 高频预览时，内存不再单调递增，而是在小范围内波动
- ✅ 切换图像时，内存峰值降低，且能快速回落
- ✅ 长时间运行时，内存使用保持稳定

如果第一批效果不够，继续实施第二批和第三批修复。

---

## 🔴 高频预览更新内存快速累积问题（新发现）

### 问题现象

- ✅ 更换图片不再增加内存占用（第一批修复有效）
- ❌ **但频繁调整 slider 触发 preview 更新后，内存会非常快速地单调递增**

### Root Cause 分析

经过代码深度分析，发现问题的根本原因：

#### 1. Pipeline 处理总是创建新的 ImageData 对象

**位置：** `divere/core/pipeline_processor.py` 和 `divere/core/color_space.py`

**问题：**
- `apply_full_precision_pipeline()` 返回 `image.copy_with_new_array(new_array)` - **总是创建新的 ImageData**
- `convert_to_display_space()` 虽然修改 `image.array`，但 pipeline 函数本身创建新对象
- 每次 preview 都会创建**全新的 ImageData 对象**，包含 ~17MB 的 numpy array

```python
# pipeline_processor.py
return image.copy_with_new_array(working_array)  # 创建新的 ImageData

# Worker 中的流程
result_image = self.the_enlarger.apply_full_pipeline(...)  # 新对象 1
result_image = self.color_space_manager.convert_to_display_space(...)  # 可能返回新对象
self.signals.result.emit(result_image)  # 发送到信号队列
```

#### 2. Qt 信号槽机制持有多个 result_image 对象

**问题：**
- 每个 preview worker 完成后都会 `emit(result_image)`
- Qt 信号槽机制会**持有**所有通过信号传递的对象，直到所有连接的槽处理完成
- 在高频预览场景中：
  - 多个 preview 任务可能在队列中
  - 每个任务创建新的 `result_image`（~17MB）
  - 即使过时的结果会被丢弃（generation 检查），但**在信号队列中的 ImageData 对象仍然被持有**
  - 对象生命周期：创建 → 进入信号队列 → 等待处理 → 被丢弃/处理 → 释放

**时间线示例（高频调整时）：**
```
T0: 用户拖动 slider
T1: Preview 1 开始 → 创建 result_image_1 (~17MB)
T2: 用户继续拖动 → Preview 2 开始 → 创建 result_image_2 (~17MB)
T3: Preview 1 完成 → emit(result_image_1) → 进入信号队列
T4: Preview 2 完成 → emit(result_image_2) → 进入信号队列
T5: 槽处理 result_image_1 → 过时，丢弃
T6: 槽处理 result_image_2 → 最新，发送到 UI
```

**问题：** 在 T3-T6 期间，两个 ~17MB 的 ImageData 对象同时存在于内存中。

#### 3. UI 层也会持有 result_image

**位置：** `divere/ui/preview_widget.py` 的 `set_image()`

**问题：**
- `set_image(result_image)` 会持有 `result_image`
- 如果下一个 preview 结果到达时，旧的 `current_image` 虽然会被释放，但：
  - 释放操作在 `set_image()` 中执行
  - 如果信号队列中有多个结果，旧的还未处理完，新的就已经到达
  - 导致多个 ImageData 对象同时被持有

#### 4. Generation 机制只能丢弃，不能阻止创建

**问题：**
- Generation 机制在 `_on_preview_result_from_signals()` 中检查
- 但这**发生在对象创建之后**
- 即使过时结果会被丢弃，但**对象已经创建，内存已经分配**

---

### 解决方案：复用单一 Preview Result 缓冲区

**核心思想：** 不要每次 preview 都创建新的 ImageData，而是维护一个固定的预览结果缓冲区，每次直接更新缓冲区的内容。

#### 方案设计

**1. 在 ApplicationContext 中维护预览结果缓冲区**

```python
class ApplicationContext(QObject):
    def __init__(self):
        # ... 现有代码 ...
        
        # 新增：预览结果缓冲区（复用同一块内存）
        self._preview_result_buffer: Optional[ImageData] = None
```

**2. 修改 Pipeline 函数支持"就地更新"模式**

**选项 A（推荐）：** 修改 pipeline 函数，支持接收目标 ImageData

```python
# the_enlarger.py 或 pipeline_processor.py
def apply_full_precision_pipeline(self, image: ImageData, params: ColorGradingParams,
                                 target_image: Optional[ImageData] = None,  # 新增参数
                                 ...) -> ImageData:
    """
    如果提供了 target_image，直接更新其 array，而不是创建新对象
    否则创建新对象（向后兼容）
    """
    # ... pipeline 处理逻辑，生成 result_array ...
    
    if target_image is not None:
        # 就地更新：直接更新目标对象的 array
        target_image.array = result_array
        target_image.color_space = new_color_space  # 更新元数据
        # 更新其他元数据...
        return target_image
    else:
        # 向后兼容：创建新对象
        return image.copy_with_new_array(result_array)
```

**选项 B（更简单）：** 在 Worker 中直接复用缓冲区

```python
# app_context.py 的 _PreviewWorker.run()
@Slot()
def run(self):
    try:
        # ... pipeline 处理 ...
        result_image = self.the_enlarger.apply_full_pipeline(...)
        result_image = self.color_space_manager.convert_to_display_space(...)
        
        # 关键修复：如果存在缓冲区，复用其 array
        if self.shared_buffer is not None:
            # 直接更新缓冲区的内容，而不是创建新对象
            self.shared_buffer.array = result_image.array
            self.shared_buffer.color_space = result_image.color_space
            # 更新其他元数据...
            result_image = self.shared_buffer  # 使用缓冲区对象
        else:
            # 首次创建，保存为缓冲区
            self.shared_buffer = result_image
            
        self.signals.result.emit(result_image)
    except Exception as e:
        # ...
```

**3. 修改 Worker 初始化，传入缓冲区引用**

```python
# app_context.py 的 _trigger_preview_update()
def _trigger_preview_update(self):
    # ... 现有代码 ...
    
    # 确保预览结果缓冲区存在
    if self._preview_result_buffer is None:
        # 首次创建缓冲区（使用当前 proxy 的尺寸）
        h, w = self._current_proxy.array.shape[:2]
        # 创建空的 ImageData 作为缓冲区
        self._preview_result_buffer = ImageData(
            array=None,  # 稍后在 pipeline 中填充
            width=w,
            height=h,
            channels=3,
            dtype=np.float32,
            color_space="DisplayP3"
        )
    
    worker = _PreviewWorker(
        image=proxy_view,
        params=params_view,
        the_enlarger=self.the_enlarger,
        color_space_manager=self.color_space_manager,
        convert_to_monochrome_in_idt=self.should_convert_to_monochrome(),
        shared_signals=self._preview_signals,
        shared_buffer=self._preview_result_buffer  # 新增：传入缓冲区引用
    )
```

**4. 在切换图像时清空缓冲区**

```python
# app_context.py 的 load_image()
def load_image(self, file_path: str):
    # ... 现有清理代码 ...
    
    # 新增：清空预览结果缓冲区
    if self._preview_result_buffer is not None:
        if hasattr(self._preview_result_buffer, 'array') and self._preview_result_buffer.array is not None:
            self._preview_result_buffer.array = None
        self._preview_result_buffer = None
```

---

### 实现细节和注意事项

#### 1. 线程安全

- Preview Worker 在后台线程运行
- 如果直接修改共享缓冲区，需要考虑线程安全
- **解决方案：** Worker 处理完成后，在主线程（信号槽中）更新缓冲区

#### 2. 向后兼容性

- Pipeline 函数需要保持向后兼容
- 如果没有传入 `target_image`，应该创建新对象（现有行为）
- 只有在明确传入时才进行就地更新

#### 3. Array 尺寸变化

- 如果新的 preview 结果尺寸与缓冲区不同，需要重新分配
- **处理：** 检查尺寸，如果不匹配，释放旧 array 并创建新的

```python
# 在 Worker 或信号槽中
if self.shared_buffer.array is None or self.shared_buffer.array.shape != result_image.array.shape:
    # 尺寸不匹配，重新分配
    self.shared_buffer.array = result_image.array.copy()
else:
    # 尺寸匹配，直接覆盖（避免创建新 array）
    np.copyto(self.shared_buffer.array, result_image.array)
```

#### 4. 信号槽中的对象生命周期

- 即使使用缓冲区，Qt 信号槽仍然会持有引用
- **但关键区别：** 始终是同一个对象，不会累积多个对象
- 旧的引用会被新的覆盖，对象本身不会累积

---

### 实施优先级

**🔴 最高优先级** - 立即实施

这个方案比之前的修复更加根本，直接解决了频繁预览导致的内存累积问题。

### 实施步骤

1. **第一步：** 在 ApplicationContext 中添加 `_preview_result_buffer`
2. **第二步：** 修改 `_PreviewWorker` 支持传入缓冲区
3. **第三步：** 修改 Worker 的 `run()` 方法，复用缓冲区而不是创建新对象
4. **第四步：** 在切换图像时清空缓冲区
5. **第五步：** 测试验证内存不再累积

### 预期效果

- ✅ 频繁调整 slider 时，内存不再快速累积
- ✅ 始终只维护一个预览结果 ImageData 对象（~17MB）
- ✅ 即使有多个 preview 任务在队列中，也只会有一个 result_image 对象
- ✅ 内存使用在小范围内波动，不再单调递增

---

## 总结

### 已修复的问题

- ✅ 更换图片时的内存累积（第一批修复）
- ✅ QPixmap GPU 资源释放
- ✅ 切换图像时的清理

### 待修复的问题

- 🔴 **高频预览更新时的内存快速累积**（本文档新分析）
  - 根本原因：每次 preview 都创建新的 ImageData 对象
  - 解决方案：复用单一预览结果缓冲区

### 建议实施顺序

1. **立即实施：** 复用单一 Preview Result 缓冲区方案（本文档新方案）
2. **已验证有效：** 第一批 QPixmap 和清理修复（已实施）
3. **可选优化：** QPixmap 缓存机制（如果内存使用仍不够理想）

---

## 📊 完整调用链和对象生命周期分析

### 从 Slider 拖动到 Preview 更新的完整调用链

#### 1. 用户交互阶段

```
用户拖动 Slider
  ↓
ParameterPanel.PrecisionSlider.valueChanged
  ↓
ParameterPanel._on_*_slider_changed() (例如 _on_gamma_slider_changed)
  ↓
parameter_changed.emit()  [Qt Signal]
```

**信号连接：**
- `ParameterPanel.parameter_changed` → `MainWindow.on_parameter_changed`

---

#### 2. 参数更新阶段

```
MainWindow.on_parameter_changed()
  ↓
parameter_panel.get_current_params()  [创建新的 ColorGradingParams 对象]
  ↓
ApplicationContext.update_params(new_params)
  ↓
self._current_params = new_params  [更新参数]
self.params_changed.emit(self._current_params)  [Qt Signal]
self._trigger_preview_update()  [立即触发预览更新]
```

**关键对象：**
- `new_params: ColorGradingParams` - 新参数对象（已优化使用 shallow_copy，开销小）
- `self._current_params` - 更新后的参数

---

#### 3. 预览触发阶段

```
ApplicationContext._trigger_preview_update()
  ↓
检查: _preview_busy? → 如果忙，设置 _preview_pending，返回
  ↓
设置: _preview_busy = True
  ↓
_preview_generation += 1  [分配唯一 ID]
  ↓
创建: proxy_view = self._current_proxy.view()  [共享数组，不复制]
创建: params_view = self._current_params.shallow_copy()  [共享参数，不复制]
  ↓
检查/创建: _preview_result_buffer  [如果不存在或尺寸变化]
  ↓
创建: _PreviewWorker(image=proxy_view, params=params_view, shared_buffer=...)
  ↓
设置: worker.generation = gen
设置: self._preview_signals.generation = gen
  ↓
thread_pool.start(worker)  [后台线程执行]
  ↓
清理: del proxy_view, del params_view, gc.collect()
```

**关键对象：**
- `proxy_view: ImageData` - 共享 `_current_proxy.array` 的视图（~几KB开销）
- `params_view: ColorGradingParams` - 参数的浅拷贝（~几KB开销）
- `_PreviewWorker` - Worker 对象（持有 proxy_view 和 params_view 的引用）

---

#### 4. Pipeline 处理阶段（后台线程）

```
_PreviewWorker.run()  [后台线程]
  ↓
self.the_enlarger.apply_full_pipeline(self.image, self.params, ...)
  ↓
  FilmPipelineProcessor.apply_full_precision_pipeline()
    ↓
    working_array = image.array.copy()  [⚠️ 创建新数组 ~17MB]
    ↓
    [处理过程：色彩转换、矩阵、曲线等，修改 working_array]
    ↓
    return image.copy_with_new_array(working_array)  [⚠️ 创建新 ImageData 对象]
  ↓
result_image = self.color_space_manager.convert_to_display_space(result_image, "DisplayP3")
  ↓
  ColorSpaceManager.convert_to_display_space()
    ↓
    image.array = self._apply_color_conversion(...)  [修改现有 array]
    image.array = self._apply_gamma(...)  [修改现有 array]
    ↓
    return image  [返回同一个对象，但 array 已被修改]
  ↓
[关键修复：复用缓冲区]
if self.shared_buffer is not None:
  if 尺寸匹配:
    np.copyto(self.shared_buffer.array, result_image.array)  [覆盖到缓冲区]
  else:
    self.shared_buffer.array = result_image.array  [转移数组所有权]
  ↓
  tmp.array = None  [释放临时对象的数组引用]
  del tmp
  result_image = self.shared_buffer
  ↓
self.signals.result.emit(result_image)  [Qt Signal，发射到主线程]
  ↓
finally:
  del self.image  [释放 proxy_view]
  del self.params  [释放 params_view]
  gc.collect()  [触发垃圾回收]
  self.signals.finished.emit()
```

**关键对象生命周期：**
- `working_array: np.ndarray` - Pipeline 中创建的数组（~17MB）
  - **创建点：** `image.array.copy()`（line 400）
  - **生命周期：** 在整个 pipeline 处理期间存在
  - **下场：** 
    - 如果使用缓冲区且尺寸匹配：通过 `np.copyto()` 复制到缓冲区后被 GC 回收
    - 如果使用缓冲区且尺寸不匹配：数组转移到缓冲区，临时对象释放引用后 GC 回收
    - 如果不使用缓冲区：成为 `result_image.array`，通过信号传递

- `result_image: ImageData` (Pipeline 返回的)
  - **创建点：** `image.copy_with_new_array(working_array)`（line 516）
  - **生命周期：** Worker.run() 执行期间
  - **下场：**
    - 如果使用缓冲区：数组转移到缓冲区后，临时对象被删除
    - 如果不使用缓冲区：通过信号传递到主线程

- `shared_buffer: ImageData` (如果存在)
  - **创建点：** `_trigger_preview_update()` 中首次创建
  - **生命周期：** 长期存在，直到切换图像
  - **array 生命周期：**
    - 首次：从 result_image 转移过来
    - 后续：如果尺寸匹配，通过 `np.copyto()` 覆盖；如果不匹配，重新分配

---

#### 5. 信号传递阶段（Qt 事件循环）

```
_PreviewWorkerSignals.result.emit(result_image)  [后台线程 → 主线程]
  ↓
[Qt 事件队列]
  ↓
ApplicationContext._on_preview_result_from_signals(result_image)  [主线程 Slot]
  ↓
检查: generation 是否匹配
  ↓
如果过时: return  [不发送到 UI，但对象仍在信号队列中]
  ↓
如果最新: self.preview_updated.emit(result_image)  [Qt Signal]
```

**关键对象：**
- `result_image: ImageData` - 在信号队列中被 Qt 持有
  - **生命周期：** 从 emit 到所有连接的 slot 处理完成
  - **如果过时：** 对象仍在信号队列中，直到 slot 返回后才被释放
  - **如果最新：** 继续传递到 UI

**⚠️ 问题：** 即使过时结果被丢弃，但 ImageData 对象仍在信号队列中，直到事件循环处理完成。在高频场景中，多个过时结果可能同时在队列中。

---

#### 6. UI 更新阶段

```
ApplicationContext.preview_updated.emit(result_image)  [Qt Signal]
  ↓
MainWindow._on_preview_updated(result_image)
  ↓
preview_widget.set_image(result_image)
  ↓
PreviewWidget.set_image(image_data)
  ↓
检查: current_image is not image_data?
  ↓
如果是不同对象: 
  current_image.array = None  [释放旧数组]
  del current_image
  ↓
current_image = image_data  [持有新的 ImageData]
  ↓
_update_display()
  ↓
_array_to_pixmap(self.current_image.array)
  ↓
  [类型转换和准备]
  qimage = QImage(array.data, ...)  [引用 numpy array]
  qimage_independent = qimage.copy()  [⚠️ 创建新的 QImage ~17MB]
  pixmap = QPixmap.fromImage(qimage_independent)  [⚠️ 创建新的 QPixmap ~17MB]
  ↓
image_label.set_source_pixmap(pixmap)
  ↓
PreviewCanvas.set_source_pixmap(pixmap)
  ↓
旧的 _source_pixmap.detach()  [释放 GPU 资源]
旧的 _source_pixmap = None
_source_pixmap = pixmap  [设置新的 pixmap]
  ↓
update()  [触发重绘]
```

**关键对象生命周期：**

1. **`result_image: ImageData` (从信号传递来的)**
   - **生命周期：** 从信号传递到 `set_image()` 完成
   - **下场：** 
     - 如果与 `current_image` 是同一个对象（缓冲区复用）：继续保留
     - 如果是不同对象：在 `set_image()` 中被释放（`array = None`）

2. **`qimage: QImage` (临时对象)**
   - **创建点：** `QImage(array.data, ...)`（line 1449）
   - **生命周期：** 在 `_array_to_pixmap()` 函数内
   - **下场：** 函数返回后被 GC 回收

3. **`qimage_independent: QImage`**
   - **创建点：** `qimage.copy()`（line 1471）
   - **大小：** ~17MB
   - **生命周期：** 从创建到 `QPixmap.fromImage()` 完成
   - **下场：** 函数返回后被 GC 回收

4. **`pixmap: QPixmap`**
   - **创建点：** `QPixmap.fromImage(qimage_independent)`（line 1475）
   - **大小：** ~17MB（GPU 纹理缓存 + 系统内存）
   - **生命周期：** 
     - 从创建到下次 `set_source_pixmap()` 调用
     - 在 `set_source_pixmap()` 中，旧 pixmap 被 `detach()` 和释放
   - **下场：**
     - 旧 pixmap：`detach()` → `= None` → Qt 延迟释放 GPU 资源
     - 新 pixmap：被 `_source_pixmap` 持有，直到下次更新

5. **`current_image: ImageData` (PreviewWidget 持有)**
   - **生命周期：** 从 `set_image()` 到下次 `set_image()` 或 `reset()`
   - **下场：**
     - 下次 `set_image()` 时，如果对象不同，先释放 array，然后删除对象
     - 如果对象相同（缓冲区复用），不释放，继续使用

---

### 对象生命周期总览（单次 Preview 更新）

#### 创建的对象（按时间顺序）

| 对象 | 创建位置 | 大小 | 生命周期 | 释放时机 | 问题点 |
|------|---------|------|---------|---------|--------|
| `proxy_view` | `_trigger_preview_update()` | ~几KB | 直到 Worker 完成 | Worker.run() finally | ✅ 已优化 |
| `params_view` | `_trigger_preview_update()` | ~几KB | 直到 Worker 完成 | Worker.run() finally | ✅ 已优化 |
| `working_array` | `apply_full_precision_pipeline()` | ~17MB | Pipeline 处理期间 | Worker 中转移到缓冲区后 GC | ⚠️ 每次创建新数组 |
| `result_image` (临时) | `copy_with_new_array()` | ~17MB | Worker.run() 期间 | 转移到缓冲区后删除 | ⚠️ 即使转移，对象本身也存在一段时间 |
| `qimage` | `_array_to_pixmap()` | ~几KB | 函数内 | 函数返回后 GC | ✅ 开销小 |
| `qimage_independent` | `qimage.copy()` | ~17MB | 函数内 | 函数返回后 GC | ⚠️ 每次创建新 QImage |
| `pixmap: QPixmap` | `QPixmap.fromImage()` | ~17MB | 直到下次更新 | `set_source_pixmap()` 时 detach() | ⚠️ Qt 可能延迟释放 GPU 资源 |

#### 复用的对象

| 对象 | 位置 | 大小 | 生命周期 | 更新方式 |
|------|------|------|---------|---------|
| `_preview_result_buffer` | `ApplicationContext` | ~17MB | 长期（直到切换图像） | 每次预览更新其 array |
| `_current_proxy.array` | `ApplicationContext` | ~17MB | 长期（直到切换图像） | 不改变（只读） |

---

### 🔴 发现的内存累积问题点

#### 问题 1: Pipeline 中每次创建新的 working_array

**位置：** `pipeline_processor.py:400`

```python
working_array = image.array.copy()  # ⚠️ 每次创建 ~17MB 新数组
```

**问题：**
- 即使最终转移到缓冲区，但在 Pipeline 处理期间，`working_array` 和 `result_image.array` 同时存在
- 在 Worker 转移到缓冲区之前，有两份 ~17MB 数组在内存中

**时间线：**
```
T0: working_array = image.array.copy()  [17MB]
T1: [Pipeline 处理，修改 working_array]
T2: result_image = image.copy_with_new_array(working_array)  [result_image.array = working_array]
    → 此时：working_array 和 result_image.array 是同一个数组引用（不是两份）
T3: Worker 转移到缓冲区
T4: Worker 完成，working_array 和临时 result_image 对象被 GC
```

**分析：** 实际上 `copy_with_new_array()` 只是转移引用，不会复制数组。所以只有一份数组。但问题是：
- Pipeline 处理过程中，会创建多个中间数组（色彩转换、矩阵运算等）
- 这些中间数组在处理期间会累积

#### 问题 2: 每次创建新的 QImage 和 QPixmap

**位置：** `preview_widget.py:_array_to_pixmap()`

```python
qimage = QImage(array.data, ...)  # 引用数组
qimage_independent = qimage.copy()  # ⚠️ 创建新的 ~17MB QImage
pixmap = QPixmap.fromImage(qimage_independent)  # ⚠️ 创建新的 ~17MB QPixmap
```

**问题：**
- 每次预览更新都创建新的 QImage 和 QPixmap
- 旧的 QPixmap 虽然调用了 `detach()`，但 Qt 可能延迟释放 GPU 资源
- 在高频预览时，多个 QPixmap 可能同时存在（旧的还没释放，新的就创建了）

**时间线：**
```
T0: Preview 1 完成 → 创建 QPixmap_1 (~17MB GPU)
T1: Preview 2 完成 → 创建 QPixmap_2 (~17MB GPU)
    → QPixmap_1 调用 detach()，但 GPU 资源可能还没释放
T2: Preview 3 完成 → 创建 QPixmap_3 (~17MB GPU)
    → QPixmap_1、QPixmap_2 的 GPU 资源可能还在缓存中
```

#### 问题 3: Qt 信号队列中的对象持有

**位置：** Qt 信号槽机制

**问题：**
- 每个 `emit(result_image)` 都会让 Qt 持有 `result_image` 对象，直到所有连接的 slot 处理完成
- 在高频预览时，多个 `result_image` 对象可能在信号队列中排队
- 即使过时的结果被丢弃，但对象仍在队列中，直到事件循环处理完成

**时间线：**
```
T0: Worker 1 完成 → emit(result_image_1) → 进入信号队列
T1: Worker 2 完成 → emit(result_image_2) → 进入信号队列
T2: [事件循环处理]
    → _on_preview_result_from_signals(result_image_1) → 过时，丢弃
    → _on_preview_result_from_signals(result_image_2) → 最新，发送到 UI
T3: [UI 处理完成，result_image_1 和 result_image_2 才被释放]
```

**在 T0-T3 期间：** 两个 ImageData 对象（可能都是缓冲区引用，但仍在队列中）

#### 问题 4: Pipeline 内部中间数组累积

**位置：** Pipeline 处理过程中的各种操作

**问题：**
- 色彩转换、矩阵运算、曲线处理等都会创建临时数组
- 这些临时数组在处理期间会累积，直到函数返回

**示例：**
```python
working_array = image.array.copy()  # 17MB
working_array = self._apply_colorspace_transform(working_array, ...)  # 可能创建临时数组
working_array = self.math_ops.apply_full_math_pipeline(...)  # 内部可能创建多个临时数组
```

---

### 内存累积的根本原因总结

1. **Pipeline 中的数组创建无法避免**
   - Pipeline 需要创建 `working_array` 来处理图像
   - 虽然最终转移到缓冲区，但处理期间的临时数组会累积

2. **QPixmap 的 GPU 资源延迟释放**
   - Qt 的 QPixmap 在 GPU 中缓存，`detach()` 后仍可能延迟释放
   - 高频预览时，多个 QPixmap 的 GPU 资源同时存在

3. **Qt 信号队列的对象持有**
   - 即使使用缓冲区，但信号队列仍会持有 ImageData 对象的引用
   - 多个预览任务完成时，多个对象引用在队列中排队

4. **每次 UI 更新都创建新的 QImage/QPixmap**
   - 没有复用机制，每次都是全新的对象
   - 旧的资源还没完全释放，新的就创建了

---

### 进一步优化方向

1. **Pipeline 内部数组复用**
   - 在 Pipeline 处理函数中，传入工作数组缓冲区
   - 避免每次创建新的 `working_array`

2. **QPixmap 复用机制**
   - 维护一个 QPixmap 缓冲区
   - 只有在尺寸变化时才重新创建

3. **信号队列清理**
   - 在 `_trigger_preview_update()` 开始时，处理 pending 的信号事件
   - 确保旧的结果在处理前就被丢弃

4. **Worker 完成后的立即清理**
   - 在 Worker 完成后，立即在主线程中清理临时对象
   - 不依赖 GC 的延迟回收