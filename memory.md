# Memory / Heap High-Water Mark Problem Description

## 1. Context

- Desktop app: Python + NumPy + PySide6/Qt.
- Core workload: interactive **preview pipeline** for large images:
  - User drags multiple sliders (exposure, contrast, curves, etc.).
  - Each change triggers `_trigger_preview_update()` → spawns a `_PreviewWorker` in a `QThreadPool`.
  - Worker runs a heavy `apply_full_pipeline(...)` + `convert_to_display_space(...)` on a **proxy image** (still large, e.g. several megapixels).
- Program runs on macOS; memory is observed via **Activity Monitor**.

## 2. What is observed (Symptoms)

1. While dragging sliders, the **“Memory” column (footprint)** in Activity Monitor grows over time:
   - 2 GB → 3.3 GB → 4.6 GB → … → 10 GB，
   - It looks **stair-step / monotonic** rather than “up-and-down”.
2. After cleaning all app-level caches and releasing current/preview images:
   - Internal memory report (`total_estimated_mb`) ≈ **1.3 GB**,
   - Activity Monitor **Real Memory** ≈ **2.2 GB**（reasonable, same order as 1.3 GB + Qt/VM overhead），
   - But the “Memory” / footprint still shows ~10 GB and does **not** drop back.
3. There is no evidence of many live workers piling up:
   - At most one `_PreviewWorker` is active at a time (`_preview_busy` + `_preview_pending` + `maxThreadCount(1)`).
   - Old previews and masks are explicitly released (`array = None`, caches cleared, etc.).

## 3. Terminology / Root Cause (Allocator / Heap High-Water Mark)

This is **not** a classic Python-level memory leak (no unbounded growth in live Python objects).  
The pattern matches a **heap high-water mark / allocator behavior** issue:

- The process uses a **heap allocator** (system `malloc` on macOS, underneath Python/NumPy/Qt).
- Each preview pipeline allocates several **large NumPy arrays** (full-resolution or proxy-resolution intermediates).
- When a pipeline needs **more memory than any previous pipeline**, the allocator:
  - Requests additional pages / arenas from the OS;
  - The process **heap grows** to satisfy this new **peak**.
- After the pipeline finishes:
  - NumPy arrays are freed at the Python level (no more references);
  - The allocator **reuses** those pages for future allocations but usually **does NOT return them to the OS**.
  - These freed regions become **free blocks inside the heap**, *not* a smaller heap.
- Therefore:
  - The **set of live objects** stays around ~1–2 GB,
  - But the **heap footprint** (what Activity Monitor reports as “Memory”) keeps the **maximum size ever requested**: this is the **high-water mark**.

That is exactly the “stair-step” pattern:

> 每次预览如果在内部需要的峰值内存稍微大于历史纪录，allocator 就再向前多要一块堆。  
> 之前那一阶虽然被逻辑上 free 了，但不会归还 OS，只是变成 allocator 内部的空洞。  
> 所以从外面看就是：**2 GB → 3.3 GB → 4.6 GB → …，只增不减**。

### Real Memory vs Footprint

- **Real Memory** ≈ current physical pages actually in use (e.g. 2.2 GB).
- The **“Memory”** column / footprint includes:
  - The whole expanded heap (including free blocks),
  - Compressed memory, file-backed mappings, caches, etc.
- Our own internal accounting (`total_estimated_mb` + `numpy_total_mb`) is in the same ballpark as **Real Memory**, not the 10 GB footprint, confirming this is **allocator high-water mark**, not “10 GB worth of live NumPy arrays”.

## 4. “Stair-Step” Problem Restated Precisely

> The preview pipeline allocates large temporary NumPy arrays on each invocation.  
> Peak memory usage for a single preview is not constant: depending on slider states / LUT combinations / intermediate stages, some previews require slightly more temporary memory than previous ones.  
> The underlying allocator satisfies each new peak by expanding the heap but rarely releases those pages back to the OS.  
> As a result, the process heap **monotonically increases** in size over the session, even though the actual live data volume remains roughly constant.  
> This manifests as Activity Monitor’s “Memory” usage climbing in a **stair-step fashion**, without dropping when previews finish or caches are cleared.

We want to **stop the stairs from climbing** (stabilize the peak), even if we cannot shrink the existing high-water mark without restarting the process.

## 5. What needs to be changed (high-level requirements)

Goal: **Make each preview use a bounded, stable amount of heap**, so the allocator can **reuse** existing large blocks instead of requesting new ones.

### 5.1 Introduce a reusable Preview Workspace (buffer pool)

- Add a `PreviewWorkspace` object that owns a small fixed set of large NumPy buffers:
  - e.g. 2–3 full-image buffers: `buf1`, `buf2`, `buf3`, with shape = proxy resolution, dtype = `float32`.
- Store it in the long-lived context (e.g. `ApplicationContext._preview_workspace`).
- When a preview is triggered:
  - `_trigger_preview_update()` obtains a `PreviewWorkspace` for the current proxy shape and passes it to `_PreviewWorker`.
  - The workspace is reused across all preview calls **as long as the proxy resolution does not change**.
- Only when the proxy resolution changes (e.g. different zoom / different image size) do we reallocate the workspace (one new step in the “ladder”).

### 5.2 Rewrite the preview pipeline to be buffer-reusing / in-place

For `TheEnlarger.apply_full_pipeline` and `ColorSpaceManager.convert_to_display_space`:

- Current style (likely):

  - Each stage returns a **new** NumPy array (`arr = f(arr)`),
  - Multiple full-res temporary arrays coexist in one pipeline call.

- Required refactor:

  - API takes a `PreviewWorkspace` (or explicit `out` buffers).
  - Each stage becomes an **in-place** or **out-parameter** version:

    - Example: `np.multiply(src, gain, out=dst)` instead of `src * gain`.
    - Functions like `_apply_exposure`, `_apply_contrast`, `_apply_s_curve`, tone mapping, matrix transforms, etc., should become `_xxx_inplace(src, params, out)`.

  - Pipeline uses **ping-pong** between `buf1`, `buf2`, `buf3`:

    - `buf1` ← normalized input  
    - `buf2` ← exposure  
    - `buf1` ← contrast  
    - `buf2` ← curve  
    - `buf3` ← color space transform / gamma / quantization …

  - Final result uses **one copy** into a fresh array only when constructing the returned `ImageData` object, or even reuses one of the workspace buffers if lifetime rules allow.

Effect:

- For a fixed proxy resolution, the maximum number of simultaneously-live full-res arrays becomes **constant**, e.g.:

  - 3–4 × (H × W × C × float32),

- The allocator sees **repeated reuse of the same large blocks** rather than new `np.empty` on every preview.
- Once the workspace is allocated, further previews do **not** increase heap high-water mark (no more “stairs”).

### 5.3 Limit and bound all caches

To avoid additional unbounded growth, all caches must have **hard caps** and **evict with LRU**:

- `ImageManager._proxy_cache`:
  - Use `OrderedDict` LRU.
  - `_max_cache_size` kept small (e.g. 8–16 proxies).
  - On eviction, clear the evicted proxy’s `.array` to release its NumPy buffer.

- `LUTProcessor._lut_cache`:
  - `OrderedDict` LRU with `_lut_cache_max_size` (e.g. 32 or 64).
  - Prevent slider exploration from accumulating thousands of LUT arrays.

- Color-space transform caches:
  - Same pattern: bounded size, LRU, and explicit clear on eviction.

Caches can still push up the **initial** high-water mark, but once their size stabilizes, they no longer drive the heap upward.

### 5.4 Aggressive cleanup hook (optional)

Provide a function such as `_clear_all_caches()` that:

- Clears:
  - current/preview image data,
  - proxy cache,
  - LUT caches,
  - color-space caches,
  - Qt’s `QPixmapCache`.
- Calls `gc.collect()`.

This will **not shrink** the allocator’s high-water mark, but it ensures:

- There are no “forgotten” large objects,
- Future allocations reuse existing arenas instead of growing further.

### 5.5 Optional, long-term: isolate previews in a separate process

If the high-water mark itself becomes unacceptable:

- Move the heavy preview pipeline into a **separate worker process** (e.g. via `multiprocessing` or `QProcess`).
- Main GUI process sends proxy + parameters, receives result image.
- When the worker process footprint becomes too large, the main process kills and restarts it.
- This is the only reliable way to **truly reset** the allocator high-water mark without restarting the whole application.

## 6. Success Criteria

After the refactor:

1. For a fixed proxy resolution:
   - Repeated slider drags **do not increase** Activity Monitor “Memory” / footprint beyond one or two initial steps.
   - Heap footprint stabilizes around a predictable constant (live data + workspace + bounded caches).
2. Internal accounting (`total_estimated_mb` + global NumPy heap scan) matches **Real Memory** order of magnitude.
3. Any further increase in footprint comes only from:
   - Changing proxy resolution (new workspace allocation),
   - Deliberately increasing cache caps,
   - Or opening very large new images.

The key design principle for the follow-up implementation is:

> **Eliminate per-preview dynamic heap growth by reusing a fixed set of large buffers and bounding all caches.**  
> This prevents the allocator from walking up the “stair-step” of ever-larger heap requests and keeps the memory footprint stable over time.