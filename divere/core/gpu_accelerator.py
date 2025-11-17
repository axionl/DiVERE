"""
GPU加速器模块 - 跨平台GPU加速支持
支持OpenCL, CUDA, Metal等多种GPU计算后端
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
import platform
from abc import ABC, abstractmethod

# 导入debug logger
from ..utils.debug_logger import debug, info, warning, error

# GPU库导入（可选）
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    import Metal
    import MetalPerformanceShaders as MPS
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class GPUComputeEngine(ABC):
    """GPU计算引擎抽象基类"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查计算引擎是否可用"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        pass
    
    @abstractmethod
    def density_inversion_gpu(self, image: np.ndarray, gamma: float, 
                             dmax: float, pivot: float) -> np.ndarray:
        """GPU加速的密度反相"""
        pass
    
    @abstractmethod
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        """GPU加速的曲线处理"""
        pass


class OpenCLEngine(GPUComputeEngine):
    """OpenCL计算引擎 - 跨平台支持"""
    
    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.program = None
        self._initialize()
    
    def _is_problematic_windows_device(self, device, platform) -> bool:
        """检查是否为Windows上可能有问题的设备"""
        try:
            device_name = device.name.lower()
            vendor_name = getattr(device, 'vendor', '').lower()
            platform_name = platform.name.lower()
            
            # Intel集显的OpenCL驱动在Windows上经常有问题
            intel_keywords = ['intel', 'hd graphics', 'iris', 'uhd graphics']
            if any(keyword in device_name or keyword in vendor_name for keyword in intel_keywords):
                debug(f"    检测到Intel集显设备: {device.name}", "GPU")
                return True
            
            # 某些老旧的OpenCL实现
            if 'microsoft' in platform_name or 'basic render driver' in device_name:
                debug(f"    检测到Microsoft基础渲染驱动: {device.name}", "GPU")
                return True
            
            # 内存过小的设备（通常是集显）
            memory_mb = device.global_mem_size // (1024 * 1024)
            if memory_mb < 512:  # 小于512MB的设备通常是集显
                debug(f"    设备内存过小({memory_mb}MB)，可能是集显", "GPU")
                return True
                
            return False
            
        except Exception as e:
            warning(f"    无法检查设备兼容性: {e}", "GPU")
            return True  # 保守策略：无法检查的设备认为有问题
    
    def _initialize(self):
        """初始化OpenCL环境"""
        if not OPENCL_AVAILABLE:
            debug("PyOpenCL未安装，跳过OpenCL引擎初始化", "GPU")
            return
        
        try:
            info("开始初始化OpenCL环擎", "GPU")
            
            # 寻找最佳GPU设备
            platforms = cl.get_platforms()
            info(f"发现{len(platforms)}个OpenCL平台", "GPU")
            
            best_device = None
            best_compute_units = 0
            best_memory_mb = 0
            
            for i, platform in enumerate(platforms):
                debug(f"平台[{i}]: {platform.name} ({platform.vendor})", "GPU")
                
                try:
                    devices = platform.get_devices()
                    for j, device in enumerate(devices):
                        debug(f"  设备[{j}]: {device.name} ({device.type})", "GPU")
                        
                        # 只考虑GPU设备
                        if not (device.type & cl.device_type.GPU):
                            debug(f"    跳过非GPU设备", "GPU")
                            continue
                        
                        # 获取设备信息
                        compute_units = device.max_compute_units
                        memory_mb = device.global_mem_size // (1024 * 1024)
                        
                        debug(f"    计算单元: {compute_units}, 显存: {memory_mb}MB", "GPU")
                        
                        # Windows特定检查：过滤可能有问题的设备
                        import platform as sys_platform
                        if sys_platform.system() == 'Windows':
                            if self._is_problematic_windows_device(device, platform):
                                warning(f"    跳过可能有问题的Windows设备: {device.name}", "GPU")
                                continue
                        
                        # 内存检查：至少需要256MB
                        if memory_mb < 256:
                            warning(f"    设备内存不足({memory_mb}MB < 256MB)，跳过", "GPU")
                            continue
                        
                        # 选择最佳设备：优先考虑计算单元，其次内存
                        is_better = (compute_units > best_compute_units or 
                                    (compute_units == best_compute_units and memory_mb > best_memory_mb))
                        
                        if is_better:
                            best_device = device
                            best_compute_units = compute_units
                            best_memory_mb = memory_mb
                            debug(f"    选为最佳设备候选", "GPU")
                            
                except cl.Error as e:
                    warning(f"  无法获取平台{i}的设备信息: {e}", "GPU")
                    continue
            
            if best_device:
                info(f"选择OpenCL设备: {best_device.name} ({best_compute_units}CU, {best_memory_mb}MB)", "GPU")
                
                # 创建上下文和队列
                self.device = best_device
                self.context = cl.Context([best_device])
                self.queue = cl.CommandQueue(self.context)
                
                # 构建内核
                self._build_kernels()
                
                if self.program is not None:
                    info("OpenCL引擎初始化成功", "GPU")
                else:
                    error("OpenCL内核编译失败，引擎不可用", "GPU")
            else:
                warning("未找到合适的OpenCL GPU设备", "GPU")
                
        except Exception as e:
            error(f"OpenCL初始化失败: {e}", "GPU")
            import traceback
            debug(f"OpenCL初始化异常详情:\n{traceback.format_exc()}", "GPU")
    
    def _build_kernels(self):
        """编译OpenCL内核"""
        kernel_source = '''
        __kernel void density_inversion(__global const float* input,
                                       __global float* output,
                                       const float gamma,
                                       const float dmax,
                                       const float pivot,
                                       const int invert,
                                       const int size)
        {
            int i = get_global_id(0);
            if (i >= size) return;

            // 避免log(0)
            float safe_val = fmax(input[i], 1e-10f);

            // 密度反相计算（根据 invert 控制正负号）
            float log_img = log10(safe_val);
            float original_density = invert ? -log_img : log_img;
            float adjusted_density = pivot + (original_density - pivot) * gamma - dmax;

            // 转回线性空间
            output[i] = pow(10.0f, adjusted_density);
        }
        
        __kernel void curve_lut_apply(__global const float* input,
                                     __global float* output,
                                     __global const float* lut,
                                     const int image_size,
                                     const int lut_size)
        {
            int i = get_global_id(0);
            if (i >= image_size) return;
            
            // 归一化到LUT索引范围
            float normalized = 1.0f - clamp(input[i] * 0.000152587890625f, 0.0f, 1.0f);
            float index_f = normalized * (lut_size - 1);
            int index = (int)index_f;
            
            // 简单索引（可以改为插值）
            index = clamp(index, 0, lut_size - 1);
            output[i] = lut[index];
        }
        '''
        
        try:
            # 尝试编译内核
            debug("开始编译OpenCL内核", "GPU")
            info(f"为设备{self.device.name}编译OpenCL内核", "GPU")
            self.program = cl.Program(self.context, kernel_source).build()
            info("OpenCL内核编译成功", "GPU")
            
            # 验证内核函数是否可用
            try:
                density_kernel = self.program.density_inversion
                curve_kernel = self.program.curve_lut_apply
                debug("内核函数验证成功", "GPU")
            except AttributeError as e:
                error(f"内核函数验证失败: {e}", "GPU")
                self.program = None
                
        except cl.CompileError as e:
            error(f"OpenCL内核编译错误: {e}", "GPU")
            debug(f"编译错误详情:\n{e}", "GPU")
            self.program = None
        except cl.Error as e:
            error(f"OpenCL编译时发生错误: {e}", "GPU")
            self.program = None
        except Exception as e:
            error(f"OpenCL内核编译发生未知错误: {e}", "GPU")
            import traceback
            debug(f"编译异常详情:\n{traceback.format_exc()}", "GPU")
            self.program = None
    
    def is_available(self) -> bool:
        """检查OpenCL是否可用"""
        return (OPENCL_AVAILABLE and 
                self.context is not None and 
                self.queue is not None and 
                self.program is not None)
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "name": self.device.name,
            "type": "OpenCL",
            "compute_units": self.device.max_compute_units,
            "global_memory_mb": self.device.global_mem_size // 1024 // 1024,
            "max_work_group_size": self.device.max_work_group_size
        }
    
    def density_inversion_gpu(self, image: np.ndarray, gamma: float,
                             dmax: float, pivot: float, invert: bool = True) -> np.ndarray:
        """GPU加速的密度反相"""
        if not self.is_available():
            raise RuntimeError("OpenCL不可用")

        debug(f"OpenCL密度反相处理: 图像{image.shape}, gamma={gamma:.3f}, invert={invert}", "GPU")
        
        try:
            # 展平数组以简化处理
            original_shape = image.shape
            image_flat = image.flatten().astype(np.float32)
            output_flat = np.zeros_like(image_flat)
            
            debug(f"处理{len(image_flat)}个像素", "GPU")
            
            # 检查内存大小
            memory_needed = len(image_flat) * 8  # input + output buffers
            device_memory = self.device.global_mem_size
            if memory_needed > device_memory * 0.8:  # 不使用超过80%的显存
                warning(f"内存需求({memory_needed//1024//1024}MB)接近设备限制({device_memory//1024//1024}MB)", "GPU")
            
            # 创建OpenCL缓冲区
            mf = cl.mem_flags
            input_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                 hostbuf=image_flat)
            output_buf = cl.Buffer(self.context, mf.WRITE_ONLY, image_flat.nbytes)
            
            # 执行内核
            global_size = (len(image_flat),)
            event = self.program.density_inversion(
                self.queue, global_size, None,
                input_buf, output_buf,
                np.float32(gamma), np.float32(dmax), np.float32(pivot),
                np.int32(1 if invert else 0),  # OpenCL 使用 int 表示 bool
                np.int32(len(image_flat))
            )
            
            # 等待执行完成
            event.wait()
            
            # 读取结果
            cl.enqueue_copy(self.queue, output_flat, output_buf)
            
            debug("OpenCL密度反相处理完成", "GPU")
            
            # 恢复原始形状
            return output_flat.reshape(original_shape)
            
        except cl.MemoryError as e:
            error(f"OpenCL内存不足: {e}", "GPU")
            raise RuntimeError(f"GPU内存不足: {e}")
        except cl.Error as e:
            error(f"OpenCL执行错误: {e}", "GPU")
            raise RuntimeError(f"GPU执行失败: {e}")
        except Exception as e:
            error(f"OpenCL密度反相处理发生未知错误: {e}", "GPU")
            import traceback
            debug(f"异常详情:\n{traceback.format_exc()}", "GPU")
            raise
    
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        """GPU加速的曲线处理"""
        if not self.is_available():
            raise RuntimeError("OpenCL不可用")
        
        debug(f"OpenCL曲线处理: 图像{density_array.shape}, LUT大小{len(lut)}", "GPU")
        
        try:
            # 展平数组
            original_shape = density_array.shape
            density_flat = density_array.flatten().astype(np.float32)
            output_flat = np.zeros_like(density_flat)
            lut_float = lut.astype(np.float32)
            
            debug(f"处理{len(density_flat)}个像素，LUT:{len(lut_float)}个条目", "GPU")
            
            # 检查内存大小
            memory_needed = len(density_flat) * 8 + len(lut_float) * 4  # buffers
            device_memory = self.device.global_mem_size
            if memory_needed > device_memory * 0.8:
                warning(f"内存需求({memory_needed//1024//1024}MB)接近设备限制({device_memory//1024//1024}MB)", "GPU")
            
            # 创建OpenCL缓冲区
            mf = cl.mem_flags
            input_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                 hostbuf=density_flat)
            output_buf = cl.Buffer(self.context, mf.WRITE_ONLY, density_flat.nbytes)
            lut_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=lut_float)
            
            # 执行内核
            global_size = (len(density_flat),)
            event = self.program.curve_lut_apply(
                self.queue, global_size, None,
                input_buf, output_buf, lut_buf,
                np.int32(len(density_flat)), np.int32(len(lut_float))
            )
            
            # 等待执行完成
            event.wait()
            
            # 读取结果
            cl.enqueue_copy(self.queue, output_flat, output_buf)
            
            debug("OpenCL曲线处理完成", "GPU")
            
            # 恢复原始形状
            return output_flat.reshape(original_shape)
            
        except cl.MemoryError as e:
            error(f"OpenCL内存不足: {e}", "GPU")
            raise RuntimeError(f"GPU内存不足: {e}")
        except cl.Error as e:
            error(f"OpenCL执行错误: {e}", "GPU")
            raise RuntimeError(f"GPU执行失败: {e}")
        except Exception as e:
            error(f"OpenCL曲线处理发生未知错误: {e}", "GPU")
            import traceback
            debug(f"异常详情:\n{traceback.format_exc()}", "GPU")
            raise


class CUDAEngine(GPUComputeEngine):
    """CUDA计算引擎 - NVIDIA GPU专用"""
    
    def __init__(self):
        self._available = CUDA_AVAILABLE
        if self._available:
            try:
                # 测试CUDA可用性
                debug("尝试初始化CUDA引擎", "GPU")
                cp.cuda.Device(0).use()
                info("CUDA引擎初始化成功", "GPU")
            except Exception as e:
                warning(f"CUDA引擎初始化失败: {e}", "GPU")
                self._available = False
        else:
            debug("CuPy未安装，跳过CUDA引擎", "GPU")
    
    def is_available(self) -> bool:
        return self._available
    
    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_available():
            return {"available": False}
        
        device = cp.cuda.Device()
        return {
            "available": True,
            "name": device.attributes["Name"],
            "type": "CUDA",
            "compute_capability": device.compute_capability,
            "memory_mb": device.mem_info[1] // 1024 // 1024
        }
    
    def density_inversion_gpu(self, image: np.ndarray, gamma: float,
                             dmax: float, pivot: float, invert: bool = True) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("CUDA不可用")

        # 转移到GPU
        image_gpu = cp.asarray(image)

        # 避免log(0)
        safe_img = cp.maximum(image_gpu, 1e-10)

        # 密度反相计算（根据 invert 控制正负号）
        log_img = cp.log10(safe_img)
        original_density = -log_img if invert else log_img
        adjusted_density = pivot + (original_density - pivot) * gamma - dmax
        result_gpu = cp.power(10.0, adjusted_density)

        # 转回CPU
        return cp.asnumpy(result_gpu)
    
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("CUDA不可用")
        
        # 转移到GPU
        density_gpu = cp.asarray(density_array)
        lut_gpu = cp.asarray(lut)
        
        # 这里需要实现CUDA版本的LUT查表
        # 简化版本：使用CuPy的interp
        # 实际实现需要优化的索引操作
        
        # 转回CPU（占位实现）
        return cp.asnumpy(density_gpu)


class MetalEngine(GPUComputeEngine):
    """Metal计算引擎 - macOS原生最优性能"""
    
    def __init__(self):
        self.device = None
        self.command_queue = None
        self.library = None
        self._initialize()
    
    def _initialize(self):
        """初始化Metal环境"""
        if not METAL_AVAILABLE:
            return
        
        try:
            # 创建Metal设备和命令队列
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device:
                self.command_queue = self.device.newCommandQueue()
                self._create_compute_library()
        except Exception as e:
            error(f"Metal初始化失败: {e}", "GPU")
    
    def _create_compute_library(self):
        """创建Metal计算着色器库"""
        # Metal着色器源代码
        metal_source = '''
        #include <metal_stdlib>
        using namespace metal;
        
        // 密度反相计算着色器（高精度版本）
        kernel void density_inversion(device const float* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     constant float& gamma [[buffer(2)]],
                                     constant float& dmax [[buffer(3)]],
                                     constant float& pivot [[buffer(4)]],
                                     constant bool& invert [[buffer(5)]],
                                     uint index [[thread_position_in_grid]])
        {
            // 确保与CPU版本相同的精度处理
            float safe_val = max(input[index], 1e-10f);

            // 密度反相计算 - 根据 invert 控制正负号
            float log_img = log10(safe_val);
            float original_density = invert ? -log_img : log_img;
            float adjusted_density = pivot + (original_density - pivot) * gamma - dmax;

            // 转回线性空间 - 使用precise标志确保精度
            output[index] = precise::pow(10.0f, adjusted_density);
        }
        
        // LUT查表着色器
        kernel void lut_apply(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             device const float* lut [[buffer(2)]],
                             constant uint& lut_size [[buffer(3)]],
                             uint index [[thread_position_in_grid]])
        {
            // 归一化到LUT索引范围
            float normalized = 1.0f - clamp(input[index] * 0.000152587890625f, 0.0f, 1.0f);
            float index_f = normalized * (lut_size - 1);
            uint lut_index = uint(index_f);
            
            // 边界检查
            lut_index = min(lut_index, lut_size - 1);
            output[index] = lut[lut_index];
        }
        
        // 矩阵乘法着色器（用于色彩空间转换）
        kernel void matrix_multiply_3x3(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       device const float* matrix [[buffer(2)]],
                                       uint index [[thread_position_in_grid]])
        {
            uint pixel_index = index / 3;
            uint component = index % 3;
            uint base_index = pixel_index * 3;
            
            float result = 0.0f;
            for (uint i = 0; i < 3; i++) {
                result += matrix[component * 3 + i] * input[base_index + i];
            }
            output[index] = result;
        }
        '''
        
        try:
            # 编译Metal着色器
            library = self.device.newLibraryWithSource_options_error_(
                metal_source, None, None
            )
            if library[0]:
                self.library = library[0]
            else:
                error(f"Metal着色器编译失败: {library[1]}", "GPU")
        except Exception as e:
            error(f"Metal着色器创建失败: {e}", "GPU")
    
    def is_available(self) -> bool:
        """检查Metal是否可用"""
        return (METAL_AVAILABLE and 
                self.device is not None and 
                self.command_queue is not None and 
                self.library is not None)
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "name": self.device.name(),
            "type": "Metal",
            "unified_memory": self.device.hasUnifiedMemory(),
            "max_buffer_mb": self.device.maxBufferLength() // 1024 // 1024,
            "low_power": self.device.isLowPower()
        }
    
    def density_inversion_gpu(self, image: np.ndarray, gamma: float,
                             dmax: float, pivot: float, invert: bool = True) -> np.ndarray:
        """Metal加速的密度反相"""
        if not self.is_available():
            raise RuntimeError("Metal不可用")
        
        # 展平数组
        original_shape = image.shape
        image_flat = image.flatten().astype(np.float32)
        output_flat = np.zeros_like(image_flat)
        
        # 创建Metal缓冲区
        input_buffer = self.device.newBufferWithBytes_length_options_(
            image_flat.tobytes(), 
            len(image_flat) * 4,  # float32 = 4 bytes
            Metal.MTLResourceStorageModeShared
        )
        
        output_buffer = self.device.newBufferWithLength_options_(
            len(image_flat) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        # 参数缓冲区
        gamma_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([gamma], dtype=np.float32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        dmax_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([dmax], dtype=np.float32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        pivot_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([pivot], dtype=np.float32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )

        # invert 参数缓冲区（Metal 支持 bool 类型）
        invert_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([invert], dtype=np.bool_).tobytes(), 1,
            Metal.MTLResourceStorageModeShared
        )

        # 创建计算管线
        function = self.library.newFunctionWithName_("density_inversion")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        # 创建命令缓冲区和编码器
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        # 设置管线和缓冲区
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(gamma_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(dmax_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(pivot_buffer, 0, 4)
        compute_encoder.setBuffer_offset_atIndex_(invert_buffer, 0, 5)
        
        # 计算线程网格
        threads_per_threadgroup = Metal.MTLSize(256, 1, 1)
        threadgroups = Metal.MTLSize(
            (len(image_flat) + 255) // 256, 1, 1
        )
        
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # 读取结果 - 使用正确的Metal缓冲区访问方法
        result_ptr = output_buffer.contents()
        # 创建numpy数组直接指向Metal缓冲区内存
        result_array = np.frombuffer(
            result_ptr.as_buffer(len(image_flat) * 4), 
            dtype=np.float32
        ).copy()  # copy确保数据独立
        
        return result_array.reshape(original_shape)
    
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        """Metal加速的曲线处理"""
        if not self.is_available():
            raise RuntimeError("Metal不可用")
        
        # 展平数组
        original_shape = density_array.shape
        density_flat = density_array.flatten().astype(np.float32)
        output_flat = np.zeros_like(density_flat)
        lut_float = lut.astype(np.float32)
        
        # 创建Metal缓冲区
        input_buffer = self.device.newBufferWithBytes_length_options_(
            density_flat.tobytes(), 
            len(density_flat) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        output_buffer = self.device.newBufferWithLength_options_(
            len(density_flat) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        lut_buffer = self.device.newBufferWithBytes_length_options_(
            lut_float.tobytes(),
            len(lut_float) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        lut_size_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([len(lut_float)], dtype=np.uint32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        # 创建计算管线
        function = self.library.newFunctionWithName_("lut_apply")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        # 创建命令缓冲区和编码器
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        # 设置管线和缓冲区
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(lut_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(lut_size_buffer, 0, 3)
        
        # 计算线程网格
        threads_per_threadgroup = Metal.MTLSize(256, 1, 1)
        threadgroups = Metal.MTLSize(
            (len(density_flat) + 255) // 256, 1, 1
        )
        
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # 读取结果 - 使用正确的Metal缓冲区访问方法
        result_ptr = output_buffer.contents()
        # 创建numpy数组直接指向Metal缓冲区内存
        result_array = np.frombuffer(
            result_ptr.as_buffer(len(density_flat) * 4), 
            dtype=np.float32
        ).copy()  # copy确保数据独立
        
        return result_array.reshape(original_shape)


class GPUAccelerator:
    """GPU加速器 - 统一的GPU加速接口"""
    
    def __init__(self):
        self.engines = []
        self.active_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化所有可用的计算引擎"""
        import os
        
        # 检查是否禁用GPU加速
        if os.environ.get('DIVERE_DISABLE_GPU', '').lower() in ('1', 'true', 'yes'):
            info("检测到DIVERE_DISABLE_GPU环境变量，禁用GPU加速", "GPU")
            return
        
        # 根据平台调整引擎优先级
        if platform.system() == 'Windows':
            # Windows: CUDA > OpenCL > Metal（Metal在Windows上不可用）
            engines_to_try = [
                ("CUDA", CUDAEngine),
                ("OpenCL", OpenCLEngine),
                ("Metal", MetalEngine),
            ]
            info("Windows平台：优先尝试CUDA，然后OpenCL", "GPU")
        elif platform.system() == 'Darwin':
            # macOS: Metal > OpenCL > CUDA（优先原生加速）
            engines_to_try = [
                ("Metal", MetalEngine),
                ("OpenCL", OpenCLEngine), 
                ("CUDA", CUDAEngine),
            ]
            info("macOS平台：优先尝试Metal，然后OpenCL", "GPU")
        else:
            # Linux等: OpenCL > CUDA > Metal
            engines_to_try = [
                ("OpenCL", OpenCLEngine),
                ("CUDA", CUDAEngine),
                ("Metal", MetalEngine),
            ]
            info("Linux平台：优先尝试OpenCL，然后CUDA", "GPU")
        
        info("开始初始化GPU引擎", "GPU")
        
        for name, engine_class in engines_to_try:
            try:
                debug(f"尝试初始化{name}引擎", "GPU")
                engine = engine_class()
                if engine.is_available():
                    self.engines.append((name, engine))
                    if self.active_engine is None:
                        self.active_engine = engine
                        info(f"🚀 使用GPU引擎: {name}", "GPU")
                        # 获取设备信息
                        device_info = engine.get_device_info()
                        debug(f"设备信息: {device_info}", "GPU")
                else:
                    debug(f"{name}引擎不可用", "GPU")
            except Exception as e:
                warning(f"⚠️  {name}引擎初始化失败: {e}", "GPU")
                debug(f"{name}引擎初始化异常详情: {e}", "GPU")
        
        if not self.engines:
            warning("未找到可用的GPU引擎，将使用CPU计算", "GPU")
        else:
            available_engines = [name for name, _ in self.engines]
            info(f"可用GPU引擎: {available_engines}", "GPU")
    
    def is_available(self) -> bool:
        """检查是否有可用的GPU加速"""
        return self.active_engine is not None
    
    def get_available_engines(self) -> List[str]:
        """获取所有可用的引擎列表"""
        return [name for name, engine in self.engines]
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取当前激活设备的信息"""
        if not self.is_available():
            return {"available": False, "fallback": "CPU"}
        
        info = self.active_engine.get_device_info()
        info["engines_available"] = self.get_available_engines()
        return info
    
    def set_active_engine(self, engine_name: str) -> bool:
        """切换激活的计算引擎"""
        for name, engine in self.engines:
            if name == engine_name:
                self.active_engine = engine
                info(f"🔄 切换到GPU引擎: {engine_name}", "GPU")
                return True
        return False
    
    def density_inversion_accelerated(self, image: np.ndarray, gamma: float,
                                     dmax: float, pivot: float, invert: bool = True) -> np.ndarray:
        """GPU加速的密度反相，自动回退到CPU"""
        if not self.is_available():
            debug("没有可用的GPU引擎，使用CPU处理密度反相", "GPU")
            return self._density_inversion_cpu(image, gamma, dmax, pivot, invert)

        try:
            debug(f"使用{type(self.active_engine).__name__}处理密度反相", "GPU")
            result = self.active_engine.density_inversion_gpu(image, gamma, dmax, pivot, invert)
            debug("GPU密度反相处理成功", "GPU")
            return result
        except Exception as e:
            error(f"GPU加速失败，回退到CPU: {e}", "GPU")
            debug(f"GPU失败详情: {e}", "GPU")
            return self._density_inversion_cpu(image, gamma, dmax, pivot, invert)
    
    def curve_processing_accelerated(self, density_array: np.ndarray, 
                                   lut: np.ndarray) -> np.ndarray:
        """GPU加速的曲线处理，自动回退到CPU"""
        if not self.is_available():
            debug("没有可用的GPU引擎，使用CPU处理曲线", "GPU")
            return self._curve_processing_cpu(density_array, lut)
        
        try:
            debug(f"使用{type(self.active_engine).__name__}处理曲线", "GPU")
            result = self.active_engine.curve_processing_gpu(density_array, lut)
            debug("GPU曲线处理成功", "GPU")
            return result
        except Exception as e:
            error(f"GPU加速失败，回退到CPU: {e}", "GPU")
            debug(f"GPU失败详情: {e}", "GPU")
            return self._curve_processing_cpu(density_array, lut)
    
    def _density_inversion_cpu(self, image: np.ndarray, gamma: float,
                              dmax: float, pivot: float, invert: bool = True) -> np.ndarray:
        """CPU版本的密度反相（回退方案）"""
        debug(f"CPU密度反相处理: 图像{image.shape}, gamma={gamma:.3f}, invert={invert}", "GPU")
        safe_img = np.maximum(image, 1e-10)
        log_img = np.log10(safe_img)
        original_density = -log_img if invert else log_img
        adjusted_density = pivot + (original_density - pivot) * gamma - dmax
        result = np.power(10.0, adjusted_density)
        debug("CPU密度反相处理完成", "GPU")
        return result
    
    def _curve_processing_cpu(self, density_array: np.ndarray, 
                             lut: np.ndarray) -> np.ndarray:
        """CPU版本的曲线处理（回退方案）- 使用高精度插值"""
        debug(f"CPU曲线处理: 图像{density_array.shape}, LUT大小{len(lut)}", "GPU")
        
        # 高精度线性插值实现，避免简单索引造成的量化误差
        inv_range = 1.0 / 6.5536  # LOG65536的倒数
        normalized = 1.0 - np.clip(density_array * inv_range, 0.0, 1.0)
        
        # 使用NumPy的高精度插值而不是简单索引
        lut_indices = np.linspace(0.0, 1.0, len(lut), dtype=np.float64)
        result = np.interp(normalized.flatten(), lut_indices, lut).astype(density_array.dtype)
        
        debug("CPU曲线处理完成", "GPU")
        return result.reshape(density_array.shape)


# 全局GPU加速器实例
_gpu_accelerator = None

def get_gpu_accelerator() -> GPUAccelerator:
    """获取全局GPU加速器实例"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator
