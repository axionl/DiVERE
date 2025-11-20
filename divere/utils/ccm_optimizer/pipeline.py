#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiVERE管线模拟器

严格按照DiVERE的处理流程模拟色彩处理管线，用于CCM优化。

处理流程:
1. 注册自定义输入色彩变换（基于优化参数的基色）
2. 输入色彩变换 → 工作色彩空间(ACEScg)
3. 密度反转 (RGB → 密度)
4. RGB增益调整
5. 密度曲线处理 (跳过，因为优化中不包含曲线)
6. 返回线性ACEScg RGB

注意: 不包含曲线处理和输出色彩转换
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import sys

# 添加项目根目录到路径，以便导入divere模块
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from divere.core.color_space import ColorSpaceManager
    from divere.core.data_types import ImageData, ColorGradingParams
    from divere.core.math_ops import FilmMathOps
    import colour
except ImportError as e:
    print(f"Warning: 无法导入divere模块: {e}")
    print("请确保在DiVERE项目根目录下运行")

class DiVEREPipelineSimulator:
    """DiVERE色彩处理管线模拟器"""
    
    def __init__(self, verbose=False, working_colorspace="ACEScg", color_space_manager=None):
        """初始化管线模拟器
        
        Args:
            verbose: 控制详细输出
            working_colorspace: 当前工作色彩空间名称
            color_space_manager: ColorSpaceManager实例，用于获取工作空间信息
        """
        self.verbose = verbose  # 控制详细输出
        self.working_colorspace = working_colorspace
        self.color_space_manager = color_space_manager
        # 初始化真实的DiVERE数学运算引擎
        self.math_ops = FilmMathOps()
        
        # 获取工作空间的基色和白点信息
        self._working_space_info = self._get_working_space_info()
    
    def _get_working_space_info(self):
        """获取工作空间的基色和白点信息"""
        if self.color_space_manager:
            try:
                space_info = self.color_space_manager.get_color_space_info(self.working_colorspace)
                if space_info:

                    return space_info
                else:
                    raise ValueError(f"找不到工作空间定义: {self.working_colorspace}")
            except Exception as e:
                if self.verbose:
                    print(f"错误: 无法获取工作空间信息 {self.working_colorspace}: {e}")
                raise ValueError(f"无法获取工作空间信息 {self.working_colorspace}: {e}")
        
        # 不应该回退到默认值，应该抛出错误
        raise ValueError(f"ColorSpaceManager未提供，无法获取工作空间信息: {self.working_colorspace}")

    @staticmethod
    def primaries_to_xyz_matrix(primaries, white_point):
        """从primaries和白点计算到XYZ的转换矩阵"""
        # 确保primaries是正确形状的数组
        primaries = np.asarray(primaries)
        if primaries.ndim == 1:
            # 如果是一维数组，重塑为(3,2)
            primaries = primaries.reshape(3, 2)
        
        # 将xy坐标转换为XYZ
        xyz_primaries = np.zeros((3, 3))
        for i in range(3):
            x, y = primaries[i]
            # 防止除零错误
            if abs(y) < 1e-10:
                y = 1e-10
            z = 1 - x - y
            xyz_primaries[:, i] = [x/y, 1.0, z/y]
        
        # 白点XYZ
        white_point = np.asarray(white_point)
        wx, wy = white_point
        # 防止除零错误
        if abs(wy) < 1e-10:
            wy = 1e-10
        wz = 1 - wx - wy
        white_xyz = np.array([wx/wy, 1.0, wz/wy])
        
        # 计算scaling factors，增强数值稳定性
        try:
            # 检查矩阵条件数，避免数值不稳定
            cond_num = np.linalg.cond(xyz_primaries)
            if cond_num > 1e12:  # 条件数过大，使用伪逆
                scaling = np.linalg.pinv(xyz_primaries) @ white_xyz
            else:
                scaling = np.linalg.solve(xyz_primaries, white_xyz)
        except (np.linalg.LinAlgError, RuntimeError, MemoryError):
            # 回退到默认缩放因子
            scaling = np.array([1.0, 1.0, 1.0])
        
        # 构建最终的转换矩阵
        return xyz_primaries * scaling[np.newaxis, :]
    
    def simulate_full_pipeline(self, input_rgb_patches: Dict[str, Tuple[float, float, float]],
                              primaries_xy: np.ndarray,
                              white_point_xy: Optional[np.ndarray] = None,
                              gamma: float = 2.0,
                              dmax: float = 2.0,
                              r_gain: float = 0.0,
                              b_gain: float = 0.0,
                              correction_matrix: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float, float]]:
        """
        一体化DiVERE管线模拟器 - 所有操作在一个函数内完成。
        
        完整流程:
        1. 注册输入色彩变换，转换到当前工作空间
        2. original_density = -log10(safe_rgb)  
        3. adjusted_density = pivot + (original_density - pivot) * gamma - dmax
        4. 应用密度校正矩阵 (如果有)
        5. 添加 R/B 增益
        6. rgb = 10^adjusted_density
        
        Args:
            input_rgb_patches: 输入RGB色块字典
            primaries_xy: 自定义色彩空间的RGB基色xy坐标 (3, 2)
            white_point_xy: 白点xy坐标，默认D65
            gamma: 密度反差参数
            dmax: 最大密度参数  
            r_gain: R通道增益
            b_gain: B通道增益
            correction_matrix: 密度校正矩阵 (3x3)
        
        Returns:
            处理后的RGB色块字典
        """
        # ===== 步骤1: 色彩空间注册和转换 =====
        
        # 设置默认白点
        if white_point_xy is None:
            white_point_xy = np.array([0.32168, 0.33767])  # D60 很重要！Optimizer不优化白点。
        
        # 从primaries和white_point转换到当前工作空间
        # 获取当前工作空间的基色和白点定义
        ws_info = self._working_space_info
        working_primaries = ws_info['primaries']  # 已经是numpy数组格式
        working_white_point = np.array(ws_info['white_point'])
        print(working_primaries)
        print(working_white_point)
        
        # 计算转换矩阵
        input_to_xyz = self.primaries_to_xyz_matrix(primaries_xy, white_point_xy)
        working_to_xyz = self.primaries_to_xyz_matrix(working_primaries, working_white_point)
        xyz_to_working = np.linalg.inv(working_to_xyz)
        
        # 组合转换矩阵：输入空间 -> XYZ -> 当前工作空间
        input_to_working = xyz_to_working @ input_to_xyz

        # 与主管线一致：白点适应增益（简化版，匹配 ColorSpaceManager 的实现）
        def _xy_to_XYZ_normalized(xy):
            x, y = float(xy[0]), float(xy[1])
            if abs(y) < 1e-10:
                y = 1e-10
            X = x / y
            Y = 1.0
            Z = (1.0 - x - y) / y
            return np.array([X, Y, Z], dtype=float)

        src_white = np.array(white_point_xy if white_point_xy is not None else [0.3127, 0.3290], dtype=float)
        dst_white = working_white_point
        src_white_XYZ = _xy_to_XYZ_normalized(src_white)
        dst_white_XYZ = _xy_to_XYZ_normalized(dst_white)
        # 简化增益：分量比值并裁剪
        with np.errstate(divide='ignore', invalid='ignore'):
            gain_vector = np.divide(dst_white_XYZ, src_white_XYZ)
        gain_vector = np.clip(gain_vector, 0.1, 10.0)
        
        # 转换到当前工作空间并应用白点增益（与主管线一致在矩阵转换阶段进行）
        working_space_patches = {}
        for patch_id, (r, g, b) in input_rgb_patches.items():
            input_rgb = np.array([r, g, b])
            working_rgb = input_to_working @ input_rgb
            # 应用白点适应增益（逐分量）
            working_rgb = working_rgb * gain_vector
            working_space_patches[patch_id] = tuple(working_rgb.tolist())
        
        # ===== 步骤2-6: 使用真实的DiVERE核心处理 =====
        
        # 将色块数据转换为图像数组格式以使用DiVERE处理函数
        patch_array = np.array([list(working_space_patches.values())]).reshape(1, len(working_space_patches), 3)
        
        # 构造临时参数对象用于DiVERE处理
        from divere.core.data_types import ColorGradingParams
        temp_params = ColorGradingParams()
        temp_params.density_gamma = gamma
        temp_params.density_dmax = dmax
        temp_params.rgb_gains = (r_gain, 0.0, b_gain)  # G通道固定为0
        temp_params.enable_density_inversion = True
        temp_params.enable_density_matrix = (correction_matrix is not None)
        temp_params.enable_rgb_gains = True
        temp_params.enable_density_curve = False  # 优化中不包含曲线
        
        # 如果有校正矩阵，直接设置矩阵数据（不通过名称）
        if correction_matrix is not None:
            temp_params.density_matrix = correction_matrix  # 直接设置numpy数组
        
        # 使用真实的DiVERE数学管线处理
        processed_array = self.math_ops.apply_full_math_pipeline(
            patch_array, 
            temp_params,
            include_curve=False,
            enable_density_inversion=True,
            use_optimization=False
        )
        
        # 将处理结果转换回字典格式
        final_rgb_patches = {}
        patch_ids = list(working_space_patches.keys())
        for i, patch_id in enumerate(patch_ids):
            rgb_values = processed_array[0, i, :]
            final_rgb_patches[patch_id] = tuple(rgb_values.tolist())
        
        if self.verbose:
            print(f"✓ 使用真实DiVERE管线处理完成，处理了 {len(final_rgb_patches)} 个色块")
        return final_rgb_patches

if __name__ == "__main__":
    # 简单的测试代码
    print("DiVERE管线模拟器已加载") 
