"""
IDT优化器

使用CMA-ES算法优化3x3线性变换矩阵，将初始RGB向量变换到目标RGB向量。
基于divere现有的CMA-ES实现进行适配。
"""

import numpy as np
from typing import Dict, Optional, Callable, Tuple
import sys
from pathlib import Path

# 导入CMA-ES库
try:
    import cma
except ImportError:
    raise ImportError("请安装cma库: pip install cma")


class IDTOptimizer:
    """IDT优化器"""
    
    def __init__(self, status_callback: Optional[Callable] = None):
        """
        初始化优化器
        
        Args:
            status_callback: 状态回调函数，用于显示优化进度
        """
        self.status_callback = status_callback
        
        # 优化参数边界（允许负值和>1的值以支持色彩空间转换）
        self.bounds = {
            'lower': [-100.0] * 9,  # 9个矩阵元素的下界
            'upper': [100.0] * 9    # 9个矩阵元素的上界
        }
        
        # 初始猜测：接近单位矩阵但允许一些变化
        self.initial_guess = np.array([
            1.0, 0.0, 0.0,  # 第一行: [1, 0, 0]
            0.0, 1.0, 0.0,  # 第二行: [0, 1, 0]
            0.0, 0.0, 1.0   # 第三行: [0, 0, 1]
        ])
        
        # 存储目标色彩空间信息（用于结果计算）
        self.target_colorspace = None
        
    def objective_function(self, params: np.ndarray, initial_rgb: np.ndarray) -> float:
        """
        目标函数：计算CCM变换后与目标RGB的误差
        
        目标：将initial_rgb变换到[[1,0,0], [0,1,0], [0,0,1]]
        
        Args:
            params: 9个优化参数（3x3矩阵展平）
            initial_rgb: 初始RGB值矩阵 (3x3)
            
        Returns:
            均方误差
        """
        try:
            # 将参数重构为3x3矩阵
            ccm = params.reshape(3, 3)
            
            # 目标RGB矩阵（单位矩阵）
            target_rgb = np.eye(3)
            
            # 计算变换后的RGB值
            transformed_rgb = ccm @ initial_rgb
            
            # 计算均方误差
            mse = np.mean((transformed_rgb - target_rgb) ** 2)
            
            return float(mse)
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"目标函数计算错误: {e}")
            return float('inf')
    
    def optimize(self, initial_rgb: np.ndarray, max_iter: int = 1000, 
                tolerance: float = 1e-8, target_colorspace: str = None) -> Dict:
        """
        执行优化
        
        Args:
            initial_rgb: 初始RGB值矩阵 (3x3)，每行对应一个通道的RGB值
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            target_colorspace: 目标色彩空间名称
            
        Returns:
            优化结果字典
        """
        # 存储目标色彩空间信息
        self.target_colorspace = target_colorspace
        
        if self.status_callback:
            self.status_callback("开始IDT优化...")
            self.status_callback(f"目标色彩空间: {target_colorspace}")
            self.status_callback(f"初始RGB矩阵:\n{initial_rgb}")
        
        # CMA-ES参数设置
        sigma0 = 0.3  # 初始步长
        opts = {
            'bounds': [self.bounds['lower'], self.bounds['upper']],
            'maxiter': int(max_iter),
            'ftarget': max(float(tolerance), 1e-10),
            'verb_disp': 0,  # 禁用CMA-ES内置显示
            'verbose': -1,   # 禁用详细输出
            'popsize': max(int(4 + 3 * np.log(9)), 20),  # 种群大小
            'tolfun': 1e-12,
            'tolx': 1e-12,
        }
        
        try:
            # 初始化CMA-ES
            es = cma.CMAEvolutionStrategy(self.initial_guess, sigma0, opts)
            
            best_mse = float('inf')
            best_params = None
            
            # 优化循环
            while not es.stop():
                # 获取候选解
                xs = es.ask()
                
                # 计算目标函数值
                fs = []
                for x in xs:
                    f_val = self.objective_function(x, initial_rgb)
                    fs.append(f_val)
                
                # 更新算法状态
                es.tell(xs, fs)
                
                # 记录最佳结果
                gen_best = float(np.min(fs))
                if gen_best < best_mse:
                    best_mse = gen_best
                    best_idx = np.argmin(fs)
                    best_params = xs[best_idx].copy()
                
                # 显示进度
                message = f"迭代 {es.countiter:3d}: MSE={gen_best:.8f} (最优={best_mse:.8f})"
                if self.status_callback:
                    self.status_callback(message)
                else:
                    print(message)
            
            # 获取最终结果
            res = es.result
            final_params = np.array(res.xbest, dtype=float)
            final_mse = float(res.fbest)
            nit = int(es.countiter)
            
            # 重构为3x3矩阵
            final_ccm = final_params.reshape(3, 3)
            
            # 验证结果
            transformed_rgb = final_ccm @ initial_rgb
            target_rgb = np.eye(3)
            verification_mse = np.mean((transformed_rgb - target_rgb) ** 2)
            
            if self.status_callback:
                self.status_callback("✓ 优化完成")
                self.status_callback(f"最终MSE: {final_mse:.8f}")
                self.status_callback(f"验证MSE: {verification_mse:.8f}")
                self.status_callback(f"迭代次数: {nit}")
                self.status_callback(f"最优CCM矩阵:\n{final_ccm}")
            
            return {
                'success': True,
                'ccm_matrix': final_ccm,
                'mse': final_mse,
                'verification_mse': verification_mse,
                'iterations': nit,
                'transformed_rgb': transformed_rgb,
                'target_rgb': target_rgb,
                'target_colorspace': self.target_colorspace,
                'convergence_info': {
                    'final_sigma': float(es.result.stds[0]) if hasattr(es.result, 'stds') else None,
                    'stop_reason': str(es.stop())
                }
            }
            
        except Exception as e:
            error_msg = f"优化失败: {e}"
            if self.status_callback:
                self.status_callback(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'ccm_matrix': None,
                'mse': float('inf'),
                'iterations': 0
            }
    
    def evaluate_ccm(self, ccm: np.ndarray, initial_rgb: np.ndarray) -> Dict:
        """
        评估给定CCM矩阵的性能
        
        Args:
            ccm: 3x3 CCM矩阵
            initial_rgb: 初始RGB值矩阵 (3x3)
            
        Returns:
            评估结果字典
        """
        try:
            # 计算变换后的RGB值
            transformed_rgb = ccm @ initial_rgb
            target_rgb = np.eye(3)
            
            # 计算各种误差指标
            mse = np.mean((transformed_rgb - target_rgb) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(transformed_rgb - target_rgb))
            max_error = np.max(np.abs(transformed_rgb - target_rgb))
            
            # 计算每个通道的误差
            channel_errors = []
            for i in range(3):
                channel_mse = np.mean((transformed_rgb[i, :] - target_rgb[i, :]) ** 2)
                channel_errors.append(float(channel_mse))
            
            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'max_error': float(max_error),
                'channel_errors': channel_errors,
                'transformed_rgb': transformed_rgb.tolist(),
                'target_rgb': target_rgb.tolist()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'mse': float('inf')
            }
    
    def suggest_initial_guess(self, initial_rgb: np.ndarray) -> np.ndarray:
        """
        基于初始RGB值建议更好的初始猜测
        
        Args:
            initial_rgb: 初始RGB值矩阵 (3x3)
            
        Returns:
            建议的初始猜测参数
        """
        try:
            # 尝试使用伪逆作为初始猜测
            target_rgb = np.eye(3)
            ccm_guess = target_rgb @ np.linalg.pinv(initial_rgb)
            
            # 限制在合理范围内
            ccm_guess = np.clip(ccm_guess, self.bounds['lower'][0], self.bounds['upper'][0])
            
            return ccm_guess.flatten()
            
        except Exception:
            # 如果计算失败，返回默认的单位矩阵
            return self.initial_guess.copy()
    
    def set_bounds(self, lower: float, upper: float):
        """
        设置优化边界
        
        Args:
            lower: 下边界
            upper: 上边界
        """
        self.bounds['lower'] = [lower] * 9
        self.bounds['upper'] = [upper] * 9