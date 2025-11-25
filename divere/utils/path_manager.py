"""
路径管理模块
使用addpath来管理各种配置和资源路径
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Import debug logger
try:
    from .debug_logger import debug, info, warning, error, log_path_search, log_file_operation
except ImportError:
    # Fallback if debug logger is not available
    def debug(msg, module=None): pass
    def info(msg, module=None): pass
    def warning(msg, module=None): pass
    def error(msg, module=None): pass
    def log_path_search(desc, paths, found=None, module=None): pass
    def log_file_operation(op, path, success=True, err=None, module=None): pass

class PathManager:
    """路径管理器，使用addpath方式管理各种路径"""
    
    def __init__(self):
        self._paths: Dict[str, List[str]] = {
            "config": [],
            "defaults": [],
            "colorspace": [],
            "curves": [],
            "matrices": [],
            "assets": [],
            "models": [],
            "test_data": [],
            "i18n": []
        }
        self._initialized = False
        self._initialize_paths()
    
    def _initialize_paths(self):
        """初始化所有路径"""
        if self._initialized:
            return
            
        info("Initializing PathManager paths", "PathManager")
        
        # 获取项目根目录
        project_root = self._get_project_root()
        info(f"Using project root: {project_root}", "PathManager")
        
        # 根据环境使用不同的路径构建策略
        if self._is_pyinstaller_bundle():
            # PyInstaller打包环境（Windows和其他平台）
            self._initialize_pyinstaller_paths()
        elif self._is_macos_app_bundle():
            # macOS .app bundle
            self._initialize_app_bundle_paths(project_root)
        else:
            # 开发环境
            self._initialize_development_paths(project_root)
    
    def _initialize_development_paths(self, project_root: str):
        """初始化开发环境路径"""
        info("Using development environment paths", "PathManager")
        
        # 配置路径
        config_paths = [
            os.path.join(project_root, "config"),
            os.path.join(project_root, "config", "defaults"),
            os.path.join(project_root, "config", "colorchecker"),
            os.path.join(project_root, "config", "colorspace"),
            os.path.join(project_root, "config", "curves"),
            os.path.join(project_root, "config", "matrices")
        ]
        self._paths["config"].extend(config_paths)
        debug(f"Config paths: {config_paths}", "PathManager")
        
        # 默认预设路径
        defaults_paths = [
            os.path.join(project_root, "config", "defaults")
        ]
        self._paths["defaults"].extend(defaults_paths)
        debug(f"Defaults paths: {defaults_paths}", "PathManager")
        
        # 色彩空间路径
        self._paths["colorspace"].extend([
            os.path.join(project_root, "config", "colorspace"),
            os.path.join(project_root, "config", "colorspace", "legacy"),
            os.path.join(project_root, "config", "colorspace", "icc")
        ])
        
        # 曲线路径
        self._paths["curves"].extend([
            os.path.join(project_root, "config", "curves")
        ])
        
        # 矩阵路径
        self._paths["matrices"].extend([
            os.path.join(project_root, "config", "matrices")
        ])
        
        # 资源路径
        self._paths["assets"].extend([
            os.path.join(project_root, "divere", "assets")
        ])
        
        # 模型路径
        self._paths["models"].extend([
            os.path.join(project_root, "divere", "models")
        ])
        
        # 测试数据路径
        self._paths["test_data"].extend([
            os.path.join(project_root, "test_scans")
        ])

        # i18n 路径
        self._paths["i18n"].extend([
            os.path.join(project_root, "divere", "assets", "i18n")
        ])

        # 添加所有路径到Python路径
        self._add_all_paths()

        self._initialized = True
    
    def _initialize_pyinstaller_paths(self):
        """初始化PyInstaller打包环境路径（主要用于Windows）"""
        info("Using PyInstaller bundle paths", "PathManager")
        
        # 获取PyInstaller bundle目录
        bundle_dir = self._get_pyinstaller_root()
        
        # Windows上，资源文件通常在exe同目录
        # 在其他平台或使用--onefile时，可能在_MEIPASS临时目录
        if getattr(sys, 'frozen', False) and not hasattr(sys, '_MEIPASS'):
            # Windows --onedir模式：文件在exe同目录
            base_dir = os.path.dirname(sys.executable)
            info(f"Using Windows onedir mode, base directory: {base_dir}", "PathManager")
        else:
            # --onefile模式或其他平台：使用bundle_dir
            base_dir = bundle_dir
            info(f"Using bundle directory: {base_dir}", "PathManager")
        
        # 配置路径
        config_paths = [
            os.path.join(base_dir, "config"),
            os.path.join(base_dir, "config", "defaults"),
            os.path.join(base_dir, "config", "colorchecker"),
            os.path.join(base_dir, "config", "colorspace"),
            os.path.join(base_dir, "config", "curves"),
            os.path.join(base_dir, "config", "matrices"),
            # 兼容性路径
            os.path.join(base_dir, "_internal", "config"),
            os.path.join(base_dir, "_internal", "config", "defaults"),
            os.path.join(base_dir, "_internal", "config", "colorchecker"),
            os.path.join(base_dir, "_internal", "config", "colorspace"),
            os.path.join(base_dir, "_internal", "config", "curves"),
            os.path.join(base_dir, "_internal", "config", "matrices")
        ]
        self._paths["config"].extend(config_paths)
        debug(f"PyInstaller config paths: {config_paths}", "PathManager")
        
        # 默认预设路径
        defaults_paths = [
            os.path.join(base_dir, "config", "defaults"),
            os.path.join(base_dir, "_internal", "config", "defaults")
        ]
        self._paths["defaults"].extend(defaults_paths)
        debug(f"PyInstaller defaults paths: {defaults_paths}", "PathManager")
        
        # 色彩空间路径
        self._paths["colorspace"].extend([
            os.path.join(base_dir, "config", "colorspace"),
            os.path.join(base_dir, "config", "colorspace", "legacy"),
            os.path.join(base_dir, "config", "colorspace", "icc"),
            os.path.join(base_dir, "_internal", "config", "colorspace"),
            os.path.join(base_dir, "_internal", "config", "colorspace", "legacy"),
            os.path.join(base_dir, "_internal", "config", "colorspace", "icc")
        ])
        
        # 曲线路径
        self._paths["curves"].extend([
            os.path.join(base_dir, "config", "curves"),
            os.path.join(base_dir, "_internal", "config", "curves")
        ])
        
        # 矩阵路径
        self._paths["matrices"].extend([
            os.path.join(base_dir, "config", "matrices"),
            os.path.join(base_dir, "_internal", "config", "matrices")
        ])
        
        # 资源路径
        self._paths["assets"].extend([
            os.path.join(base_dir, "assets"),
            os.path.join(base_dir, "_internal", "assets")
        ])
        
        # 模型路径
        self._paths["models"].extend([
            os.path.join(base_dir, "models"),
            os.path.join(base_dir, "_internal", "models")
        ])
        
        # 测试数据路径（打包版本通常不包含）
        self._paths["test_data"].extend([
            os.path.join(base_dir, "test_scans"),
            os.path.join(base_dir, "_internal", "test_scans")
        ])

        # i18n 路径
        self._paths["i18n"].extend([
            os.path.join(base_dir, "assets", "i18n"),
            os.path.join(base_dir, "_internal", "assets", "i18n")
        ])

        # 添加所有路径到Python路径
        self._add_all_paths()

        self._initialized = True
        
        # 调试输出：显示实际存在的路径
        info("Checking which PyInstaller paths actually exist:", "PathManager")
        for category, paths in self._paths.items():
            existing = [p for p in paths if os.path.exists(p)]
            if existing:
                info(f"  {category}: {len(existing)} paths exist", "PathManager")
                for p in existing[:2]:  # 只显示前两个
                    debug(f"    - {p}", "PathManager")
    
    def _initialize_app_bundle_paths(self, executable_dir: str):
        """初始化 macOS app bundle 路径"""
        info("Using macOS app bundle paths", "PathManager")
        
        # 在 app bundle 中，配置文件可能在多个位置
        # 优先级：1. 可执行文件旁 2. Contents/Resources 3. 其他可能位置
        bundle_contents = os.path.dirname(executable_dir)  # Contents/
        bundle_root = os.path.dirname(bundle_contents)     # DiVERE.app/
        
        debug(f"Bundle contents dir: {bundle_contents}", "PathManager")
        debug(f"Bundle root dir: {bundle_root}", "PathManager")
        
        # 候选配置路径 (按优先级排序)
        config_candidates = [
            # 1. 直接在可执行文件旁边 (Contents/MacOS/config/)
            os.path.join(executable_dir, "config"),
            os.path.join(executable_dir, "config", "defaults"),
            os.path.join(executable_dir, "config", "colorchecker"),
            os.path.join(executable_dir, "config", "colorspace"),
            os.path.join(executable_dir, "config", "curves"),
            os.path.join(executable_dir, "config", "matrices"),
            
            # 2. Contents/Resources/ 目录
            os.path.join(bundle_contents, "Resources", "config"),
            os.path.join(bundle_contents, "Resources", "config", "defaults"),
            os.path.join(bundle_contents, "Resources", "config", "colorchecker"),
            os.path.join(bundle_contents, "Resources", "config", "colorspace"),
            os.path.join(bundle_contents, "Resources", "config", "curves"),
            os.path.join(bundle_contents, "Resources", "config", "matrices"),
            
            # 3. 保持兼容开发环境结构 (Contents/MacOS/divere/config/)
            os.path.join(executable_dir, "divere", "config"),
            os.path.join(executable_dir, "divere", "config", "defaults"),
            os.path.join(executable_dir, "divere", "config", "colorchecker"),
            os.path.join(executable_dir, "divere", "config", "colorspace"),
            os.path.join(executable_dir, "divere", "config", "curves"),
            os.path.join(executable_dir, "divere", "config", "matrices")
        ]
        
        self._paths["config"].extend(config_candidates)
        debug(f"App bundle config paths: {config_candidates}", "PathManager")
        
        # 默认预设路径
        defaults_candidates = [
            os.path.join(executable_dir, "config", "defaults"),
            os.path.join(bundle_contents, "Resources", "config", "defaults"),
            os.path.join(executable_dir, "divere", "config", "defaults")
        ]
        self._paths["defaults"].extend(defaults_candidates)
        debug(f"App bundle defaults paths: {defaults_candidates}", "PathManager")
        
        # 色彩空间路径
        colorspace_candidates = [
            os.path.join(executable_dir, "config", "colorspace"),
            os.path.join(executable_dir, "config", "colorspace", "legacy"),
            os.path.join(executable_dir, "config", "colorspace", "icc"),
            os.path.join(bundle_contents, "Resources", "config", "colorspace"),
            os.path.join(bundle_contents, "Resources", "config", "colorspace", "legacy"),
            os.path.join(bundle_contents, "Resources", "config", "colorspace", "icc"),
            os.path.join(executable_dir, "divere", "config", "colorspace"),
            os.path.join(executable_dir, "divere", "config", "colorspace", "legacy"),
            os.path.join(executable_dir, "divere", "config", "colorspace", "icc")
        ]
        self._paths["colorspace"].extend(colorspace_candidates)
        
        # 曲线路径
        curves_candidates = [
            os.path.join(executable_dir, "config", "curves"),
            os.path.join(bundle_contents, "Resources", "config", "curves"),
            os.path.join(executable_dir, "divere", "config", "curves")
        ]
        self._paths["curves"].extend(curves_candidates)
        
        # 矩阵路径
        matrices_candidates = [
            os.path.join(executable_dir, "config", "matrices"),
            os.path.join(bundle_contents, "Resources", "config", "matrices"),
            os.path.join(executable_dir, "divere", "config", "matrices")
        ]
        self._paths["matrices"].extend(matrices_candidates)
        
        # 资源路径
        assets_candidates = [
            os.path.join(executable_dir, "assets"),
            os.path.join(bundle_contents, "Resources", "assets"),
            os.path.join(executable_dir, "divere", "assets")
        ]
        self._paths["assets"].extend(assets_candidates)
        
        # 模型路径
        models_candidates = [
            os.path.join(executable_dir, "models"),
            os.path.join(bundle_contents, "Resources", "models"),
            os.path.join(executable_dir, "divere", "models")
        ]
        self._paths["models"].extend(models_candidates)
        
        # 测试数据路径 (在 app bundle 中通常不存在)
        test_data_candidates = [
            os.path.join(executable_dir, "test_scans"),
            os.path.join(bundle_contents, "Resources", "test_scans")
        ]
        self._paths["test_data"].extend(test_data_candidates)

        # i18n 路径
        i18n_candidates = [
            os.path.join(executable_dir, "assets", "i18n"),
            os.path.join(bundle_contents, "Resources", "assets", "i18n"),
            os.path.join(executable_dir, "divere", "assets", "i18n")
        ]
        self._paths["i18n"].extend(i18n_candidates)

        # 添加所有路径到Python路径
        self._add_all_paths()

        self._initialized = True
    
    def _get_project_root(self) -> str:
        """获取项目根目录"""
        info("Starting project root detection", "PathManager")
        
        # 检查是否在 macOS app bundle 中运行
        if self._is_macos_app_bundle():
            return self._get_app_bundle_root()
        
        # 标准开发环境路径检测
        current_file = os.path.abspath(__file__)
        debug(f"Current file: {current_file}", "PathManager")
        
        # 从当前文件向上查找，直到找到包含pyproject.toml的目录
        current_dir = os.path.dirname(current_file)
        searched_dirs = []
        
        while current_dir != os.path.dirname(current_dir):
            searched_dirs.append(current_dir)
            pyproject_path = os.path.join(current_dir, "pyproject.toml")
            debug(f"Checking for pyproject.toml at: {pyproject_path}", "PathManager")
            
            if os.path.exists(pyproject_path):
                info(f"Found pyproject.toml at: {current_dir}", "PathManager")
                log_path_search("pyproject.toml search", searched_dirs, current_dir, "PathManager")
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # 如果找不到，使用当前工作目录
        fallback = os.getcwd()
        warning(f"pyproject.toml not found, using fallback: {fallback}", "PathManager")
        log_path_search("pyproject.toml search (not found)", searched_dirs, fallback, "PathManager")
        
        return fallback
    
    def _is_pyinstaller_bundle(self) -> bool:
        """检测是否在PyInstaller打包的环境中运行"""
        try:
            # 检查PyInstaller特有的属性
            is_frozen = getattr(sys, 'frozen', False)
            has_meipass = hasattr(sys, '_MEIPASS')
            
            if is_frozen or has_meipass:
                info(f"Detected PyInstaller bundle (frozen={is_frozen}, _MEIPASS={has_meipass})", "PathManager")
                if has_meipass:
                    debug(f"PyInstaller temp path: {sys._MEIPASS}", "PathManager")
                if is_frozen:
                    debug(f"Executable path: {sys.executable}", "PathManager")
                return True
            
            return False
        except Exception as e:
            debug(f"PyInstaller detection failed: {e}", "PathManager")
            return False
    
    def _is_macos_app_bundle(self) -> bool:
        """检测是否在 macOS app bundle 中运行"""
        try:
            # 检查 sys.argv[0] 是否包含 .app/Contents/MacOS/
            argv0 = os.path.abspath(sys.argv[0])
            is_bundle = '.app/Contents/MacOS' in argv0
            
            if is_bundle:
                info(f"Detected macOS app bundle: {argv0}", "PathManager")
            else:
                debug(f"Not in app bundle, argv[0]: {argv0}", "PathManager")
                
            return is_bundle
        except Exception as e:
            debug(f"App bundle detection failed: {e}", "PathManager")
            return False
    
    def _get_pyinstaller_root(self) -> str:
        """获取PyInstaller bundle的根目录"""
        info("Getting PyInstaller bundle root", "PathManager")
        
        try:
            if hasattr(sys, '_MEIPASS') and sys._MEIPASS:
                # PyInstaller临时解压目录
                bundle_dir = sys._MEIPASS
                info(f"Using PyInstaller _MEIPASS: {bundle_dir}", "PathManager")
                return bundle_dir
            elif getattr(sys, 'frozen', False):
                # exe所在目录
                bundle_dir = os.path.dirname(sys.executable)
                info(f"Using frozen executable directory: {bundle_dir}", "PathManager")
                return bundle_dir
            else:
                # 回退到当前工作目录
                fallback = os.getcwd()
                warning(f"PyInstaller attributes not found, using fallback: {fallback}", "PathManager")
                return fallback
                
        except Exception as e:
            error(f"Failed to get PyInstaller root: {e}", "PathManager")
            fallback = os.getcwd()
            warning(f"Using fallback for PyInstaller: {fallback}", "PathManager")
            return fallback
    
    def _get_app_bundle_root(self) -> str:
        """获取 macOS app bundle 的根目录"""
        info("Using macOS app bundle path resolution", "PathManager")
        
        try:
            # 获取可执行文件路径
            executable_path = os.path.abspath(sys.argv[0])
            debug(f"Executable path: {executable_path}", "PathManager")
            
            # 获取可执行文件所在目录 (Contents/MacOS/)
            executable_dir = os.path.dirname(executable_path)
            info(f"Executable directory: {executable_dir}", "PathManager")
            
            return executable_dir
            
        except Exception as e:
            error(f"Failed to get app bundle root: {e}", "PathManager")
            # 回退到当前工作目录
            fallback = os.getcwd()
            warning(f"Using fallback for app bundle: {fallback}", "PathManager")
            return fallback
    
    def _add_all_paths(self):
        """将所有路径添加到Python路径"""
        for path_list in self._paths.values():
            for path in path_list:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
    
    def add_path(self, category: str, path: str):
        """添加新路径到指定类别"""
        if category not in self._paths:
            self._paths[category] = []
        
        if path not in self._paths[category]:
            self._paths[category].append(path)
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
    
    def remove_path(self, category: str, path: str):
        """从指定类别移除路径"""
        if category in self._paths and path in self._paths[category]:
            self._paths[category].remove(path)
            if path in sys.path:
                sys.path.remove(path)
    
    def get_paths(self, category: str) -> List[str]:
        """获取指定类别的所有路径"""
        return self._paths.get(category, [])
    
    def find_file(self, filename: str, category: str = None) -> Optional[str]:
        """在指定类别中查找文件"""
        if category:
            # 在指定类别中查找
            for path in self._paths.get(category, []):
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    return file_path
        else:
            # 在所有类别中查找
            for path_list in self._paths.values():
                for path in path_list:
                    file_path = os.path.join(path, filename)
                    if os.path.exists(file_path):
                        return file_path
        
        return None
    
    def find_files_by_pattern(self, pattern: str, category: str = None) -> List[str]:
        """根据模式查找文件"""
        import glob
        found_files = []
        
        if category:
            # 在指定类别中查找
            for path in self._paths.get(category, []):
                search_pattern = os.path.join(path, pattern)
                found_files.extend(glob.glob(search_pattern))
        else:
            # 在所有类别中查找
            for path_list in self._paths.values():
                for path in path_list:
                    search_pattern = os.path.join(path, pattern)
                    found_files.extend(glob.glob(search_pattern))
        
        return found_files
    
    def resolve_path(self, relative_path: str, category: str = None) -> Optional[str]:
        """解析相对路径为绝对路径"""
        info(f"Resolving path: '{relative_path}' in category: {category}", "PathManager")
        
        candidate_paths = []
        
        if category:
            # 在指定类别中查找
            for path in self._paths.get(category, []):
                full_path = os.path.join(path, relative_path)
                candidate_paths.append(full_path)
                if os.path.exists(full_path):
                    log_path_search(f"resolve_path('{relative_path}', '{category}')", candidate_paths, full_path, "PathManager")
                    return full_path
        else:
            # 在所有类别中查找
            for path_list in self._paths.values():
                for path in path_list:
                    full_path = os.path.join(path, relative_path)
                    candidate_paths.append(full_path)
                    if os.path.exists(full_path):
                        log_path_search(f"resolve_path('{relative_path}', all categories)", candidate_paths, full_path, "PathManager")
                        return full_path
        
        # Log failed search
        log_path_search(f"resolve_path('{relative_path}', {category}) - FAILED", candidate_paths, None, "PathManager")
        return None
    
    def get_default_preset_path(self, preset_name: str) -> Optional[str]:
        """获取默认预设文件的完整路径"""
        info(f"Looking for default preset: '{preset_name}'", "PathManager")
        result = self.find_file(preset_name, "defaults")
        if result:
            info(f"Found default preset: {result}", "PathManager")
        else:
            warning(f"Default preset not found: '{preset_name}'", "PathManager")
        return result
    
    def get_config_path(self, config_name: str) -> Optional[str]:
        """获取配置文件的完整路径"""
        return self.find_file(config_name, "config")
    
    def list_default_presets(self) -> List[str]:
        """列出所有可用的默认预设文件"""
        preset_files = []
        for path in self._paths["defaults"]:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith('.json'):
                        preset_files.append(file)
        return sorted(preset_files)
    
    def get_path_info(self) -> Dict[str, Any]:
        """获取路径信息"""
        info = {}
        for category, paths in self._paths.items():
            info[category] = {
                "paths": paths,
                "exists": [os.path.exists(p) for p in paths],
                "files": []
            }
            
            # 统计每个路径下的文件数量
            for path in paths:
                if os.path.exists(path):
                    try:
                        files = os.listdir(path)
                        info[category]["files"].append({
                            "path": path,
                            "count": len(files),
                            "sample_files": files[:5]  # 前5个文件作为示例
                        })
                    except Exception:
                        info[category]["files"].append({
                            "path": path,
                            "count": 0,
                            "sample_files": []
                        })
        
        return info


# 全局路径管理器实例
path_manager = PathManager()

# 便捷函数
def add_path(category: str, path: str):
    """添加路径的便捷函数"""
    path_manager.add_path(category, path)

def find_file(filename: str, category: str = None) -> Optional[str]:
    """查找文件的便捷函数"""
    return path_manager.find_file(filename, category)

def resolve_path(relative_path: str, category: str = None) -> Optional[str]:
    """解析路径的便捷函数"""
    return path_manager.resolve_path(relative_path, category)

def get_default_preset_path(preset_name: str) -> Optional[str]:
    """获取默认预设路径的便捷函数"""
    return path_manager.get_default_preset_path(preset_name)

def list_default_presets() -> List[str]:
    """列出默认预设的便捷函数"""
    return path_manager.list_default_presets()
