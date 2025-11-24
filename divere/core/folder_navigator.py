"""
文件夹导航管理器
用于管理和导航同一文件夹下的图片文件
"""

import os
import re
from typing import List, Optional
from pathlib import Path

from PySide6.QtCore import QObject, Signal


class FolderNavigator(QObject):
    """文件夹导航管理器"""
    
    # 信号：文件发生变化时发射
    file_changed = Signal(str)  # 新文件的完整路径
    
    def __init__(self, image_manager=None):
        super().__init__()
        self.image_manager = image_manager
        self._current_folder: Optional[str] = None
        self._file_list: List[str] = []  # 文件名列表（不含路径）
        self._current_index: int = -1
    
    def _natural_sort_key(self, filename: str) -> tuple:
        """自然排序键，正确处理数字序列（如img1.jpg < img10.jpg）"""
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        
        return [convert(c) for c in re.split(r'(\d+)', filename)]
    
    def _scan_folder(self, folder_path: str) -> List[str]:
        """扫描文件夹，获取所有支持的图片文件"""
        if not self.image_manager:
            return []
        
        try:
            files = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and self.image_manager.is_supported_format(file_path):
                    files.append(filename)
            
            # 使用自然排序
            files.sort(key=self._natural_sort_key)
            return files
        
        except (OSError, PermissionError):
            return []
    
    def update_folder(self, current_file_path: str):
        """更新当前文件夹，扫描所有图片文件"""
        if not current_file_path or not os.path.isfile(current_file_path):
            self._reset()
            return
        
        folder_path = os.path.dirname(current_file_path)
        current_filename = os.path.basename(current_file_path)
        
        # 如果文件夹没有变化，只需要更新当前索引
        if folder_path == self._current_folder:
            try:
                self._current_index = self._file_list.index(current_filename)
            except ValueError:
                # 当前文件不在列表中，重新扫描
                self._scan_and_update(folder_path, current_filename)
        else:
            # 文件夹发生变化，重新扫描
            self._scan_and_update(folder_path, current_filename)
    
    def _scan_and_update(self, folder_path: str, current_filename: str):
        """扫描文件夹并更新状态"""
        self._current_folder = folder_path
        self._file_list = self._scan_folder(folder_path)
        
        # 查找当前文件的索引
        try:
            self._current_index = self._file_list.index(current_filename)
        except ValueError:
            self._current_index = -1
    
    def _reset(self):
        """重置状态"""
        self._current_folder = None
        self._file_list = []
        self._current_index = -1
    
    def get_file_list(self) -> List[str]:
        """获取当前文件夹的文件名列表"""
        return self._file_list.copy()
    
    def get_current_index(self) -> int:
        """获取当前文件的索引"""
        return self._current_index
    
    def get_current_filename(self) -> Optional[str]:
        """获取当前文件名"""
        if 0 <= self._current_index < len(self._file_list):
            return self._file_list[self._current_index]
        return None
    
    def navigate_to_index(self, index: int, force_reload: bool = False):
        """跳转到指定索引的文件

        Args:
            index: 文件索引
            force_reload: 强制重新加载，即使已经是当前文件
        """
        if not (0 <= index < len(self._file_list)):
            return

        if index == self._current_index and not force_reload:
            return  # 已经是当前文件且不强制重新加载

        self._current_index = index
        filename = self._file_list[index]
        file_path = os.path.join(self._current_folder, filename)

        # 发射信号通知文件变化
        self.file_changed.emit(file_path)
    
    def navigate_previous(self):
        """导航到上一张图片"""
        if self._current_index > 0:
            self.navigate_to_index(self._current_index - 1)
    
    def navigate_next(self):
        """导航到下一张图片"""
        if self._current_index < len(self._file_list) - 1:
            self.navigate_to_index(self._current_index + 1)
    
    def can_navigate_previous(self) -> bool:
        """是否可以导航到上一张"""
        return self._current_index > 0
    
    def can_navigate_next(self) -> bool:
        """是否可以导航到下一张"""
        return self._current_index < len(self._file_list) - 1
    
    def get_navigation_info(self) -> dict:
        """获取导航信息"""
        return {
            'current_index': self._current_index,
            'total_count': len(self._file_list),
            'current_filename': self.get_current_filename(),
            'can_previous': self.can_navigate_previous(),
            'can_next': self.can_navigate_next(),
            'file_list': self.get_file_list()
        }