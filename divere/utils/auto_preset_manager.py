
"""
自动预设管理器
负责根据图片位置自动加载和保存调色预设。
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any, List

from divere.core.data_types import Preset, PresetBundle


class AutoPresetManager:
    """
    管理与图像文件在同一目录下的自动预设文件。
    预设文件名为 'divere_presets.json'。
    """
    PRESET_FILENAME = "divere_presets.json"

    def __init__(self):
        # 兼容：同时支持单 Preset 与 Bundle
        self._presets: Dict[str, Preset] = {}
        self._bundles: Dict[str, PresetBundle] = {}
        self._extra_data: Dict[str, dict] = {}  # 保存非预设数据（如folder_default）
        self._preset_file_path: Optional[Path] = None

    def _load_presets_from_file(self) -> None:
        """从文件加载预设到缓存"""
        self._presets = {}
        self._bundles = {}
        self._extra_data = {}  # 清空额外数据
        if self._preset_file_path and self._preset_file_path.exists():
            try:
                with open(self._preset_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for filename, payload in data.items():
                        if not isinstance(payload, dict):
                            continue
                        payload_type = payload.get("type")
                        if payload_type == "contactsheet":
                            # v3 contactsheet
                            self._bundles[filename] = PresetBundle.from_dict(payload)
                        elif payload_type == "single" or payload_type is None:
                            # 检查是否为特殊键（如folder_default）
                            if filename == "folder_default":
                                self._extra_data[filename] = payload
                            else:
                                # v3 single（或未标明但结构为 single）
                                self._presets[filename] = Preset.from_dict(payload)
                        else:
                            # 将未知类型保存为额外数据，而不是忽略
                            self._extra_data[filename] = payload
            except (IOError, json.JSONDecodeError):
                # 如果文件损坏或无法读取，则视为空文件
                pass

    def _save_presets_to_file(self) -> None:
        """将缓存中的预设保存到文件"""
        if self._preset_file_path:
            try:
                self._preset_file_path.parent.mkdir(parents=True, exist_ok=True)
                # 先保存额外数据（如folder_default）
                data_to_save: Dict[str, dict] = {}
                for key, value in self._extra_data.items():
                    data_to_save[key] = value
                
                # 然后保存 bundle 和 preset
                for filename, bundle in self._bundles.items():
                    data_to_save[filename] = bundle.to_dict()
                for filename, preset in self._presets.items():
                    # 若同名 bundle 已存在，则跳过旧 preset
                    if filename not in data_to_save:
                        data_to_save[filename] = preset.to_dict()
                        
                with open(self._preset_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            except IOError:
                # 处理写入错误
                pass

    def set_active_directory(self, directory: str) -> None:
        """
        设置当前活动目录，并加载该目录下的预设文件。
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return

        new_preset_file = dir_path / self.PRESET_FILENAME
        if self._preset_file_path != new_preset_file:
            self._preset_file_path = new_preset_file
            self._load_presets_from_file()

    def get_preset_for_image(self, image_path: str) -> Optional[Preset]:
        """
        获取指定图像文件的预设。
        """
        image_filename = Path(image_path).name
        # 若存在 bundle，返回其 contactsheet 作为默认入口（保持旧接口语义）
        if image_filename in self._bundles:
            return self._bundles[image_filename].contactsheet
        return self._presets.get(image_filename)

    def save_preset_for_image(self, image_path: str, preset: Preset) -> None:
        """
        为指定图像文件保存或更新预设。
        """
        image_filename = Path(image_path).name
        # 迁移期：保存single preset时，若存在同名bundle，移除bundle以避免歧义
        if image_filename in self._bundles:
            del self._bundles[image_filename]
        self._presets[image_filename] = preset
        self._save_presets_to_file()

    def get_current_preset_file_path(self) -> Optional[Path]:
        """返回当前预设文件的路径"""
        return self._preset_file_path

    # === 新增：Bundle 接口 ===
    def get_bundle_for_image(self, image_path: str) -> Optional[PresetBundle]:
        image_filename = Path(image_path).name
        return self._bundles.get(image_filename)

    def save_bundle_for_image(self, image_path: str, bundle: PresetBundle) -> None:
        image_filename = Path(image_path).name
        self._bundles[image_filename] = bundle
        # 迁移期：若存在同名旧 preset，移除以避免歧义
        if image_filename in self._presets:
            del self._presets[image_filename]
        self._save_presets_to_file()

    # === 新增：folder_default 接口 ===
    def save_folder_default(self, idt_data: Dict[str, Any], cc_params_data: Dict[str, Any]) -> None:
        """
        保存folder_default设置到当前预设文件。
        """
        self._extra_data['folder_default'] = {
            'idt': idt_data,
            'cc_params': cc_params_data
        }
        self._save_presets_to_file()

    def load_folder_default(self) -> Optional[Dict[str, Any]]:
        """
        从当前预设文件加载folder_default设置。
        
        Returns:
            包含'idt'和'cc_params'的字典，如果不存在则返回None。
        """
        return self._extra_data.get('folder_default')

    def has_folder_default(self) -> bool:
        """
        检查当前预设文件是否存在folder_default设置。
        """
        return 'folder_default' in self._extra_data

    # === 新增：获取所有预设的接口 ===
    def get_all_presets(self) -> Dict[str, Preset]:
        """获取所有单预设的副本"""
        return self._presets.copy()

    def get_all_bundles(self) -> Dict[str, PresetBundle]:
        """获取所有预设包的副本"""
        return self._bundles.copy()

    def get_preset_filenames(self) -> List[str]:
        """获取所有预设文件名列表（包括单预设和预设包）"""
        filenames = list(self._presets.keys()) + list(self._bundles.keys())
        return sorted(filenames)
