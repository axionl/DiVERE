#!/bin/bash
# build_macos.sh — 本地 macOS PyInstaller 打包脚本（修复 backports 依赖问题）

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DiVERE macOS 本地打包脚本 ===${NC}"

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version)
echo -e "${YELLOW}使用 Python 版本: $PYTHON_VERSION${NC}"

# 检查是否安装了 PyInstaller
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo -e "${RED}错误: 未安装 PyInstaller${NC}"
    echo "请运行: pip install pyinstaller pyinstaller-hooks-contrib"
    exit 1
fi

# 清理旧的构建文件
echo -e "${YELLOW}清理旧的构建文件...${NC}"
rm -rf build dist *.spec

# 准备图标
ICON_PNG="divere/assets/icon.png"
if [ -f "$ICON_PNG" ]; then
    echo -e "${YELLOW}生成应用图标...${NC}"
    mkdir -p icons/icon.iconset
    sips -z 16 16   "$ICON_PNG" --out icons/icon.iconset/icon_16x16.png
    sips -z 32 32   "$ICON_PNG" --out icons/icon.iconset/icon_16x16@2x.png
    sips -z 32 32   "$ICON_PNG" --out icons/icon.iconset/icon_32x32.png
    sips -z 64 64   "$ICON_PNG" --out icons/icon.iconset/icon_32x32@2x.png
    sips -z 128 128 "$ICON_PNG" --out icons/icon.iconset/icon_128x128.png
    sips -z 256 256 "$ICON_PNG" --out icons/icon.iconset/icon_128x128@2x.png
    sips -z 256 256 "$ICON_PNG" --out icons/icon.iconset/icon_256x256.png
    sips -z 512 512 "$ICON_PNG" --out icons/icon.iconset/icon_256x256@2x.png
    sips -z 512 512 "$ICON_PNG" --out icons/icon.iconset/icon_512x512.png
    sips -z 1024 1024 "$ICON_PNG" --out icons/icon.iconset/icon_512x512@2x.png
    iconutil -c icns -o icons/app.icns icons/icon.iconset
    ICON_ARG="--icon icons/app.icns"
else
    echo -e "${YELLOW}警告: 未找到图标文件 $ICON_PNG${NC}"
    ICON_ARG=""
fi

# 使用 PyInstaller 打包
echo -e "${GREEN}开始 PyInstaller 打包...${NC}"
python3 -m PyInstaller \
  -y --clean --noconfirm \
  --windowed \
  --name "DiVERE" \
  ${ICON_ARG} \
  --add-data "config:config" \
  --add-data "divere/assets:assets" \
  --add-data "divere/models:models" \
  --collect-all onnxruntime \
  --collect-all pyopencl \
  --collect-all tifffile \
  --collect-all imagecodecs \
  --copy-metadata imageio \
  --copy-metadata colour-science \
  --copy-metadata scipy \
  --hidden-import "scipy.interpolate" \
  --hidden-import "scipy.optimize" \
  --hidden-import "scipy.ndimage" \
  --hidden-import "tifffile" \
  --hidden-import "imagecodecs" \
  --hidden-import "cma" \
  --exclude-module "pkg_resources" \
  --exclude-module "tkinter" \
  --exclude-module "matplotlib" \
  divere/__main__.py

echo -e "${GREEN}PyInstaller 构建完成${NC}"

# 查找生成的 .app 包
app_bundle=$(find dist -type d -name "*.app" | head -1)
if [ -z "$app_bundle" ]; then
    echo -e "${RED}错误: 未找到生成的 .app 包${NC}"
    exit 1
fi

echo -e "${YELLOW}重新定位资源文件到 Contents/MacOS...${NC}"
resources_dir="$app_bundle/Contents/Resources"
macos_dir="$app_bundle/Contents/MacOS"

# 移动资源文件到 MacOS 目录（运行时兼容性）
if [ -d "$resources_dir/config" ]; then
    rm -rf "$macos_dir/config"
    cp -a "$resources_dir/config" "$macos_dir/config"
fi
if [ -d "$resources_dir/models" ]; then
    rm -rf "$macos_dir/models"
    cp -a "$resources_dir/models" "$macos_dir/models"
fi
if [ -d "$resources_dir/assets" ]; then
    rm -rf "$macos_dir/assets"
    cp -a "$resources_dir/assets" "$macos_dir/assets"
fi

# Ad-hoc 签名
echo -e "${YELLOW}对应用包进行 ad-hoc 签名...${NC}"
codesign --force --deep -s - "$app_bundle" || true

# 验证目录结构
echo -e "${YELLOW}验证打包内容...${NC}"
required_dirs=(
  "Contents/MacOS/config"
  "Contents/MacOS/config/colorspace"
  "Contents/MacOS/config/curves"
  "Contents/MacOS/config/matrices"
  "Contents/MacOS/models"
  "Contents/MacOS/assets"
)

missing_dirs=0
for dir in "${required_dirs[@]}"; do
  if [ ! -d "$app_bundle/$dir" ]; then
    echo -e "${RED}缺少目录: $dir${NC}"
    missing_dirs=1
  else
    echo -e "${GREEN}✓ 目录存在: $dir${NC}"
  fi
done

if [ $missing_dirs -ne 0 ]; then
  echo -e "${RED}错误: 缺少必需的目录${NC}"
  exit 1
fi

echo -e "${GREEN}=== 打包完成 ===${NC}"
echo -e "应用包位置: ${YELLOW}$app_bundle${NC}"
echo ""
echo -e "${GREEN}测试应用:${NC}"
echo -e "  open \"$app_bundle\""
echo ""
echo -e "${GREEN}或从终端运行查看错误信息:${NC}"
echo -e "  \"$app_bundle/Contents/MacOS/DiVERE\""
