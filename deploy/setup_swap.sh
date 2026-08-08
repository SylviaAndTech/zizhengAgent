#!/usr/bin/env bash
# 在2核4G这类小内存云主机上加一块swap，防止偶尔的大文件上传/大PDF渲染让内存
# 瞬时冲高时被OOM killer直接杀掉进程——swap不解决内存紧张的根本问题（会变慢），
# 只是给这种偶发尖峰留一个安全垫，宁可慢几秒也不要进程被杀掉。
#
# 用法：部署到云主机后，用root权限跑一次
#   sudo bash deploy/setup_swap.sh
#
# 60G系统盘留4G出来做swap文件完全不影响够用（应用本身不往磁盘存文件）。
set -euo pipefail

SWAP_FILE=/swapfile
SWAP_SIZE_GB=4

if [ "$(id -u)" -ne 0 ]; then
  echo "请用 sudo 运行" >&2
  exit 1
fi

if swapon --show | grep -q "$SWAP_FILE"; then
  echo "swap文件已存在并已启用，跳过：$SWAP_FILE"
  exit 0
fi

if [ -f "$SWAP_FILE" ]; then
  echo "发现同名文件但未启用，先删除重建：$SWAP_FILE"
  rm -f "$SWAP_FILE"
fi

fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_FILE" || dd if=/dev/zero of="$SWAP_FILE" bs=1M count=$((SWAP_SIZE_GB * 1024))
chmod 600 "$SWAP_FILE"
mkswap "$SWAP_FILE"
swapon "$SWAP_FILE"

# 开机自动挂载
if ! grep -q "^$SWAP_FILE " /etc/fstab; then
  echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
fi

# 云主机上磁盘通常是SSD，swappiness调低一点，尽量少用swap、只在真正紧张时才用
sysctl -w vm.swappiness=10
if ! grep -q "^vm.swappiness" /etc/sysctl.conf 2>/dev/null; then
  echo "vm.swappiness=10" >> /etc/sysctl.conf
fi

echo "完成，当前swap状态："
swapon --show
free -h
