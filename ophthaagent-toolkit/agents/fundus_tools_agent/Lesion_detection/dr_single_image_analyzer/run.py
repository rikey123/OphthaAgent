#!/usr/bin/env python3
"""
DR分析器启动脚本
简化命令行调用
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dr_analyzer import main

if __name__ == "__main__":
    main()
