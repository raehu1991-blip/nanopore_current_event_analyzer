#!/usr/bin/env python3
"""
Current Event Analyzer - 主入口脚本
电流事件分析器的主启动脚本

使用方法:
    python run.py

作者: Current Event Analyzer Team
版本: 1.0.0
"""

import sys
import os

# 添加src目录到Python路径，确保可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    import main_analyzer
    
    if __name__ == '__main__':
        print("正在启动电流事件分析器...")
        print("Current Event Analyzer v1.0.0")
        print("=" * 50)
        main_analyzer.main()
        
except ImportError as e:
    print(f"错误：无法导入主分析器模块")
    print(f"详细信息：{e}")
    print("请确保所有依赖包已正确安装：")
    print("pip install -r requirements.txt")
    print("\n尝试直接运行主程序：")
    print("python src/main_analyzer.py")
    sys.exit(1)
    
except Exception as e:
    print(f"启动时发生错误：{e}")
    print("如果问题持续存在，请检查是否在图形环境中运行程序")
    sys.exit(1)
