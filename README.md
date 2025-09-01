# Current Event Analyzer - 电流事件分析器

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/your-username/current-event-analyzer)

一个基于PyQt5的电流事件分析工具，支持TDMS和NPZ文件格式，提供交互式数据分析和多文件数据可视化功能。

![Main Interface](docs/images/main_interface.png)

## ✨ 主要功能

### 🔬 单事件分析模块
- **多格式支持**：TDMS文件（250kHz采样）和NPZ文件
- **交互式可视化**：可缩放、平移的时间-电流图表
- **智能峰值检测**：基于scipy.find_peaks的高精度检测算法
- **参数可调**：阈值、prominence、height等参数实时调节
- **选择性导出**：峰值审核功能，支持选择性CSV导出
- **视觉定制**：可自定义背景颜色和显示样式

### 📊 多文件可视化模块
- **批量导入**：支持多个CSV文件同时导入和分析
- **多种图表**：Box Plot、Histogram、Scatter Plot
- **跨文件对比**：不同数据源的统计对比分析
- **交互式控制**：实时图表更新和清除功能
- **智能列选择**：自动识别数值列，支持混合选择

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-username/current-event-analyzer.git
cd current-event-analyzer

# 安装依赖包
pip install -r requirements.txt
```

### 运行程序

```bash
# 方法1：使用入口脚本
python run.py

# 方法2：直接运行主程序
python src/main_analyzer.py
```

### 运行示例

```bash
# 峰值检测算法示例
python examples/peak_detection_example.py

# 运行测试
python tests/test_basic.py
```

## 📁 项目结构

```
current-event-analyzer/
├── 🗂️ src/                     # 源代码目录
│   └── main_analyzer.py       # 主分析器程序
├── 📚 docs/                    # 文档目录  
│   ├── README.md              # 详细说明文档
│   └── requirements.md        # 需求规格文档
├── 📊 data/                    # 数据目录
│   └── test_data.csv          # 示例测试数据
├── 💡 examples/                # 示例目录
│   └── peak_detection_example.py # 峰值检测示例
├── 🧪 tests/                   # 测试目录
│   └── test_basic.py          # 基础功能测试
├── 📋 requirements.txt         # Python依赖包
├── 🚀 run.py                   # 主入口脚本
├── ⚙️ setup.py                 # 安装配置
├── 📖 README.md                # 本文件
├── 📄 LICENSE                  # MIT许可证
└── 🚫 .gitignore              # Git忽略文件
```

## 🔧 依赖要求

### 必需依赖
- **Python** 3.7+
- **PyQt5** 5.15.0+ - GUI框架
- **numpy** 1.21.0+ - 数值计算
- **pandas** 1.3.0+ - 数据处理
- **scipy** 1.7.0+ - 科学计算和峰值检测
- **pyqtgraph** 0.12.0+ - 高性能数据可视化
- **npTDMS** 1.3.0+ - TDMS文件支持

### 可选依赖
- **matplotlib** 3.4.0+ - 额外的绘图功能
- **seaborn** 0.11.0+ - 统计图表美化
- **pytest** 6.0.0+ - 测试框架

## 📖 使用指南

### 单事件分析

1. **加载数据**：点击"Load TDMS/NPZ File Folder"选择数据文件夹
2. **设置参数**：调整阈值(Threshold)、显著性(Prominence)、高度(Height)
3. **执行分析**：点击"Analyze"按钮进行峰值检测
4. **审核结果**：在峰值列表中选择要导出的峰值
5. **导出数据**：点击"Export Selected to CSV"保存结果

### 多文件可视化

1. **导入文件**：点击"导入CSV文件"批量加载数据
2. **选择列**：点击文件查看可用列，双击添加到分析列表
3. **选择图表**：从Box Plot、Histogram、Scatter Plot中选择
4. **生成图表**：点击"Update Plot"创建可视化
5. **管理数据**：使用"Clear Plot"或"Clear Data"重置

## 🔬 算法原理

### 峰值检测算法
- 使用 `scipy.signal.find_peaks` 进行峰值识别
- 支持prominence（显著性）和height（高度）双重筛选
- 自动计算峰值起点和相对参数
- 输出完整的峰值特征数据

### 数据处理流程
1. **数据加载**：TDMS文件按250kHz采样率处理
2. **阈值筛选**：提取低于设定阈值的数据段
3. **峰值检测**：在筛选数据中识别显著峰值
4. **特征计算**：计算振幅、时间、相对参数等
5. **结果输出**：标准化CSV格式导出

## 📊 输出数据格式

CSV文件包含以下字段：

| 字段名 | 说明 | 单位 |
|--------|------|------|
| file_name | 源文件名 | - |
| peak_number | 峰值序号 | - |
| total_time | 事件总时间 | 秒 |
| peak_t | 峰值时间 | 秒 |
| peak_i | 峰值电流 | 安培 |
| peak_start_t | 起点时间 | 秒 |
| peak_start_i | 起点电流 | 安培 |
| peak_amplitude | 峰值振幅 | 安培 |
| peak_rel_t | 相对时间比例 | - |
| peak_rel_i | 相对电流比例 | - |

## 🛠️ 开发环境

### 推荐IDE
- **Visual Studio Code** + Python扩展
- **PyCharm** Professional/Community
- **Jupyter Lab** (用于数据分析)

### 开发安装
```bash
# 开发模式安装
pip install -e .

# 安装开发依赖
pip install pytest flake8 black

# 运行测试
pytest tests/

# 代码格式化
black src/ tests/ examples/
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. **Fork** 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 **Pull Request**

### 代码规范
- 遵循 PEP 8 Python代码规范
- 添加适当的文档字符串
- 为新功能编写测试用例
- 保持代码简洁可读

## 📝 更新日志

### v1.0.0 (2024-09-01)
- ✨ 初始版本发布
- 🔬 单事件分析功能
- 📊 多文件可视化功能
- 🎯 峰值检测算法
- 📱 PyQt5图形界面
- 📄 完整文档和示例

## ❓ 常见问题

### Q: 支持哪些文件格式？
A: 目前支持TDMS文件（Lab VIEW格式）、NPZ文件（numpy压缩格式）和CSV文件。

### Q: 程序无法启动怎么办？
A: 请确保安装了所有依赖包，特别是PyQt5。在Windows上可能需要安装Visual C++运行库。

### Q: 如何处理大文件？
A: 程序针对大文件进行了优化，使用pyqtgraph进行高性能绘图。建议内存不低于4GB。

### Q: 可以自定义峰值检测参数吗？
A: 是的，可以实时调整阈值、prominence、height等参数，并立即看到效果。

## 📞 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-username/current-event-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/current-event-analyzer/discussions)
- **Email**: your.email@example.com

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE) - 查看文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI框架
- [pyqtgraph](http://www.pyqtgraph.org/) - 高性能绘图库
- [SciPy](https://scipy.org/) - 科学计算库
- [NumPy](https://numpy.org/) - 数值计算基础
- [pandas](https://pandas.pydata.org/) - 数据分析工具

---

⭐ 如果这个项目对你有帮助，请给它一个星标！

