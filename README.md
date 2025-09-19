# Current Event Analyzer - 电流事件分析器

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/your-username/current-event-analyzer)

一个基于PyQt5的专业电流事件分析工具，支持多种文件格式，提供先进的峰值检测算法和强大的多文件数据可视化功能。

![Main Interface](docs/images/main_interface.png)

## ✨ 主要功能

### 🔬 单事件分析模块
- **多格式支持**：TDMS文件（250kHz采样）、NPZ文件和BIN文件
- **交互式可视化**：可缩放、平移的时间-电流图表，支持十字准线定位
- **多种峰值检测算法**：
  - Scipy find_peaks（基础算法）
  - 小波变换检测（PyWavelets）
  - 机器学习检测（scikit-learn）
- **智能参数调节**：阈值、prominence、小波参数等实时调节
- **峰值审核系统**：支持保留/删除标记，Position Label编辑
- **选择性导出**：仅导出标记为保留的峰值数据
- **视觉定制**：可自定义背景颜色、显示prominence竖线等

### 📊 多文件可视化模块
- **批量导入**：支持多个CSV文件同时导入和分析，自动去重
- **多种图表类型**：
  - 单列分析：Box Plot、Histogram、Violin Plot
  - 双列分析：Scatter Plot、KDE Plot
- **累积绘图**：支持多次绘图叠加，不同颜色区分
- **跨文件对比**：不同数据源的统计对比分析
- **交互式控制**：实时图表更新、清除和坐标显示
- **智能列选择**：自动识别数值列，支持搜索过滤
- **数据导出**：支持导出当前数据、统计摘要和图像

## 🌟 特色功能

### 🔍 智能峰值检测
- **多算法支持**：三种不同的峰值检测算法，适应不同数据特征
- **参数可视化**：实时显示prominence竖线，直观展示峰值显著性
- **基线自动计算**：智能计算基线值，提高检测准确性

### 📋 专业审核系统
- **峰值标记**：支持保留/删除标记，便于质量控制
- **Position Label**：可编辑的位置标签，支持数据分类
- **选择性导出**：仅导出审核通过的峰值数据

### 📊 高级可视化
- **累积绘图**：支持多次绘图叠加，便于对比分析
- **交互式操作**：十字准线定位、坐标实时显示
- **多格式支持**：Box Plot、Violin Plot、KDE Plot等专业图表

### 🚀 性能优化
- **大文件支持**：优化的内存使用，支持大文件处理
- **实时响应**：参数调整即时生效，无需重新加载
- **跨平台兼容**：支持Windows、macOS、Linux系统

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

### 可选依赖（增强功能）
- **PyWavelets** 1.3.0+ - 小波变换峰值检测
- **scikit-learn** 1.0.0+ - 机器学习峰值检测
- **matplotlib** 3.4.0+ - 额外的绘图功能
- **seaborn** 0.11.0+ - 统计图表美化
- **pytest** 6.0.0+ - 测试框架

## 📖 使用指南

### 单事件分析

1. **加载数据**：点击"Load File Folder"选择包含TDMS/NPZ/BIN文件的文件夹
2. **选择算法**：从下拉菜单选择峰值检测算法（Scipy、小波变换、机器学习）
3. **设置参数**：
   - 阈值(Threshold)：相对于基线的阈值
   - Prominence：峰值显著性参数
   - 小波参数：小波类型、尺度范围、阈值（小波算法）
   - ML模型：预训练模型路径（机器学习算法）
4. **执行分析**：点击"Analyze"按钮进行峰值检测
5. **审核结果**：
   - 在峰值表格中查看检测结果
   - 使用复选框选择要保留的峰值
   - 点击"保留"/"删除"按钮标记峰值状态
   - 编辑Position Labels（可选）
6. **导出数据**：点击"Export to CSV"保存标记为保留的峰值数据

### 多文件可视化

1. **导入文件**：点击"导入CSV文件"批量加载数据（支持增量导入）
2. **选择分析模式**：
   - 单列分析：Box Plot、Histogram、Violin Plot
   - 双列分析：Scatter Plot、KDE Plot
3. **选择数据**：
   - 单列模式：勾选要分析的列（可多选）
   - 双列模式：选择X轴和Y轴列
4. **生成图表**：点击"Update Plot"创建可视化（支持累积绘图）
5. **管理数据**：
   - 使用"Clear Plot"清除当前图表
   - 使用"Clear Data"清除所有数据
   - 导出当前数据、统计摘要或图像

## 🔬 算法原理

### 峰值检测算法

#### 1. Scipy find_peaks（基础算法）
- 使用 `scipy.signal.find_peaks` 进行峰值识别
- 支持prominence（显著性）和distance（距离）参数
- 自动计算峰值起点和相对参数
- 适用于大多数标准电流事件检测

#### 2. 小波变换检测（Wavelet Transform）
- 使用PyWavelets库进行连续小波变换
- 支持多种小波类型：db4、db6、sym4、coif4、haar
- 可调节尺度范围和阈值参数
- 适用于复杂噪声环境下的峰值检测

#### 3. 机器学习检测（Machine Learning）
- 基于scikit-learn的随机森林分类器
- 提取滑动窗口的统计特征（均值、标准差、偏度、峰度等）
- 支持预训练模型加载
- 适用于需要高精度检测的场景

### 数据处理流程
1. **数据加载**：
   - TDMS文件：按250kHz采样率处理
   - NPZ文件：直接加载current和time数组
   - BIN文件：按250kHz采样率读取，自动转换为纳安(nA)
2. **基线计算**：使用前200个数据点的平均值作为基线
3. **阈值筛选**：提取低于设定阈值的数据段
4. **峰值检测**：根据选择的算法识别显著峰值
5. **特征计算**：计算振幅、时间、相对参数、左右基点等
6. **结果输出**：标准化CSV格式导出，支持Position Label

## 📊 输出数据格式

CSV文件包含以下字段：

| 字段名 | 说明 | 单位 |
|--------|------|------|
| file_name | 源文件名 | - |
| peak_number | 峰值序号 | - |
| total_time | 事件总时间 | 秒 |
| peak_t | 峰值时间 | 秒 |
| peak_i | 峰值电流 | 安培/纳安 |
| peak_start_t | 起点时间 | 秒 |
| peak_start_i | 起点电流 | 安培/纳安 |
| peak_amplitude | 峰值振幅（prominence值） | 安培/纳安 |
| peak_rel_t | 相对时间比例 | - |
| peak_rel_i | 相对电流比例 | - |
| left_base_t | 左基点时间 | 秒 |
| left_base_i | 左基点电流 | 安培/纳安 |
| right_base_t | 右基点时间 | 秒 |
| right_base_i | 右基点电流 | 安培/纳安 |
| position_label | 位置标签（可选） | - |

### 单位说明
- **TDMS/NPZ文件**：电流单位为安培(A)
- **BIN文件**：电流单位为纳安(nA)，程序自动转换
- **时间单位**：所有时间相关字段均为秒(s)

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

### v2.0.0 (2024-12-19)
- 🆕 新增BIN文件格式支持
- 🔬 三种峰值检测算法：Scipy、小波变换、机器学习
- 📊 增强的多文件可视化：累积绘图、更多图表类型
- 🏷️ Position Label编辑功能
- ✅ 峰值审核系统：保留/删除标记
- 🎨 视觉增强：十字准线、prominence竖线显示
- 📈 改进的导出功能：数据、统计、图像导出
- 🔧 参数优化：小波参数、ML模型支持

### v1.0.0 (2024-09-01)
- ✨ 初始版本发布
- 🔬 单事件分析功能
- 📊 多文件可视化功能
- 🎯 峰值检测算法
- 📱 PyQt5图形界面
- 📄 完整文档和示例

## ❓ 常见问题

### Q: 支持哪些文件格式？
A: 目前支持TDMS文件（Lab VIEW格式）、NPZ文件（numpy压缩格式）、BIN文件（二进制格式）和CSV文件。

### Q: 程序无法启动怎么办？
A: 请确保安装了所有依赖包，特别是PyQt5。在Windows上可能需要安装Visual C++运行库。macOS用户需要设置环境变量。

### Q: 如何处理大文件？
A: 程序针对大文件进行了优化，使用pyqtgraph进行高性能绘图。建议内存不低于4GB。BIN文件会自动转换为纳安单位。

### Q: 可以自定义峰值检测参数吗？
A: 是的，支持三种检测算法：
- Scipy：可调整阈值、prominence等参数
- 小波变换：可调整小波类型、尺度范围、阈值
- 机器学习：支持预训练模型加载

### Q: 如何编辑Position Labels？
A: 在峰值审核界面，点击"Position Labels"按钮，可以为标记为保留的峰值设置位置标签。

### Q: 多文件可视化支持哪些图表类型？
A: 支持单列分析（Box Plot、Histogram、Violin Plot）和双列分析（Scatter Plot、KDE Plot），支持累积绘图。

### Q: 如何导出分析结果？
A: 单事件分析：导出标记为保留的峰值数据；多文件可视化：支持导出当前数据、统计摘要和图像。

### Q: 小波变换和机器学习算法需要额外安装什么？
A: 小波变换需要安装PyWavelets：`pip install PyWavelets`
机器学习需要安装scikit-learn：`pip install scikit-learn`

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

