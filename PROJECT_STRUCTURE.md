# 项目结构整理完成

## 整理后的项目结构

```
analyze_single_event/
├── src/                          # 源代码目录
│   └── main_analyzer.py         # 主分析器程序（整合版本）
├── docs/                        # 文档目录
│   ├── README.md               # 项目说明文档（更新版）
│   └── requirements.md         # 需求文档
├── data/                        # 数据目录
│   └── test_data.csv           # 测试数据
├── examples/                    # 示例目录
│   └── peak_detection_example.py # 峰值检测示例脚本
├── tests/                       # 测试目录
│   └── test_basic.py           # 基础功能测试
├── requirements.txt             # Python依赖包列表（更新版）
├── run.py                      # 主入口脚本
├── setup.py                    # 项目安装配置
└── PROJECT_STRUCTURE.md        # 本文件 - 项目结构说明
```

## 整理内容总结

### 1. 文件合并与整理
- **合并了多个版本的主程序**：`main.py`、`main_2.py`、`multi_func_analyzer.py` 整合为 `src/main_analyzer.py`
- **保留了最完整的功能**：包括单事件分析和多文件可视化功能
- **创建了独立的示例**：`examples/peak_detection_example.py` 展示峰值检测算法

### 2. 目录结构优化
- **src/**：存放所有源代码
- **docs/**：存放所有文档文件
- **data/**：存放数据文件
- **examples/**：存放示例脚本
- **tests/**：存放测试文件

### 3. 新增文件
- **run.py**：主入口脚本，简化程序启动
- **setup.py**：项目安装配置，支持pip安装
- **examples/peak_detection_example.py**：独立的峰值检测示例
- **tests/test_basic.py**：基础功能测试套件

### 4. 文档更新
- **docs/README.md**：完整的项目说明，包含使用方法和技术文档
- **docs/requirements.md**：详细的需求规格文档
- **requirements.txt**：更新的依赖包列表，包含所有必要的包

### 5. 删除的重复文件
- 根目录下的旧版本源文件
- 重复的文档文件
- 旧的数据文件

## 使用方法

### 启动程序
```bash
# 方法1：直接运行主入口脚本
python run.py

# 方法2：直接运行主程序
python src/main_analyzer.py
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行测试
```bash
python tests/test_basic.py
```

### 运行示例
```bash
python examples/peak_detection_example.py
```

## 主要功能

### 单事件分析模块
- TDMS和NPZ文件加载
- 交互式数据可视化
- 峰值检测和分析
- 可选择性数据导出
- 参数可调的分析算法

### 多文件可视化模块
- CSV文件批量导入
- 多种图表类型（Box Plot、Histogram、Scatter Plot）
- 跨文件数据对比
- 交互式图表控制

## 技术栈
- **PyQt5**：GUI框架
- **pyqtgraph**：高性能数据可视化
- **numpy/pandas**：数据处理
- **scipy**：科学计算和峰值检测
- **npTDMS**：TDMS文件支持

## 下一步改进建议
1. 添加更多的测试用例
2. 添加配置文件支持
3. 添加插件系统
4. 优化大文件处理性能
5. 添加更多的数据导出格式

## 维护说明
- 主要源代码在 `src/main_analyzer.py`
- 文档在 `docs/` 目录下
- 新功能测试添加到 `tests/` 目录
- 示例脚本添加到 `examples/` 目录
