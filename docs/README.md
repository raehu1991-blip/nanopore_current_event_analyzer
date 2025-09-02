# Current Event Analyzer - 电流事件分析器

一个基于PyQt5的电流事件分析工具，支持TDMS和NPZ文件格式，提供交互式数据分析和多文件数据可视化功能。

## 项目结构

```
analyze_single_event/
├── src/                    # 源代码目录
│   └── main_analyzer.py   # 主分析器程序
├── docs/                  # 文档目录
│   ├── README.md          # 项目说明文档
│   └── requirements.md    # 需求文档
├── data/                  # 数据目录
│   └── test_data.csv      # 测试数据
├── examples/              # 示例目录
├── tests/                 # 测试目录
├── requirements.txt       # Python依赖包
└── run.py                # 主入口脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行程序

```bash
python run.py
```

## 功能模块

### 模块1：单事件分析 (Single Event Analysis)

#### 功能特点
- 支持TDMS、NPZ和BIN文件加载
- 所有文件格式的采样频率都是250kHz
- BIN文件的电流自动从安培(A)转换为纳安(nA)显示
- 横坐标是时间，纵坐标是电流
- 交互式图表显示（可缩放、平移）
- 阈值分析和峰值检测
- 峰值审核和选择性导出
- 背景颜色自定义

#### 使用说明
1. 点击"Load TDMS/NPZ/BIN File Folder"加载数据文件夹
2. 使用Previous/Next按钮浏览不同文件
3. 设置threshold值（相对于基线的正值，默认0.1），实际阈值 = 基线 - 相对阈值
4. 调整分析参数：prominence（默认0.08）和背景颜色
5. 点击"Analyze"进行峰值分析
6. 在"峰值审核"中选择要保存的峰值
7. 点击"Export Selected to CSV"导出选中的峰值数据
8. CSV包含：file_name, peak_number, total_time, peak_t, peak_i, peak_start_i, peak_start_t, peak_amplitude, peak_rel_t, peak_rel_i

### 模块2：多文件数据可视化 (Multi-file Data Visualization)

#### 功能特点
- 支持导入多个CSV文件
- 灵活的列选择和数据筛选
- 多种统计图表类型
- 跨文件数据对比分析
- 图表数据导出功能

#### 数据导入
- 支持批量导入CSV文件（支持增量导入，自动去重）
- 文件列表管理：显示已导入文件，支持清除所有数据
- 智能列选择：选择文件后显示该文件的所有列
- 列标识系统：选中列以"文件名.列名"格式标识
- 自动识别CSV文件结构和列名

#### 可视化类型

##### 分析模式
- **Box Plot**：箱线图，显示数据的分布和异常值
- **Histogram**：直方图，显示数据的频率分布
- **Scatter Plot**：散点图，显示两个变量之间的关系

#### 交互功能
- **文件管理**：增量导入、自动去重、清除所有数据
- **智能列选择**：
  - 点击文件显示该文件所有列
  - 添加列到选中列表（格式：文件名.列名）
  - 支持多文件多列混合选择
- **图表控制**：
  - 3种图表类型可选
  - Clear Plot：清除当前图表
  - Update Plot：刷新图表显示

#### 使用流程
1. **导入数据**：点击"导入CSV文件"，支持多次导入（自动去重）
2. **选择列**：
   - 点击文件列表中的文件，查看该文件的所有列
   - 双击列名添加到"选中列"（显示为"文件名.列名"）
   - 可从多个文件选择多个列进行混合分析
3. **设置图表参数**：
   - 选择Box Plot/Histogram/Scatter Plot
4. **绘制图表**：
   - 点击"Update Plot"：创建新图表
   - 点击"Clear Plot"：清除当前图表
5. **管理数据**：使用"Clear Data"清除所有导入的文件数据

## 界面布局

### 单事件分析界面
- 左侧：文件加载 → 数据分析 → 导出结果 → 峰值审核 → 分析参数 → 分析结果
- 右侧：交互式时间-电流图表

### 多文件可视化界面
- 左侧：文件管理 → 列选择 → 选中列显示
- 中心：统计图表显示区域（pyqtgraph）
- 右侧：图表控制

## 技术实现

### 核心算法
- 峰值检测：scipy.find_peaks
- 起点检测：从峰值向前寻找上升趋势结束点
- 统计分析：numpy统计函数
- 可视化：pyqtgraph

### 数据格式
- 输入：TDMS文件（250kHz采样）、NPZ文件、CSV文件
- 输出：标准化CSV格式，包含完整的峰值分析数据
  - peak_rel_t = (peak_t - event_start_t) / total_time（峰值相对时间比例）
  - peak_rel_i = -peak_amplitude / peak_start_i（峰值相对电流比例）

## 技术栈

- **GUI框架**：PyQt5
- **数据处理**：numpy, pandas
- **可视化**：pyqtgraph
- **科学计算**：scipy
- **文件格式**：npTDMS (TDMS文件支持)

## 开发环境要求

- Python 3.7+
- PyQt5 5.15.0+
- 其他依赖见 requirements.txt

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请提交 Issue 或联系开发者。

