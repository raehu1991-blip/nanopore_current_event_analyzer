from pickle import FALSE
import sys
import os
import warnings

# 抑制一些常见的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# macOS 特定设置
if sys.platform == "darwin":
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-logging'

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from scipy.signal import find_peaks
from nptdms import TdmsFile
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import stats

# Conditional imports for advanced peak detection methods
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class SingleEventAnalyzer(QWidget):
    """单事件分析界面"""
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.time_data = None
        self.sample_rate = 250000  # 250kHz as specified
        self.current_folder = None
        self.file_list = []
        self.current_file_index = 0
        self.peaks_data = []
        self.current_file_type = None  # 跟踪当前文件类型以确定单位
        self.labels_edited = False  # 跟踪是否编辑过position labels
        
        self.initUI()
        
    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)  # Reduced width
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)  # Reduce spacing between groups
        left_panel.setLayout(left_layout)
        
        # File loading section
        file_group = QGroupBox("文件加载")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Load File Folder")
        self.load_btn.clicked.connect(self.load_file_folder)
        file_layout.addWidget(self.load_btn)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.load_previous_file)
        self.next_btn.clicked.connect(self.load_next_file)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        file_layout.addLayout(nav_layout)
        
        self.file_info_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_info_label)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Analysis section
        analysis_group = QGroupBox("数据分析")
        analysis_layout = QVBoxLayout()
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_data)
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
        
        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)
        
        
        # Peak Review section
        review_group = QGroupBox("峰值审核")
        review_layout = QVBoxLayout()
        
        # Peak table with checkboxes and position labels
        self.peak_table_widget = QTableWidget()
        self.peak_table_widget.setMaximumHeight(300)
        self.peak_table_widget.setColumnCount(7)
        self.peak_table_widget.setHorizontalHeaderLabels(["选择", "峰值编号","审核状态", "Position Label", "时间 (s)", "Prominence", "宽度 (ms)"])
        
        # Set column widths
        self.peak_table_widget.setColumnWidth(0, 50)  # 选择列
        self.peak_table_widget.setColumnWidth(1, 80)  # 峰值编号
        self.peak_table_widget.setColumnWidth(2, 80) # 审核状态
        self.peak_table_widget.setColumnWidth(3, 80) # Position Label
        self.peak_table_widget.setColumnWidth(4, 80)  # 时间
        self.peak_table_widget.setColumnWidth(5, 80) # Prominence
        self.peak_table_widget.setColumnWidth(6, 80) # 宽度
        
        # Make the table read-only except for position label column
        self.peak_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # Connect cell click event for position label column
        self.peak_table_widget.cellClicked.connect(self.on_peak_table_cell_clicked)
        
        review_layout.addWidget(self.peak_table_widget)
        
        # Select all/none buttons
        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_none_btn = QPushButton("全不选")
        self.select_all_btn.clicked.connect(self.select_all_peaks)
        self.select_none_btn.clicked.connect(self.select_none_peaks)
        self.select_all_btn.setEnabled(False)
        self.select_none_btn.setEnabled(False)
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.select_none_btn)
        review_layout.addLayout(select_layout)
        
        # Review status control buttons
        status_layout = QHBoxLayout()
        self.mark_keep_btn = QPushButton("保留")
        self.mark_delete_btn = QPushButton("删除")
        self.clear_status_btn = QPushButton("清除状态")
        self.mark_keep_btn.clicked.connect(self.mark_selected_as_keep)
        self.mark_delete_btn.clicked.connect(self.mark_selected_as_delete)
        self.clear_status_btn.clicked.connect(self.clear_selected_status)
        self.mark_keep_btn.setEnabled(False)
        self.mark_delete_btn.setEnabled(False)
        self.clear_status_btn.setEnabled(False)
        status_layout.addWidget(self.mark_keep_btn)
        status_layout.addWidget(self.mark_delete_btn)
        status_layout.addWidget(self.clear_status_btn)
        review_layout.addLayout(status_layout)
        
        # Position label editing and export buttons
        label_layout = QHBoxLayout()
        self.edit_labels_btn = QPushButton("Position Labels")
        self.edit_labels_btn.clicked.connect(self.edit_position_labels)
        self.edit_labels_btn.setEnabled(False)
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        self.export_btn.setEnabled(False)
        label_layout.addWidget(self.edit_labels_btn)
        label_layout.addWidget(self.export_btn)
        review_layout.addLayout(label_layout)
        
        review_group.setLayout(review_layout)
        left_layout.addWidget(review_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # Center panel for plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', '电流 (A)', 'A')
        self.plot_widget.setLabel('bottom', '时间 (s)', 's')
        self.plot_widget.showGrid(True, True)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Set white background
        self.plot_widget.setBackground('w')
        
        # Add crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen='y')
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Connect mouse move event
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Add coordinate label
        self.coord_label = QLabel("Position: (0, 0)")
        
        # Create scrollable plot area
        plot_panel = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addWidget(self.coord_label)
        plot_panel.setLayout(plot_layout)
        
        # Add scroll area for plot panel
        scroll_area = QScrollArea()
        scroll_area.setWidget(plot_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_layout.addWidget(scroll_area, stretch=1)
        
        # Right panel for analysis parameters
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_panel.setLayout(right_layout)
        
        # Analysis parameters section
        param_group = QGroupBox("分析参数")
        param_layout = QVBoxLayout()
        
        # Algorithm selection
        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("峰值检测算法:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Scipy find_peaks", "Wavelet Transform", "Machine Learning"])
        algorithm_layout.addWidget(self.algorithm_combo)
        param_layout.addLayout(algorithm_layout)
        
        # Threshold setting
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("阈值 (绝对值):"))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-100.0, 100.0)  # 绝对值范围
        self.threshold_input.setValue(0.1)  # 默认绝对阈值
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setDecimals(3)
        thresh_layout.addWidget(self.threshold_input)
        param_layout.addLayout(thresh_layout)
        
        # Peak detection prominence parameter (更有效的峰值控制)
        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(QLabel("Prominence:"))
        self.prominence_input = QDoubleSpinBox()
        self.prominence_input.setRange(0.001, 1.0)
        self.prominence_input.setValue(0.08)  # 默认值
        self.prominence_input.setSingleStep(0.001)
        self.prominence_input.setDecimals(4)
        prominence_layout.addWidget(self.prominence_input)
        param_layout.addLayout(prominence_layout)
        
        # Window length parameter for peak detection
        wlen_layout = QHBoxLayout()
        wlen_layout.addWidget(QLabel("Window Length:"))
        self.wlen_input = QSpinBox()
        self.wlen_input.setRange(1, 200)
        self.wlen_input.setValue(40)  # 默认值
        self.wlen_input.setSingleStep(1)
        self.wlen_input.setToolTip("用于计算prominence的窗口长度（采样点数）")
        wlen_layout.addWidget(self.wlen_input)
        param_layout.addLayout(wlen_layout)
        
        # Wavelet parameters (initially hidden)
        wavelet_layout = QHBoxLayout()
        self.wavelet_type_label = QLabel("小波类型:")
        wavelet_layout.addWidget(self.wavelet_type_label)
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["db4", "db6", "sym4", "coif4", "haar"])
        wavelet_layout.addWidget(self.wavelet_combo)
        param_layout.addLayout(wavelet_layout)
        
        scales_layout = QHBoxLayout()
        self.wavelet_scales_label = QLabel("小波尺度:")
        scales_layout.addWidget(self.wavelet_scales_label)
        self.scales_input = QLineEdit("1-32")
        self.scales_input.setToolTip("小波尺度范围，例如: 1-32")
        scales_layout.addWidget(self.scales_input)
        param_layout.addLayout(scales_layout)
        
        wavelet_thresh_layout = QHBoxLayout()
        self.wavelet_thresh_label = QLabel("小波阈值:")
        wavelet_thresh_layout.addWidget(self.wavelet_thresh_label)
        self.wavelet_threshold_input = QDoubleSpinBox()
        self.wavelet_threshold_input.setRange(0.1, 10.0)
        self.wavelet_threshold_input.setValue(3.0)
        self.wavelet_threshold_input.setSingleStep(0.1)
        wavelet_thresh_layout.addWidget(self.wavelet_threshold_input)
        param_layout.addLayout(wavelet_thresh_layout)
        
        # ML parameters (initially hidden)
        ml_model_layout = QHBoxLayout()
        self.ml_model_label = QLabel("ML模型路径:")
        ml_model_layout.addWidget(self.ml_model_label)
        self.ml_model_input = QLineEdit()
        self.ml_model_input.setPlaceholderText("可选: 预训练模型路径")
        ml_model_layout.addWidget(self.ml_model_input)
        param_layout.addLayout(ml_model_layout)
        
        ml_browse_layout = QHBoxLayout()
        self.ml_browse_btn = QPushButton("浏览模型文件")
        self.ml_browse_btn.clicked.connect(self.browse_ml_model)
        ml_browse_layout.addWidget(self.ml_browse_btn)
        param_layout.addLayout(ml_browse_layout)
        
        # Background color selection
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("背景颜色:"))
        self.bg_color_combo = QComboBox()
        self.bg_color_combo.addItems(["白色", "黑色", "浅灰色", "深灰色"])
        self.bg_color_combo.setCurrentText("白色")  # 设置默认选择白色
        self.bg_color_combo.currentTextChanged.connect(self.change_background_color)
        bg_layout.addWidget(self.bg_color_combo)
        param_layout.addLayout(bg_layout)
        
        # Prominence line display toggle
        prominence_display_layout = QHBoxLayout()
        self.show_prominence_checkbox = QCheckBox("显示Prominence竖线")
        self.show_prominence_checkbox.setChecked(True)  # 默认勾选
        self.show_prominence_checkbox.stateChanged.connect(self.toggle_prominence_display)
        prominence_display_layout.addWidget(self.show_prominence_checkbox)
        param_layout.addLayout(prominence_display_layout)
        
        # FWHM line display toggle
        fwhm_display_layout = QHBoxLayout()
        self.show_fwhm_checkbox = QCheckBox("显示半高宽标记线")
        self.show_fwhm_checkbox.setChecked(True)  # 默认勾选
        self.show_fwhm_checkbox.stateChanged.connect(self.toggle_fwhm_display)
        fwhm_display_layout.addWidget(self.show_fwhm_checkbox)
        param_layout.addLayout(fwhm_display_layout)
        
        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)
        
        # Connect algorithm selection change (moved after algorithm_combo definition)
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        # Initially hide wavelet and ML parameters
        self.hide_advanced_parameters()
        
        right_layout.addStretch()
        main_layout.addWidget(right_panel)
    
    # 这里包含所有单事件分析的方法（从原main.py复制）
    def mouse_moved(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self.coord_label.setText(f"Position: ({x:.6f}, {y:.6f})")
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
    
    def change_background_color(self, color_name):
        """改变图表背景颜色"""
        color_map = {
            "白色": 'w',
            "黑色": 'k', 
            "浅灰色": (240, 240, 240),
            "深灰色": (64, 64, 64)
        }
        
        if color_name in color_map:
            self.plot_widget.setBackground(color_map[color_name])
    
    def toggle_prominence_display(self, state):
        """切换prominence竖线的显示状态"""
        # 如果当前有分析结果，重新绘制图表
        if hasattr(self, 'peaks_data') and self.peaks_data:
            self.plot_data()
            # 重新绘制分析结果（包括prominence线）
            self.redraw_analysis_results()
    
    def toggle_fwhm_display(self, state):
        """切换半高宽标记线的显示状态"""
        # 如果当前有分析结果，重新绘制图表
        if hasattr(self, 'peaks_data') and self.peaks_data:
            self.plot_data()
            # 重新绘制分析结果（包括半高宽线）
            self.redraw_analysis_results()
    
    def draw_prominence_lines(self, peak_times, peak_currents, prominences):
        """绘制prominence竖线"""
        # 检查是否应该显示prominence线
        if not self.show_prominence_checkbox.isChecked():
            return
            
        for i, (t, curr, prominence_val) in enumerate(zip(peak_times, peak_currents, prominences)):
            ymin = curr  # 竖线底部（峰值点）
            ymax = curr + prominence_val  # 竖线顶部（向上延伸prominence值）
            
            # 使用pyqtgraph的PlotDataItem绘制竖线
            prominence_line = self.plot_widget.plot([t, t], [ymin, ymax], 
                                                  pen=pg.mkPen(color=(0, 0, 255), width=2), 
                                                  name=f'Prominence_{i+1}' if i == 0 else None)
    
    def draw_fwhm_lines(self, peak_times, peak_currents, peak_widths_us):
        """绘制半高全宽标记线"""
        # 检查是否应该显示半高宽线
        if not self.show_fwhm_checkbox.isChecked():
            return
            
        for i, peak_data in enumerate(self.peaks_data):
            if i >= len(peak_times):
                break
                
            width_us = peak_data.get('peak_width_us', 0)
            if width_us <= 0:
                continue
                
            # 使用scipy.peak_widths计算的实际半高宽边界
            left_time = peak_data.get('fwhm_left_time')
            right_time = peak_data.get('fwhm_right_time')
            fwhm_height = peak_data.get('fwhm_height')
            
            if left_time is None or right_time is None or fwhm_height is None:
                continue
            
            # 绘制水平线表示半高宽
            fwhm_line = self.plot_widget.plot([left_time, right_time], [fwhm_height, fwhm_height], 
                                            pen=pg.mkPen(color=(255, 165, 0), width=2, style=Qt.DashLine), 
                                            name=f'FWHM_{i+1}' if i == 0 else None)
            
            # 绘制左右边界线
            height_range = abs(fwhm_height) * 0.1  # 根据高度动态调整边界线长度
            left_boundary = self.plot_widget.plot([left_time, left_time], 
                                                [fwhm_height - height_range, fwhm_height + height_range], 
                                                pen=pg.mkPen(color=(255, 165, 0), width=2))
            right_boundary = self.plot_widget.plot([right_time, right_time], 
                                                 [fwhm_height - height_range, fwhm_height + height_range], 
                                                 pen=pg.mkPen(color=(255, 165, 0), width=2))
            
            # 添加半高宽标签
            fwhm_text = pg.TextItem(f'FWHM: {width_us:.1f}μs', 
                                  color=(255, 165, 0), 
                                  anchor=(0.5, -0.5))
            fwhm_text.setPos(peak_times[i], fwhm_height)
            self.plot_widget.addItem(fwhm_text)
    
    def redraw_analysis_results(self):
        """重新绘制分析结果"""
        if not hasattr(self, 'peaks_data') or not self.peaks_data:
            return
            
        # 重新绘制峰值标记
        peak_times = [peak['peak_t'] for peak in self.peaks_data]
        peak_currents = [peak['peak_i'] for peak in self.peaks_data]
        
        if len(peak_times) > 0:
            self.plot_widget.plot(peak_times, peak_currents, 
                                pen=None, symbol='o', symbolBrush=(255, 0, 0), symbolSize=8, name='峰值')
            
            # 重新绘制prominence竖线
            prominences = [peak['peak_amplitude'] for peak in self.peaks_data]
            self.draw_prominence_lines(peak_times, peak_currents, prominences)
            
            # 重新绘制半高宽线
            peak_widths_us = [peak['peak_width_us'] for peak in self.peaks_data]
            self.draw_fwhm_lines(peak_times, peak_currents, peak_widths_us)
            
            # 重新绘制prominence标签
            for i, (t, curr, prominence_val) in enumerate(zip(peak_times, peak_currents, prominences)):
                text_item = pg.TextItem(f'P{i+1}', 
                                      color=(255, 0, 0), 
                                      anchor=(0.5, 1.5))
                text_item.setPos(t, curr)
                self.plot_widget.addItem(text_item)
    
    def update_current_unit_label(self):
        """根据文件类型更新电流单位标签"""
        if self.current_file_type == 'bin':
            # BIN文件显示为纳安
            self.plot_widget.setLabel('left', '电流 (nA)', 'nA')
        elif self.current_file_type == 'txt':
            # TXT文件显示为安培
            self.plot_widget.setLabel('left', '电流 (A)', 'A')
        else:
            # TDMS和NPZ文件显示为安培
            self.plot_widget.setLabel('left', '电流 (A)', 'A')
    
    def get_current_unit(self):
        """获取当前电流单位"""
        if self.current_file_type == 'bin':
            return 'nA'
        elif self.current_file_type == 'txt':
            return 'A'
        else:
            return 'A'
    
    def calculate_baseline(self):
        """计算基线值：使用前200个数据点的平均值"""
        if self.current_data is None or len(self.current_data) == 0:
            return 0.0
        
        # 取前200个点，如果数据不足200个点则取全部
        baseline_points = min(200, len(self.current_data))
        baseline_data = self.current_data[:baseline_points]
        
        return np.mean(baseline_data)

    def detect_peaks_wavelet(self, data, scales=None, wavelet='db4', threshold=3.0):
        """使用小波变换进行峰值检测"""
        if not HAS_PYWT:
            raise ImportError("PyWavelets (pywt) is required for wavelet transform peak detection. Install with: pip install PyWavelets")
        
        if scales is None:
            scales = range(1, 32)
        
        # 计算连续小波变换
        coefficients, frequencies = pywt.cwt(data, scales, wavelet)
        
        # 计算小波系数的绝对值
        abs_coefficients = np.abs(coefficients)
        
        # 通过阈值检测峰值
        peaks = []
        for i in range(len(scales)):
            # 在每个尺度上找到局部最大值
            scale_peaks, _ = find_peaks(abs_coefficients[i], height=threshold)
            peaks.extend(scale_peaks)
        
        # 去除重复的峰值并排序
        peaks = sorted(set(peaks))
        
        # 过滤掉边界附近的峰值
        peaks = [p for p in peaks if 10 < p < len(data) - 10]
        
        return np.array(peaks)

    def detect_peaks_ml(self, data, model_path=None):
        """使用机器学习进行峰值检测"""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for machine learning peak detection. Install with: pip install scikit-learn")
        
        # 提取特征：使用滑动窗口计算统计特征
        window_size = 20
        features = []
        positions = []
        
        for i in range(window_size, len(data) - window_size):
            window = data[i-window_size:i+window_size]
            features.append([
                np.mean(window),      # 平均值
                np.std(window),       # 标准差
                np.min(window),       # 最小值
                np.max(window),       # 最大值
                np.ptp(window),       # 峰峰值
                stats.skew(window),   # 偏度
                stats.kurtosis(window) # 峰度
            ])
            positions.append(i)
        
        features = np.array(features)
        positions = np.array(positions)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 加载预训练模型或使用默认模型
        if model_path and os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # 使用简单的基于规则的方法作为默认模型
            # 在实际应用中，应该训练一个真正的模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # 这里简化处理，实际需要训练数据
            # 暂时使用阈值方法作为替代
            predictions = (data[positions] < np.mean(data) - np.std(data)).astype(int)
            model.fit(features_scaled, predictions)
        
        # 预测峰值
        predictions = model.predict(features_scaled)
        peak_indices = positions[predictions == 1]
        
        return peak_indices
    
    def hide_advanced_parameters(self):
        """隐藏小波和ML参数"""
        # Hide wavelet parameters and labels
        self.wavelet_type_label.hide()
        self.wavelet_combo.hide()
        self.wavelet_scales_label.hide()
        self.scales_input.hide()
        self.wavelet_thresh_label.hide()
        self.wavelet_threshold_input.hide()
        
        # Hide ML parameters and labels
        self.ml_model_label.hide()
        self.ml_model_input.hide()
        self.ml_browse_btn.hide()

    def on_algorithm_changed(self, algorithm):
        """当算法选择改变时，显示或隐藏高级参数"""
        if algorithm == "Wavelet Transform":
            # Show wavelet parameters and labels, hide ML parameters
            self.wavelet_type_label.show()
            self.wavelet_combo.show()
            self.wavelet_scales_label.show()
            self.scales_input.show()
            self.wavelet_thresh_label.show()
            self.wavelet_threshold_input.show()
            self.ml_model_label.hide()
            self.ml_model_input.hide()
            self.ml_browse_btn.hide()
        elif algorithm == "Machine Learning":
            # Show ML parameters and labels, hide wavelet parameters
            self.wavelet_type_label.hide()
            self.wavelet_combo.hide()
            self.wavelet_scales_label.hide()
            self.scales_input.hide()
            self.wavelet_thresh_label.hide()
            self.wavelet_threshold_input.hide()
            self.ml_model_label.show()
            self.ml_model_input.show()
            self.ml_browse_btn.show()
        else:
            # Hide both for Scipy find_peaks
            self.wavelet_type_label.hide()
            self.wavelet_combo.hide()
            self.wavelet_scales_label.hide()
            self.scales_input.hide()
            self.wavelet_thresh_label.hide()
            self.wavelet_threshold_input.hide()
            self.ml_model_label.hide()
            self.ml_model_input.hide()
            self.ml_browse_btn.hide()

    def browse_ml_model(self):
        """打开文件对话框选择ML模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择ML模型文件", "", "Model files (*.pkl *.joblib);;All files (*.*)")
        if file_path:
            self.ml_model_input.setText(file_path)

    def load_file_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含TDMS/NPZ/BIN/TXT文件的文件夹")
        if folder_path:
            self.current_folder = folder_path
            self.file_list = []
            
            # Find all TDMS, NPZ, BIN and TXT files
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.tdms', '.npz', '.bin', '.txt')):
                    self.file_list.append(os.path.join(folder_path, file))
            
            if self.file_list:
                self.file_list.sort()
                self.current_file_index = 0
                self.load_current_file()
                self.prev_btn.setEnabled(len(self.file_list) > 1)
                self.next_btn.setEnabled(len(self.file_list) > 1)
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中未找到TDMS、NPZ、BIN或TXT文件。")
    
    def load_current_file(self):
        if not self.file_list:
            return
            
        file_path = self.file_list[self.current_file_index]
        file_name = os.path.basename(file_path)
        
        try:
            if file_path.lower().endswith('.tdms'):
                self.current_file_type = 'tdms'
                self.load_tdms_file(file_path)
            elif file_path.lower().endswith('.npz'):
                self.current_file_type = 'npz'
                self.load_npz_file(file_path)
            elif file_path.lower().endswith('.bin'):
                self.current_file_type = 'bin'
                self.load_bin_file(file_path)
            elif file_path.lower().endswith('.txt'):
                self.current_file_type = 'txt'
                self.load_txt_file(file_path)
                
            self.file_info_label.setText(f"文件 {self.current_file_index + 1}/{len(self.file_list)}: {file_name}")
            self.plot_data()
            self.analyze_btn.setEnabled(True)
            
            # Reset analysis results
            self.peaks_data = []
            self.peak_table_widget.setRowCount(0)  # 清空表格
            self.labels_edited = False  # 重置labels编辑状态
            self.export_btn.setEnabled(False)
            self.select_all_btn.setEnabled(False)
            self.select_none_btn.setEnabled(False)
            self.mark_keep_btn.setEnabled(False)
            self.mark_delete_btn.setEnabled(False)
            self.clear_status_btn.setEnabled(False)
            self.edit_labels_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件失败: {str(e)}")
    
    def load_tdms_file(self, file_path):
        with TdmsFile.read(file_path) as tdms_file:
            groups = tdms_file.groups()
            if not groups:
                raise ValueError("TDMS文件中未找到数据组")
            
            # Find the channel with current data
            current_channel = None
            
            for group in groups:
                for channel in group.channels():
                    # Look for current data
                    if 'current' in channel.name.lower() or current_channel is None:
                        current_channel = channel
            
            if current_channel is None:
                raise ValueError("TDMS文件中未找到电流数据通道")
            
            self.current_data = np.array(current_channel.data)
            # Generate time array with 250kHz sampling rate
            self.time_data = np.arange(len(self.current_data)) / self.sample_rate
    
    def load_npz_file(self, file_path):
        data = np.load(file_path)
        
        if 'current' not in data or 'time' not in data:
            raise ValueError("NPZ文件必须包含 'current' 和 'time' 数组")
        
        self.current_data = data['current']
        self.time_data = data['time']
    
    def load_bin_file(self, file_path):
        """加载二进制文件，假设采样频率为250kHz，将电流从A转换为nA"""
        try:
            # 尝试以double（8字节）格式读取二进制数据
            try:
                raw_current_data = np.fromfile(file_path, dtype=np.float64)
            except:
                # 如果失败，尝试以float（4字节）格式读取
                try:
                    raw_current_data = np.fromfile(file_path, dtype=np.float32)
                except:
                    # 最后尝试以16位整数格式读取
                    raw_data = np.fromfile(file_path, dtype=np.int16)
                    # 转换为浮点数，假设数据需要归一化
                    raw_current_data = raw_data.astype(np.float64)
            
            if len(raw_current_data) == 0:
                raise ValueError("BIN文件为空或格式不支持")
            
            # 将电流从安培(A)转换为纳安(nA): 1 A = 1e9 nA
            self.current_data = raw_current_data * 1e9
            
            # 使用250kHz采样频率生成时间数组
            self.time_data = np.arange(len(self.current_data)) / self.sample_rate
            
        except Exception as e:
            raise ValueError(f"加载BIN文件失败: {str(e)}")
    
    def load_txt_file(self, file_path):
        """加载TXT文件，支持多种TXT格式"""
        try:
            file_name = os.path.basename(file_path).lower()
            
            # 检查是否是时间文件
            is_time_file = ('time' in file_name or 'timed' in file_name)
            
            # 读取TXT文件
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                raise ValueError("TXT文件为空")
            
            # 解析数据
            data_values = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 支持多种格式
                    if '|' in line:
                        # 格式: "行号|数值"
                        parts = line.split('|')
                        if len(parts) >= 2:
                            value = float(parts[1])
                            data_values.append(value)
                    elif '\t' in line:
                        # 制表符分隔格式
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            value = float(parts[1])
                            data_values.append(value)
                        elif len(parts) == 1:
                            # 只有一列数据
                            value = float(parts[0])
                            data_values.append(value)
                    elif ',' in line:
                        # 逗号分隔格式
                        parts = line.split(',')
                        if len(parts) >= 2:
                            value = float(parts[1])
                            data_values.append(value)
                        elif len(parts) == 1:
                            # 只有一列数据
                            value = float(parts[0])
                            data_values.append(value)
                    elif ' ' in line and len(line.split()) > 1:
                        # 空格分隔格式（多列）
                        parts = line.split()
                        value = float(parts[1])
                        data_values.append(value)
                    else:
                        # 单列数据（最常见的情况）
                        value = float(line)
                        data_values.append(value)
                        
                except (ValueError, IndexError) as e:
                    print(f"警告: 第{line_num}行数据格式错误，跳过: {line.strip()}")
                    continue
            
            if len(data_values) == 0:
                raise ValueError("未找到有效的数值数据")
            
            data_array = np.array(data_values)
            
            if is_time_file:
                # 这是时间文件，加载为时间数据
                self.time_data = data_array
                # 创建对应的虚拟电流数据（全零）
                self.current_data = np.zeros_like(data_array)
                print(f"Debug: 加载时间文件，{len(data_array)}个时间点")
            else:
                # 这是电流文件，加载为电流数据
                self.current_data = data_array
                # 使用250kHz采样频率生成时间数组
                self.time_data = np.arange(len(data_array)) / self.sample_rate
                print(f"Debug: 加载电流文件，{len(data_array)}个数据点")
            
            print(f"Debug: TXT文件加载成功")
            print(f"Debug: 电流数据范围: {self.current_data.min():.6f} 到 {self.current_data.max():.6f}")
            print(f"Debug: 时间数据范围: {self.time_data.min():.6f} 到 {self.time_data.max():.6f}")
            
        except Exception as e:
            raise ValueError(f"加载TXT文件失败: {str(e)}")
    
    def load_previous_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()
    
    def load_next_file(self):
        if self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self.load_current_file()
    
    def plot_data(self):
        if self.current_data is None or self.time_data is None:
            return
        
        self.plot_widget.clear()
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # 更新电流单位标签
        self.update_current_unit_label()
        
        # Plot the current data
        self.plot_widget.plot(self.time_data, self.current_data, pen='b', name='电流')
        
        # Add threshold line (using absolute threshold)
        absolute_threshold = self.threshold_input.value()
        actual_threshold = absolute_threshold
        
        # Get selected algorithm and parameters
        algorithm = self.algorithm_combo.currentText()
        peaks = None
        properties = {}
        
        try:
            if algorithm == "Wavelet Transform":
                # Parse scales input (e.g., "1-32")
                scales_text = self.scales_input.text()
                if '-' in scales_text:
                    try:
                        scale_min, scale_max = map(int, scales_text.split('-'))
                        scales = range(scale_min, scale_max + 1)
                    except ValueError:
                        scales = range(1, 33)  # Default range
                else:
                    scales = range(1, 33)  # Default range
                
                wavelet = self.wavelet_combo.currentText()
                threshold = self.wavelet_threshold_input.value()
                
                # Detect peaks using wavelet transform
                peaks = self.detect_peaks_wavelet(self.current_data, scales, wavelet, threshold)
                
                # For wavelet method, we need to compute properties using find_peaks
                if len(peaks) > 0:
                    prominence = self.prominence_input.value()
                    wlen = self.wlen_input.value()
                    peaks, properties = find_peaks(-self.current_data, prominence=prominence, wlen=wlen, distance=10)
                    # Filter to only include peaks found by wavelet method
                    wavelet_peaks_set = set(peaks)
                    peaks = [p for p in peaks if p in wavelet_peaks_set]
                    # Update properties to match filtered peaks
                    if 'prominences' in properties:
                        properties['prominences'] = [prop for i, prop in enumerate(properties['prominences']) if i in wavelet_peaks_set]
                    if 'left_bases' in properties:
                        properties['left_bases'] = [prop for i, prop in enumerate(properties['left_bases']) if i in wavelet_peaks_set]
                    if 'right_bases' in properties:
                        properties['right_bases'] = [prop for i, prop in enumerate(properties['right_bases']) if i in wavelet_peaks_set]
                
            elif algorithm == "Machine Learning":
                model_path = self.ml_model_input.text().strip()
                if not model_path:
                    model_path = None
                
                # Detect peaks using machine learning
                peaks = self.detect_peaks_ml(self.current_data, model_path)
                
                # For ML method, we need to compute properties using find_peaks
                if len(peaks) > 0:
                    prominence = self.prominence_input.value()
                    wlen = self.wlen_input.value()
                    peaks, properties = find_peaks(-self.current_data, prominence=prominence, wlen=wlen, distance=10)
                    # Filter to only include peaks found by ML method
                    ml_peaks_set = set(peaks)
                    peaks = [p for p in peaks if p in ml_peaks_set]
                    # Update properties to match filtered peaks
                    if 'prominences' in properties:
                        properties['prominences'] = [prop for i, prop in enumerate(properties['prominences']) if i in ml_peaks_set]
                    if 'left_bases' in properties:
                        properties['left_bases'] = [prop for i, prop in enumerate(properties['left_bases']) if i in ml_peaks_set]
                    if 'right_bases' in properties:
                        properties['right_bases'] = [prop for i, prop in enumerate(properties['right_bases']) if i in ml_peaks_set]
                
            else:
                # Default Scipy find_peaks method
                prominence = self.prominence_input.value()
                wlen = self.wlen_input.value()
                peaks, properties = find_peaks(-self.current_data,
                                             prominence=prominence,
                                             wlen=wlen,
                                             distance=10)
        except ImportError as e:
            QMessageBox.warning(self, "依赖错误", f"{str(e)}\n请安装所需依赖。")
            return
        except Exception as e:
            QMessageBox.warning(self, "峰值检测错误", f"峰值检测失败: {str(e)}")
            return
        threshold_line = pg.InfiniteLine(pos=actual_threshold, angle=0, pen=pg.mkPen(color=(255, 0, 0), style=Qt.DashLine))
        self.plot_widget.addItem(threshold_line)
    
    def analyze_data(self):
        if self.current_data is None or self.time_data is None:
            return
        
        # 计算基线值（前200个点的平均值）
        baseline = self.calculate_baseline()
        
        # 用户输入的是绝对阈值，直接使用
        absolute_threshold = self.threshold_input.value()
        actual_threshold = absolute_threshold
        
        prominence = self.prominence_input.value()
        
        # Find data points below actual threshold
        below_threshold_mask = self.current_data < actual_threshold
        below_threshold_indices = np.where(below_threshold_mask)[0]
        
        if len(below_threshold_indices) == 0:
            return
        
        # Extract data below threshold
        below_threshold_current = self.current_data[below_threshold_indices]
        below_threshold_time = self.time_data[below_threshold_indices]
        
        # Use find_peaks with prominence and width for better peak control
        # prominence控制峰的显著性，width计算半高宽
        wlen = self.wlen_input.value()
        peaks, properties = find_peaks(-self.current_data, 
                                     prominence=prominence,
                                     width=1,  # 最小宽度（采样点数）
                                     rel_height=0.5,  # 半高宽
                                     wlen=wlen,
                                     distance=10)
        
        # Filter peaks to only include those below threshold and save corresponding properties
        threshold_peaks = []
        threshold_prominences = []
        threshold_left_bases = []
        threshold_right_bases = []
        threshold_widths = []
        threshold_left_ips = []
        threshold_right_ips = []
        threshold_width_heights = []
        
        for i, peak in enumerate(peaks):
            if self.current_data[peak] < actual_threshold:
                threshold_peaks.append(peak)
                
                # 获取prominence值
                if 'prominences' in properties and i < len(properties['prominences']):
                    threshold_prominences.append(properties['prominences'][i])
                else:
                    threshold_prominences.append(0.0)
                
                # 获取left_bases和right_bases
                if 'left_bases' in properties and i < len(properties['left_bases']):
                    threshold_left_bases.append(properties['left_bases'][i])
                else:
                    threshold_left_bases.append(peak)
                
                if 'right_bases' in properties and i < len(properties['right_bases']):
                    threshold_right_bases.append(properties['right_bases'][i])
                else:
                    threshold_right_bases.append(peak)
                
                # 获取宽度相关属性
                if 'widths' in properties and i < len(properties['widths']):
                    threshold_widths.append(properties['widths'][i])
                else:
                    threshold_widths.append(0.0)
                
                if 'left_ips' in properties and i < len(properties['left_ips']):
                    threshold_left_ips.append(properties['left_ips'][i])
                else:
                    threshold_left_ips.append(peak)
                
                if 'right_ips' in properties and i < len(properties['right_ips']):
                    threshold_right_ips.append(properties['right_ips'][i])
                else:
                    threshold_right_ips.append(peak)
                
                if 'width_heights' in properties and i < len(properties['width_heights']):
                    threshold_width_heights.append(properties['width_heights'][i])
                else:
                    threshold_width_heights.append(self.current_data[peak] / 2)
        
        peaks = np.array(threshold_peaks)
        prominences = np.array(threshold_prominences)
        left_bases = np.array(threshold_left_bases)
        right_bases = np.array(threshold_right_bases)
        
        # 计算峰值宽度（半高全宽FWHM）
        if len(peaks) > 0:
            # 使用过滤后的宽度数据，转换为微秒
            peak_widths_us = np.array(threshold_widths) / self.sample_rate * 1000000
            left_ips_times = np.array(threshold_left_ips) / self.sample_rate
            right_ips_times = np.array(threshold_right_ips) / self.sample_rate
            # 将width_heights转换为实际电流值（注意：find_peaks使用负信号，所以需要取负值）
            fwhm_heights = -np.array(threshold_width_heights)
        else:
            peak_widths_us = np.array([])
            left_ips_times = np.array([])
            right_ips_times = np.array([])
            fwhm_heights = np.array([])
        
        # 调试信息：打印prominence值和bases
        print(f"Debug: Found {len(peaks)} peaks")
        print(f"Debug: Prominences: {prominences}")
        print(f"Debug: Left bases: {left_bases}")
        print(f"Debug: Right bases: {right_bases}")
        print(f"Debug: Peak widths (μs): {peak_widths_us}")
        
        if len(peaks) == 0:
            return
        
        # Calculate total time below threshold and identify event segments
        total_time = 0
        event_segments = []  # 存储事件段信息
        
        if len(below_threshold_indices) > 1:
            segments = []
            start = 0
            for i in range(1, len(below_threshold_indices)):
                if below_threshold_indices[i] - below_threshold_indices[i-1] > 1:
                    segments.append((start, i-1))
                    start = i
            segments.append((start, len(below_threshold_indices)-1))
            
            for seg_start, seg_end in segments:
                if seg_end > seg_start:
                    time_diff = below_threshold_time[seg_end] - below_threshold_time[seg_start]
                    event_start_time = below_threshold_time[seg_start]  # 事件开始时间 
                    event_end_time = below_threshold_time[seg_end]  # 事件结束时间
                    
                    # 存储事件段信息
                    event_segments.append({
                        'start_idx': below_threshold_indices[seg_start],
                        'end_idx': below_threshold_indices[seg_end],
                        'start_time': event_start_time,
                        'end_time': event_end_time,
                        'duration': time_diff
                    })
                    
                    total_time += time_diff
        
        # Get peak information using original indices
        peak_times = self.time_data[peaks]
        peak_currents = self.current_data[peaks]
        
        # Find start points for each peak using the method from peak_detection.py
        peak_start_times = []
        peak_start_currents = []
        peak_amplitudes = []
        
        for i, peak_idx in enumerate(peaks):
            # 从峰值向前寻找起点（电流开始下降的地方）
            start_idx = peak_idx
            for j in range(peak_idx, 0, -1):
                if j == 0:
                    start_idx = 0
                    break
                # 寻找电流开始显著下降的点（上升趋势的结束点）
                if self.current_data[j] > self.current_data[j-1]:  # 找到上升趋势的结束点
                    start_idx = j
                    break
                start_idx = j
            
            start_time = self.time_data[start_idx]
            start_current = self.current_data[start_idx]
            
            peak_start_times.append(start_time)
            peak_start_currents.append(start_current)
            
            # 直接使用find_peaks返回的prominence值作为peak_amplitude
            amplitude = prominences[i] if i < len(prominences) else 0.0
            peak_amplitudes.append(amplitude)
        
        # Store results
        current_file = os.path.basename(self.file_list[self.current_file_index]) if self.file_list else "Unknown"
        
        # Function to find event start time for each peak
        def find_peak_event_start_time(peak_idx, event_segments):
            """为每个峰值找到对应的事件开始时间"""
            for segment in event_segments:
                if segment['start_idx'] <= peak_idx <= segment['end_idx']:
                    return segment['start_time']
            # 如果没找到对应的segment，返回第一个segment的开始时间或0
            return event_segments[0]['start_time'] if event_segments else 0
        
        # Function to find event duration for each peak
        def find_peak_event_duration(peak_idx, event_segments):
            """为每个峰值找到对应的事件持续时间"""
            for segment in event_segments:
                if segment['start_idx'] <= peak_idx <= segment['end_idx']:
                    return segment['duration']
            # 如果没找到对应的segment，返回0
            return 0
        
        # Function to calculate peak relative time within event
        def calculate_peak_relative_time(peak_idx, event_segments):
            """计算峰值相对于事件起始点的时间"""
            for segment in event_segments:
                if segment['start_idx'] <= peak_idx <= segment['end_idx']:
                    # 计算相对索引差
                    relative_idx = peak_idx - segment['start_idx']
                    # 转换为时间差（假设等间隔采样）
                    if len(self.time_data) > 1:
                        sampling_interval = self.time_data[1] - self.time_data[0]
                        return relative_idx * sampling_interval
                    else:
                        return 0.0
            # 如果没找到对应的segment，返回0
            return 0.0
        
        self.peaks_data = []
        for i in range(len(peaks)):
            # Find the event start time and duration for this peak
            peak_event_start_time = find_peak_event_start_time(peaks[i], event_segments)
            peak_event_duration = find_peak_event_duration(peaks[i], event_segments)
            
            # Calculate peak time relative to event start
            peak_relative_time = calculate_peak_relative_time(peaks[i], event_segments)
            
            # Calculate relative parameters
            peak_rel_t = peak_relative_time / peak_event_duration if peak_event_duration > 0 else 0
            peak_rel_i = -peak_amplitudes[i] / peak_start_currents[i] if peak_start_currents[i] != 0 else 0
            
            self.peaks_data.append({
                'file_name': current_file,
                'peak_number': i + 1,
                'duration': peak_event_duration,
                'peak_t': peak_relative_time,
                'peak_i': peak_currents[i],
                'peak_start_i': peak_start_currents[i],
                'peak_start_t': peak_start_times[i],
                'peak_amplitude': peak_amplitudes[i],  # 现在使用prominence值
                'peak_width_us': peak_widths_us[i] if i < len(peak_widths_us) else 0.0,  # 峰值宽度（微秒）
                'fwhm_left_time': left_ips_times[i] if i < len(left_ips_times) else None,  # 半高宽左边界时间
                'fwhm_right_time': right_ips_times[i] if i < len(right_ips_times) else None,  # 半高宽右边界时间
                'fwhm_height': fwhm_heights[i] if i < len(fwhm_heights) else None,  # 半高宽对应的高度
                'peak_rel_t': peak_rel_t,
                'peak_rel_i': peak_rel_i,
                'left_base_t': self.time_data[left_bases[i]] if i < len(left_bases) else None,
                'left_base_i': self.current_data[left_bases[i]] if i < len(left_bases) else None,
                'right_base_t': self.time_data[right_bases[i]] if i < len(right_bases) else None,
                'right_base_i': self.current_data[right_bases[i]] if i < len(right_bases) else None
            })
        
        
        # Populate peak table for review
        self.peak_table_widget.setRowCount(len(self.peaks_data))
        for i, peak_data in enumerate(self.peaks_data):
            # 选择列（复选框）
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.Unchecked)
            self.peak_table_widget.setItem(i, 0, checkbox_item)
            
            # 峰值编号
            peak_num_item = QTableWidgetItem(f"峰值 {i+1}")
            peak_num_item.setFlags(Qt.ItemIsEnabled)
            self.peak_table_widget.setItem(i, 1, peak_num_item)
            
            # 审核状态 (下拉框)
            status_item = QTableWidgetItem("待审核")
            status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable)
            self.peak_table_widget.setItem(i, 2, status_item)
            
            # Position Label (可编辑)
            label_item = QTableWidgetItem("")
            label_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable)
            self.peak_table_widget.setItem(i, 3, label_item)
            
            # 时间
            time_item = QTableWidgetItem(f"{peak_data['peak_t']:.6f}")
            time_item.setFlags(Qt.ItemIsEnabled)
            self.peak_table_widget.setItem(i, 4, time_item)
            
            # Prominence
            prominence_item = QTableWidgetItem(f"{peak_data['peak_amplitude']:.6f}")
            prominence_item.setFlags(Qt.ItemIsEnabled)
            self.peak_table_widget.setItem(i, 5, prominence_item)
            
            # 宽度 (μs)
            width_item = QTableWidgetItem(f"{peak_data['peak_width_us']:.2f}")
            width_item.setFlags(Qt.ItemIsEnabled)
            self.peak_table_widget.setItem(i, 6, width_item)
        
        # Plot peaks on the graph
        self.plot_data()
        
        # 不再高亮显示低于阈值的数据点
        # Add peak markers
        if len(peak_times) > 0:
            self.plot_widget.plot(peak_times, peak_currents, 
                                pen=None, symbol='o', symbolBrush=(255, 0, 0), symbolSize=8, name='峰值')
            
            # 绘制prominence竖线：从峰值点向上延伸prominence值的长度
            self.draw_prominence_lines(peak_times, peak_currents, prominences)
            
            # Add prominence value labels on the plot
            for i, (t, curr, prominence_val) in enumerate(zip(peak_times, peak_currents, prominences)):
                # Create text item for prominence values
                text_item = pg.TextItem(f'P{i+1}', 
                                      color=(255, 0, 0), 
                                      anchor=(0.5, 1.5))
                text_item.setPos(t, curr)
                self.plot_widget.addItem(text_item)
            
            # 绘制半高宽标记线
            self.draw_fwhm_lines(peak_times, peak_currents, peak_widths_us)
        
        # Add left_bases and right_bases markers
        if len(peaks) > 0:
            # 获取left_bases和right_bases对应的时间和电流值
            left_base_times = self.time_data[left_bases]
            left_base_currents = self.current_data[left_bases]
            right_base_times = self.time_data[right_bases]
            right_base_currents = self.current_data[right_bases]
            
            # 绘制left_bases（左基点）- 使用蓝色三角形
            self.plot_widget.plot(left_base_times, left_base_currents, 
                                pen=None, symbol='t1', symbolBrush=(0, 0, 255), symbolSize=10, name='左基点')
            
            # 绘制right_bases（右基点）- 使用紫色三角形
            self.plot_widget.plot(right_base_times, right_base_currents, 
                                pen=None, symbol='t2', symbolBrush=(128, 0, 128), symbolSize=10, name='右基点')
            
            # 添加连接线：从左基点到右基点
            for i in range(len(peaks)):
                # 绘制基线（从左基点到右基点的水平线）
                base_y = min(left_base_currents[i], right_base_currents[i])  # 使用较低的点作为基线
                self.plot_widget.plot([left_base_times[i], right_base_times[i]], 
                                    [base_y, base_y], 
                                    pen=pg.mkPen(color=(128, 128, 128), width=1, style=Qt.DashLine))
        
        # Enable review and export buttons
        self.select_all_btn.setEnabled(True)
        self.select_none_btn.setEnabled(True)
        self.mark_keep_btn.setEnabled(True)
        self.mark_delete_btn.setEnabled(True)
        self.clear_status_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.edit_labels_btn.setEnabled(True)
    
    def select_all_peaks(self):
        """选择所有峰值"""
        for i in range(self.peak_table_widget.rowCount()):
            item = self.peak_table_widget.item(i, 0)  # 选择列
            if item:
                item.setCheckState(Qt.Checked)
    
    def select_none_peaks(self):
        """取消选择所有峰值"""
        for i in range(self.peak_table_widget.rowCount()):
            item = self.peak_table_widget.item(i, 0)  # 选择列
            if item:
                item.setCheckState(Qt.Unchecked)
    
    def mark_selected_as_keep(self):
        """将选中的峰值标记为保留"""
        for i in range(self.peak_table_widget.rowCount()):
            checkbox_item = self.peak_table_widget.item(i, 0)  # 选择列
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                status_item = self.peak_table_widget.item(i, 2)  # 审核状态列
                if status_item:
                    status_item.setText("保留")
    
    def mark_selected_as_delete(self):
        """将选中的峰值标记为删除"""
        for i in range(self.peak_table_widget.rowCount()):
            checkbox_item = self.peak_table_widget.item(i, 0)  # 选择列
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                status_item = self.peak_table_widget.item(i, 2)  # 审核状态列
                if status_item:
                    status_item.setText("删除")
    
    def clear_selected_status(self):
        """清除选中峰值的审核状态"""
        for i in range(self.peak_table_widget.rowCount()):
            checkbox_item = self.peak_table_widget.item(i, 0)  # 选择列
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                status_item = self.peak_table_widget.item(i, 2)  # 审核状态列
                if status_item:
                    status_item.setText("待审核")
    
    def on_peak_table_cell_clicked(self, row, column):
        """处理峰值表格单元格点击事件"""
        # 如果点击的是Position Label列（第3列，索引为3）
        if column == 3:
            # 不进行任何自动重命名操作
            # 用户需要手动编辑position label
            pass
    
    def get_selected_peaks(self):
        """获取标记为保留的峰值数据"""
        selected_peaks = []
        for i in range(self.peak_table_widget.rowCount()):
            status_item = self.peak_table_widget.item(i, 2)  # 审核状态列
            if status_item and status_item.text() == "保留":
                # 获取position label
                label_item = self.peak_table_widget.item(i, 3)  # Position Label列
                position_label = label_item.text() if label_item else ""
                
                # 创建包含position label的峰值数据
                peak_data = self.peaks_data[i].copy()
                peak_data['position_label'] = position_label
                selected_peaks.append(peak_data)
        return selected_peaks
    
    def edit_position_labels(self):
        """编辑Position Labels对话框"""
        if not hasattr(self, 'peaks_data') or not self.peaks_data:
            QMessageBox.warning(self, "警告", "没有分析结果可编辑。")
            return
        
        # 只获取标记为"保留"的峰值
        kept_peaks = []
        kept_indices = []
        for i in range(self.peak_table_widget.rowCount()):
            status_item = self.peak_table_widget.item(i, 2)  # 审核状态列
            if status_item and status_item.text() == "保留":
                kept_peaks.append(self.peaks_data[i])
                kept_indices.append(i)
        
        if not kept_peaks:
            QMessageBox.warning(self, "警告", "没有标记为保留的峰值可编辑。")
            return
        
        dialog = PositionLabelDialog(kept_peaks, self)
        if dialog.exec_() == QDialog.Accepted:
            # 更新表格中的position labels
            labels = dialog.get_labels()
            for i, label in enumerate(labels):
                if i < len(kept_indices):
                    row_index = kept_indices[i]
                    label_item = self.peak_table_widget.item(row_index, 3)  # Position Label列
                    if label_item:
                        label_item.setText(label)
            
            # 标记已编辑过labels
            self.labels_edited = True
    
    def export_to_csv(self):
        if not self.peaks_data:
            QMessageBox.warning(self, "警告", "没有分析结果可导出。")
            return
        
        # Get selected peaks (only those marked as "保留")
        selected_peaks = self.get_selected_peaks()
        if not selected_peaks:
            QMessageBox.warning(self, "警告", "没有标记为保留的峰值可导出。")
            return
        
        # Create CSV filename based on folder name
        folder_name = os.path.basename(self.current_folder) if self.current_folder else "analysis"
        csv_filename = f"{folder_name}_analysis_results.csv"
        csv_path = os.path.join(self.current_folder, csv_filename)
        
        # Check if file exists
        file_exists = os.path.exists(csv_path)
        
        # 定义完整的字段名列表（包含所有FWHM相关字段）
        fieldnames = ['file_name', 'peak_number', 'duration', 'peak_t', 'peak_i', 
                     'peak_start_i', 'peak_start_t', 'peak_amplitude', 'peak_width_us', 
                     'fwhm_left_time', 'fwhm_right_time', 'fwhm_height', 'peak_rel_t', 'peak_rel_i',
                     'left_base_t', 'left_base_i', 'right_base_t', 'right_base_i']
        
        # 如果编辑过labels，添加position_label字段
        if self.labels_edited:
            fieldnames.append('position_label')
        
        # 检查现有文件的字段格式是否匹配
        need_new_header = True
        if file_exists:
            try:
                # 读取现有文件的第一行（标题行）
                with open(csv_path, 'r', encoding='utf-8') as csvfile:
                    first_line = csvfile.readline().strip()
                    existing_fields = [field.strip() for field in first_line.split(',')]
                    
                    # 检查字段是否匹配
                    if existing_fields == fieldnames:
                        need_new_header = False
                        print(f"Debug: CSV文件字段匹配，将追加数据")
                    else:
                        print(f"Debug: CSV文件字段不匹配")
                        print(f"Debug: 现有字段: {existing_fields}")
                        print(f"Debug: 期望字段: {fieldnames}")
                        print(f"Debug: 缺少的FWHM字段: {set(fieldnames) - set(existing_fields)}")
            except:
                # 如果读取失败，创建新文件
                need_new_header = True
        
        try:
            # 根据字段匹配情况决定写入模式
            if need_new_header or not file_exists:
                # 创建新文件或覆盖不兼容的文件
                if file_exists:
                    # 备份原文件
                    import shutil
                    backup_path = csv_path + ".backup"
                    shutil.copy2(csv_path, backup_path)
                    print(f"Debug: 原文件已备份到 {backup_path}")
                
                mode = 'w'  # 覆盖模式
                write_header = True
            else:
                # 追加到兼容的现有文件
                mode = 'a'  # 追加模式
                write_header = False
            
            with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入标题行（如果需要）
                if write_header:
                    writer.writeheader()
                
                # 写入选中的峰值数据
                for peak_data in selected_peaks:
                    # 创建数据副本以避免修改原始数据
                    data_to_write = peak_data.copy()
                    
                    # 如果没有编辑过labels，移除position_label字段
                    if not self.labels_edited and 'position_label' in data_to_write:
                        del data_to_write['position_label']
                    
                    # 确保所有FWHM字段都存在，如果不存在则设置为None
                    for field in fieldnames:
                        if field not in data_to_write:
                            data_to_write[field] = None
                    
                    writer.writerow(data_to_write)
            
            # 准备成功消息
            message = f"成功导出 {len(selected_peaks)} 个保留的峰值到 {csv_path}"
            if mode == 'a':
                message += "\n(数据已追加到现有文件)"
            elif file_exists:
                message += "\n(由于字段格式不匹配，已创建新文件并备份原文件)"
            else:
                message += "\n(创建新文件)"
                
            if self.labels_edited:
                message += "\n(包含Position Labels)"
            else:
                message += "\n(未包含Position Labels)"
                
            # 添加FWHM字段确认信息
            fwhm_fields = ['peak_width_us', 'fwhm_left_time', 'fwhm_right_time', 'fwhm_height']
            message += f"\n(包含FWHM相关字段: {', '.join(fwhm_fields)})"
            
            QMessageBox.information(self, "导出成功", message)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出CSV失败: {str(e)}")

class MultiFileVisualizer(QWidget):
    """多文件数据可视化界面"""
    def __init__(self):
        super().__init__()
        self.csv_files = {}  # {filename: DataFrame}
        
        # Plot history for cumulative plotting
        self.plot_history = []  # List of plot data for cumulative display
        self.current_color_index = 0  # Track color cycling
        
        self.initUI()
        
    def initUI(self):
        # Main layout - three panels: left, center, right
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel for file and column management
        left_panel = QWidget()
        left_panel.setFixedWidth(280)  # Reduced width
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        left_panel.setLayout(left_layout)
        
        # File management section
        file_group = QGroupBox("文件管理")
        file_layout = QVBoxLayout()
        
        # Import and clear buttons
        file_btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("导入CSV文件")
        self.import_btn.clicked.connect(self.import_csv_files)
        self.clear_data_btn = QPushButton("Clear Data")
        self.clear_data_btn.clicked.connect(self.clear_all_data)
        self.clear_data_btn.setEnabled(False)
        file_btn_layout.addWidget(self.import_btn)
        file_btn_layout.addWidget(self.clear_data_btn)
        file_layout.addLayout(file_btn_layout)
        
        # File list (clickable, no checkboxes)
        file_layout.addWidget(QLabel("已导入文件:"))
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(80)  # Reduced from 100
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list_widget)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Column selection section
        col_group = QGroupBox("列选择")
        col_layout = QVBoxLayout()
        
        # Search box for columns
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索列:"))
        self.column_search_input = QLineEdit()
        self.column_search_input.setPlaceholderText("输入列名搜索...")
        self.column_search_input.textChanged.connect(self.filter_columns)
        search_layout.addWidget(self.column_search_input)
        col_layout.addLayout(search_layout)
        
        # Available columns from selected file
        col_layout.addWidget(QLabel("文件列 (双击添加):"))
        self.available_columns_list = QListWidget()
        self.available_columns_list.setMaximumHeight(70)  # Reduced from 100
        self.available_columns_list.itemDoubleClicked.connect(self.add_column_to_selected)
        col_layout.addWidget(self.available_columns_list)
        
        # Store all columns for filtering
        self.all_available_columns = []
        
        col_group.setLayout(col_layout)
        left_layout.addWidget(col_group)
        
        # Selected columns section
        selected_group = QGroupBox("选中列")
        selected_layout = QVBoxLayout()
        
        self.selected_columns_list = QListWidget()
        self.selected_columns_list.setMaximumHeight(80)  # Reduced from 120
        selected_layout.addWidget(self.selected_columns_list)
        
        # Remove selected column button
        self.remove_column_btn = QPushButton("移除选中列")
        self.remove_column_btn.clicked.connect(self.remove_selected_column)
        self.remove_column_btn.setEnabled(False)
        selected_layout.addWidget(self.remove_column_btn)
        
        selected_group.setLayout(selected_layout)
        left_layout.addWidget(selected_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # Center panel for pyqtgraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Value', '')
        self.plot_widget.setLabel('bottom', 'Categories/X-axis', '')
        self.plot_widget.showGrid(True, True)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Set white background
        self.plot_widget.setBackground('w')
        
        # Add crosshair for interactive coordinates (use darker color for white background)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(255, 0, 0), width=1))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(255, 0, 0), width=1))
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Connect mouse move event for coordinate display
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Add coordinate label
        self.coord_label = QLabel("Position: (0, 0)")
        
        # Create plot panel with coordinate display
        plot_panel = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addWidget(self.coord_label)
        plot_panel.setLayout(plot_layout)
        
        main_layout.addWidget(plot_panel, stretch=1)
        
        # Right panel for chart controls
        right_panel = QWidget()
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_panel.setLayout(right_layout)
        
        # Chart control section
        chart_group = QGroupBox("图表控制")
        chart_layout = QVBoxLayout()
        
        # Column mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("分析模式:"))
        self.single_col_radio = QRadioButton("单列分析")
        self.dual_col_radio = QRadioButton("双列分析")
        self.single_col_radio.setChecked(True)
        self.single_col_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.single_col_radio)
        mode_layout.addWidget(self.dual_col_radio)
        chart_layout.addLayout(mode_layout)
        
        # Plot type selection
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("图表类型:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Box Plot", "Histogram", "Violin Plot"])  # Default single column types
        plot_layout.addWidget(self.plot_type_combo)
        chart_layout.addLayout(plot_layout)
        
        # Single column selection for single mode - with checkboxes
        self.single_column_group = QGroupBox("单列选择")
        single_col_layout = QVBoxLayout()
        
        single_col_layout.addWidget(QLabel("选择要绘制的列 (可多选):"))
        
        # Scrollable area for checkboxes
        scroll_widget = QWidget()
        self.single_column_layout = QVBoxLayout(scroll_widget)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(120)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        single_col_layout.addWidget(scroll_area)
        
        # Buttons for select all/none
        checkbox_btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_single_columns)
        self.select_none_btn = QPushButton("全不选")
        self.select_none_btn.clicked.connect(self.select_none_single_columns)
        checkbox_btn_layout.addWidget(self.select_all_btn)
        checkbox_btn_layout.addWidget(self.select_none_btn)
        single_col_layout.addLayout(checkbox_btn_layout)
        
        self.single_column_group.setLayout(single_col_layout)
        chart_layout.addWidget(self.single_column_group)
        
        # Store checkbox references
        self.single_column_checkboxes = []
        
        # Axis selection for dual column mode
        self.axis_group = QGroupBox("坐标轴设置")
        axis_layout = QVBoxLayout()
        
        # X-axis selection
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X轴:"))
        self.x_axis_combo = QComboBox()
        x_layout.addWidget(self.x_axis_combo)
        axis_layout.addLayout(x_layout)
        
        # Y-axis selection
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y轴:"))
        self.y_axis_combo = QComboBox()
        y_layout.addWidget(self.y_axis_combo)
        axis_layout.addLayout(y_layout)
        
        self.axis_group.setLayout(axis_layout)
        chart_layout.addWidget(self.axis_group)
        
        # Plot control buttons
        plot_btn_layout = QHBoxLayout()
        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self.update_plot)
        self.update_plot_btn.setEnabled(False)
        self.clear_plot_btn = QPushButton("Clear Plot")
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        self.clear_plot_btn.setEnabled(False)
        plot_btn_layout.addWidget(self.update_plot_btn)
        plot_btn_layout.addWidget(self.clear_plot_btn)
        chart_layout.addLayout(plot_btn_layout)
        
        # Bins and Color scheme on same row
        bins_color_layout = QHBoxLayout()
        bins_color_layout.addWidget(QLabel("Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(10, 100)
        self.bins_spinbox.setValue(30)
        self.bins_spinbox.setMaximumWidth(60)
        bins_color_layout.addWidget(self.bins_spinbox)
        
        bins_color_layout.addWidget(QLabel("颜色:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Set1", "Set2", "Dark2", "Pastel1", "husl"])
        bins_color_layout.addWidget(self.color_combo)
        chart_layout.addLayout(bins_color_layout)
        
        chart_group.setLayout(chart_layout)
        right_layout.addWidget(chart_group)
        
        # Export section - more compact
        export_group = QGroupBox("导出选项")
        export_layout = QVBoxLayout()
        
        # First row: Data and Stats
        export_row1 = QHBoxLayout()
        self.export_data_btn = QPushButton("导出数据")
        self.export_data_btn.clicked.connect(self.export_current_data)
        self.export_data_btn.setEnabled(False)
        export_row1.addWidget(self.export_data_btn)
        
        self.export_stats_btn = QPushButton("导出统计")
        self.export_stats_btn.clicked.connect(self.export_statistics)
        self.export_stats_btn.setEnabled(False)
        export_row1.addWidget(self.export_stats_btn)
        export_layout.addLayout(export_row1)
        
        # Second row: Image
        self.export_image_btn = QPushButton("导出图像")
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_image_btn.setEnabled(False)
        export_layout.addWidget(self.export_image_btn)
        
        export_group.setLayout(export_layout)
        right_layout.addWidget(export_group)
        
        right_layout.addStretch()
        main_layout.addWidget(right_panel)
        
        # Initialize UI state - single column mode is default
        # Use QTimer to ensure widgets are fully initialized
        QTimer.singleShot(0, self.initialize_ui_state)
    
    def mouse_moved(self, pos):
        """处理鼠标移动事件，显示坐标"""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self.coord_label.setText(f"Position: ({x:.6f}, {y:.6f})")
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
    
    def initialize_ui_state(self):
        """初始化UI状态"""
        # Force the initial mode to be properly set
        if self.single_col_radio.isChecked():
            self.single_column_group.setVisible(True)
            self.axis_group.setVisible(False)
        else:
            self.single_column_group.setVisible(False)
            self.axis_group.setVisible(True)
        
    def import_csv_files(self):
        """导入CSV文件（支持增量导入，自动去重）"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择CSV文件", "", "CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_paths:
            return
        
        new_files_count = 0
        duplicate_files = []
        
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                
                # Check for duplicates
                if filename in self.csv_files:
                    duplicate_files.append(filename)
                    continue
                
                df = pd.read_csv(file_path)
                self.csv_files[filename] = df
                
                # Add to file list (no checkboxes, just clickable)
                item = QListWidgetItem(filename)
                self.file_list_widget.addItem(item)
                new_files_count += 1
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载文件 {file_path} 失败: {str(e)}")
        
        # Show import results (without popup)
        if new_files_count > 0:
            print(f"成功导入 {new_files_count} 个新文件")
            if duplicate_files:
                print(f"跳过 {len(duplicate_files)} 个重复文件: {', '.join(duplicate_files)}")
            
            self.clear_data_btn.setEnabled(True)
            self.enable_export_buttons(True)
        elif duplicate_files:
            print(f"所选文件已存在，未导入重复文件: {', '.join(duplicate_files)}")
    
    def clear_all_data(self):
        """清除所有数据"""
        # Remove confirmation dialog - direct clear
        # 清除所有数据
        self.csv_files.clear()
        self.file_list_widget.clear()
        self.available_columns_list.clear()
        self.selected_columns_list.clear()
        self.all_available_columns.clear()
        self.column_search_input.clear()
        
        # 清除单列选择器中的复选框
        for checkbox in self.single_column_checkboxes:
            checkbox.setParent(None)
            checkbox.deleteLater()
        self.single_column_checkboxes.clear()
        
        # 清除坐标轴选择器
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        # 清除图表
        self.clear_plot()
        
        # 额外重置视图范围
        self.plot_widget.autoRange()
        
        # 禁用相关按钮
        self.clear_data_btn.setEnabled(False)
        self.remove_column_btn.setEnabled(False)
        self.update_plot_btn.setEnabled(False)
        self.clear_plot_btn.setEnabled(False)
        self.enable_export_buttons(False)
        
        print("Debug: All data cleared successfully")
    
    def filter_columns(self):
        """根据搜索文本过滤列"""
        search_text = self.column_search_input.text().lower()
        
        # 清除当前显示的列
        self.available_columns_list.clear()
        
        # 如果搜索框为空，显示所有列
        if not search_text:
            for col in self.all_available_columns:
                self.available_columns_list.addItem(col)
        else:
            # 显示匹配的列
            for col in self.all_available_columns:
                if search_text in col.lower():
                    self.available_columns_list.addItem(col)
        
        print(f"Debug: Filtered columns, showing {self.available_columns_list.count()} out of {len(self.all_available_columns)} columns")
    
    def on_file_selected(self, item):
        """文件被选中时显示该文件的列"""
        filename = item.text()
        print(f"Debug: File selected: {filename}")
        
        if filename in self.csv_files:
            df = self.csv_files[filename]
            print(f"Debug: DataFrame shape: {df.shape}")
            print(f"Debug: DataFrame columns: {list(df.columns)}")
            
            # Show only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"Debug: Numeric columns: {numeric_cols}")
            
            # Store all columns for filtering
            self.all_available_columns = numeric_cols.copy()
            
            # Clear search box when switching files
            self.column_search_input.clear()
            
            # Display all columns initially
            self.available_columns_list.clear()
            for col in numeric_cols:
                self.available_columns_list.addItem(col)
                
            print(f"Debug: Added {len(numeric_cols)} columns to available list")
        else:
            print(f"Debug: File {filename} not found in csv_files")
    
    def add_column_to_selected(self, item):
        """添加列到选中列表"""
        # Get current selected file
        current_file_item = self.file_list_widget.currentItem()
        if not current_file_item:
            QMessageBox.warning(self, "警告", "请先选择一个文件")
            return
        
        filename = current_file_item.text()
        column_name = item.text()
        
        # Create unique identifier: filename.column_name
        unique_name = f"{filename}.{column_name}"
        
        print(f"Debug: Adding column {unique_name}")
        
        # Check if already exists
        for i in range(self.selected_columns_list.count()):
            if self.selected_columns_list.item(i).text() == unique_name:
                QMessageBox.information(self, "提示", "该列已在选中列表中")
                return
        
        # Add to selected columns
        self.selected_columns_list.addItem(unique_name)
        print(f"Debug: Successfully added {unique_name} to selected columns")
        print(f"Debug: Selected columns count: {self.selected_columns_list.count()}")
        
        # List all selected columns
        selected_items = []
        for i in range(self.selected_columns_list.count()):
            selected_items.append(self.selected_columns_list.item(i).text())
        print(f"Debug: All selected columns: {selected_items}")
        
        self.remove_column_btn.setEnabled(True)
        self.update_plot_btn.setEnabled(True)
        self.clear_plot_btn.setEnabled(True)
        
        # Update selectors
        self.update_axis_selectors()
        self.update_single_column_selector()
    
    def update_single_column_selector(self):
        """更新单列选择器 - 使用复选框"""
        # Clear current checkboxes
        for checkbox in self.single_column_checkboxes:
            checkbox.setParent(None)
            checkbox.deleteLater()
        self.single_column_checkboxes.clear()
        
        # Get all selected columns
        selected_items = []
        for i in range(self.selected_columns_list.count()):
            selected_items.append(self.selected_columns_list.item(i).text())
        
        # Create checkboxes for each column
        for item in selected_items:
            checkbox = QCheckBox(item)
            checkbox.setChecked(True)  # Default to checked
            self.single_column_layout.addWidget(checkbox)
            self.single_column_checkboxes.append(checkbox)
    
    def select_all_single_columns(self):
        """选择所有单列复选框"""
        for checkbox in self.single_column_checkboxes:
            checkbox.setChecked(True)
    
    def select_none_single_columns(self):
        """取消选择所有单列复选框"""
        for checkbox in self.single_column_checkboxes:
            checkbox.setChecked(False)
    
    def get_selected_single_columns(self):
        """获取选中的单列"""
        selected_columns = []
        for checkbox in self.single_column_checkboxes:
            if checkbox.isChecked():
                selected_columns.append(checkbox.text())
        return selected_columns
    
    def update_axis_selectors(self):
        """更新坐标轴选择器"""
        # Clear current items
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        # Get all selected columns
        selected_items = []
        for i in range(self.selected_columns_list.count()):
            selected_items.append(self.selected_columns_list.item(i).text())
        
        # Add to axis selectors
        for item in selected_items:
            self.x_axis_combo.addItem(item)
            self.y_axis_combo.addItem(item)
    
    def on_mode_changed(self):
        """分析模式改变时的处理"""
        print(f"Debug: Mode changed, single_col_radio checked: {self.single_col_radio.isChecked()}")
        
        if self.single_col_radio.isChecked():
            # Single column mode
            self.plot_type_combo.clear()
            self.plot_type_combo.addItems(["Box Plot", "Histogram", "Violin Plot"])
            self.single_column_group.setVisible(True)
            self.axis_group.setVisible(False)
            self.update_single_column_selector()
            print("Debug: Set to single column mode")
        else:
            # Dual column mode
            self.plot_type_combo.clear()
            self.plot_type_combo.addItems(["Scatter Plot", "KDE Plot"])
            self.single_column_group.setVisible(False)
            self.axis_group.setVisible(True)
            self.update_axis_selectors()
            print("Debug: Set to dual column mode")
            
        # Force repaint
        self.single_column_group.repaint()
        self.axis_group.repaint()
    
    def remove_selected_column(self):
        """移除选中的列"""
        current_item = self.selected_columns_list.currentItem()
        if current_item:
            row = self.selected_columns_list.row(current_item)
            self.selected_columns_list.takeItem(row)
            
            if self.selected_columns_list.count() == 0:
                self.remove_column_btn.setEnabled(False)
                self.update_plot_btn.setEnabled(False)
                self.clear_plot_btn.setEnabled(False)
            
            # Update selectors
            self.update_axis_selectors()
            self.update_single_column_selector()
    
    def clear_plot(self):
        """清除当前图表和累积数据"""
        self.plot_widget.clear()
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Reset axis labels to default
        self.plot_widget.setLabel('left', 'Value', '')
        self.plot_widget.setLabel('bottom', 'Categories/X-axis', '')
        
        # Clear x-axis ticks
        ax = self.plot_widget.getAxis('bottom')
        ax.setTicks(None)
        
        # Clear plot history and reset color index
        self.plot_history.clear()
        self.current_color_index = 0
        print("Debug: Plot history cleared")
    
    def update_plot(self):
        """更新图表 - 支持累积绘图"""
        print("Debug: update_plot called")
        print(f"Debug: CSV files count: {len(self.csv_files)}")
        print(f"Debug: CSV files: {list(self.csv_files.keys())}")
        print(f"Debug: Selected columns count: {self.selected_columns_list.count()}")
        
        if not self.csv_files:
            print("Debug: No CSV files loaded")
            QMessageBox.warning(self, "警告", "请先导入CSV文件")
            return
            
        if self.selected_columns_list.count() == 0:
            print("Debug: No columns selected")
            QMessageBox.warning(self, "警告", "请先选择要分析的列")
            return
        
        try:
            plot_type = self.plot_type_combo.currentText()
            
            # Prepare current plot data
            current_plot_data = {
                'mode': 'single' if self.single_col_radio.isChecked() else 'dual',
                'plot_type': plot_type,
                'color_index': self.current_color_index
            }
            
            if self.single_col_radio.isChecked():
                # Single column mode - use selected checkboxes
                selected_columns = self.get_selected_single_columns()
                
                if not selected_columns:
                    QMessageBox.warning(self, "警告", "请先选择要绘制的列")
                    return
                
                current_plot_data['columns'] = selected_columns
                
            else:
                # Dual column mode
                x_column = self.x_axis_combo.currentText()
                y_column = self.y_axis_combo.currentText()
                
                if not x_column or not y_column:
                    QMessageBox.warning(self, "警告", "请选择X轴和Y轴的列")
                    return
                
                if x_column == y_column:
                    QMessageBox.warning(self, "警告", "X轴和Y轴不能是同一列")
                    return
                
                current_plot_data['x_column'] = x_column
                current_plot_data['y_column'] = y_column
            
            # Add to plot history
            self.plot_history.append(current_plot_data)
            self.current_color_index += 1
            
            # Clear and redraw all plots
            self.redraw_cumulative_plots()
            
        except Exception as e:
            import traceback
            error_msg = f"绘图时出错: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            QMessageBox.warning(self, "绘图错误", error_msg)
    
    def redraw_cumulative_plots(self):
        """重绘所有累积图表"""
        self.plot_widget.clear()
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        if not self.plot_history:
            return
        
        # Get color palette - using built-in colors
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange', 'purple', 'brown']
        
        for i, plot_data in enumerate(self.plot_history):
            color = colors[i % len(colors)]
            
            if plot_data['mode'] == 'single':
                # Generate label from column filenames
                label = self.generate_label_from_columns(plot_data['columns'])
                self.create_single_column_plot_with_color_pg(
                    plot_data['columns'], plot_data['plot_type'], color, label
                )
            else:
                # Generate label from x and y column filenames
                label = self.generate_label_from_dual_columns(plot_data['x_column'], plot_data['y_column'])
                self.create_dual_column_plot_with_color_pg(
                    plot_data['x_column'], plot_data['y_column'], 
                    plot_data['plot_type'], color, label
                )
        
        # Enable export buttons after successful plot
        self.enable_export_buttons(True)
    
    def generate_label_from_columns(self, columns):
        """从列列表生成图例标签"""
        if not columns:
            return "Unknown"
        
        # For single column, show filename.column
        if len(columns) == 1:
            col_spec = columns[0]
            last_dot_idx = col_spec.rfind('.')
            if last_dot_idx != -1:
                filename = col_spec[:last_dot_idx]
                column = col_spec[last_dot_idx + 1:]
                return f"{filename.split('_')[0]}.{column}"
            else:
                return col_spec
        
        # For few columns, show column names
        if len(columns) <= 3:
            column_names = []
            for col_spec in columns:
                last_dot_idx = col_spec.rfind('.')
                if last_dot_idx != -1:
                    column = col_spec[last_dot_idx + 1:]
                    column_names.append(column)
                else:
                    column_names.append(col_spec)
            return " + ".join(column_names)
        
        # For multiple columns, extract unique filenames
        filenames = set()
        for col_spec in columns:
            last_dot_idx = col_spec.rfind('.')
            if last_dot_idx != -1:
                filename = col_spec[:last_dot_idx]
                filenames.add(filename.split('_')[0])
        
        if len(filenames) == 1:
            return list(filenames)[0]
        elif len(filenames) > 1:
            # Multiple files, show first few
            sorted_files = sorted(list(filenames))
            if len(sorted_files) <= 2:
                return " + ".join(sorted_files)
            else:
                return f"{sorted_files[0]} + {len(sorted_files)-1} others"
        else:
            return "Unknown"
    
    def generate_label_from_dual_columns(self, x_column, y_column):
        """从双列生成图例标签"""
        # Extract filenames
        x_last_dot = x_column.rfind('.')
        y_last_dot = y_column.rfind('.')
        
        if x_last_dot == -1 or y_last_dot == -1:
            return "Unknown"
        
        x_filename = x_column[:x_last_dot]
        y_filename = y_column[:y_last_dot]
        
        if x_filename == y_filename:
            return x_filename.split('_')[0]
        else:
            return f"{x_filename.split('_')[0]} vs {y_filename.split('_')[0]}"
    
    def create_single_column_plot(self, ax, selected_columns, plot_type):
        """创建统计图表"""
        print(f"Debug: Creating {plot_type} with columns: {selected_columns}")
        
        # Parse selected columns to get data
        data_dict = {}
        
        for col_spec in selected_columns:
            try:
                # Find the last dot to split filename and column
                # This handles filenames with dots (like "file.csv.column")
                last_dot_idx = col_spec.rfind('.')
                if last_dot_idx == -1:
                    print(f"Debug: Invalid column spec format: {col_spec}")
                    continue
                    
                filename = col_spec[:last_dot_idx]
                column = col_spec[last_dot_idx + 1:]
                print(f"Debug: Processing {filename}.{column}")
                print(f"Debug: Available files: {list(self.csv_files.keys())}")
                
                if filename in self.csv_files:
                    df = self.csv_files[filename]
                    print(f"Debug: Available columns in {filename}: {list(df.columns)}")
                    
                    if column in df.columns:
                        data = df[column].dropna()
                        print(f"Debug: Data length for {col_spec}: {len(data)}")
                        if len(data) > 0:
                            data_dict[col_spec] = data
                        else:
                            print(f"Debug: No valid data in column {column}")
                    else:
                        print(f"Debug: Column '{column}' not found in {filename}")
                else:
                    print(f"Debug: File '{filename}' not found in csv_files")
            except Exception as e:
                print(f"Debug: Error processing {col_spec}: {e}")
        
        print(f"Debug: Final data_dict keys: {list(data_dict.keys())}")
        
        if not data_dict:
            ax.text(0.5, 0.5, "没有可用数据", ha='center', va='center', transform=ax.transAxes)
            return
        
        color_palette = sns.color_palette(self.color_combo.currentText(), len(data_dict))
        
        if plot_type == "Box Plot":
            data_list = list(data_dict.values())
            labels = [f"{k.split('_')[0]}" for k in data_dict.keys()]
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('none')
            ax.set_title("Box Plot")
            # Rotate labels if they are long
            if any(len(label) > 10 for label in labels):
                ax.tick_params(axis='x', rotation=45)
            
        elif plot_type == "Histogram":
            for i, (label, data) in enumerate(data_dict.items()):
                ax.hist(data, bins=self.bins_spinbox.value(), alpha=0.7, 
                       color=color_palette[i], label=f"{label.split('_')[0]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram")
            ax.legend()
            
        elif plot_type == "Violin Plot":
            data_list = list(data_dict.values())
            labels = [f"{k.split('_')[0]}" for k in data_dict.keys()]
            parts = ax.violinplot(data_list, positions=range(1, len(data_list)+1))
            for pc in parts['bodies']:
                pc.set_facecolor('none')
            ax.set_xticks(range(1, len(data_list)+1))
            ax.set_xticklabels(labels)
            if any(len(label) > 10 for label in labels):
                ax.tick_params(axis='x', rotation=45)
            ax.set_title("Violin Plot")
        
        print(f"Debug: Plot creation completed for {plot_type}")
    
    def create_single_column_plot_with_color(self, ax, selected_columns, plot_type, color, label):
        """创建带指定颜色的单列图表"""
        print(f"Debug: Creating {plot_type} with color {color} and label {label}")
        
        # Parse selected columns to get data
        data_dict = {}
        
        for col_spec in selected_columns:
            print(f"Debug: Processing column: {col_spec}")
            
            # Handle filenames with dots properly
            last_dot_idx = col_spec.rfind('.')
            if last_dot_idx == -1:
                print(f"Debug: Invalid column spec format: {col_spec}")
                continue
            filename = col_spec[:last_dot_idx]
            column = col_spec[last_dot_idx + 1:]
            
            print(f"Debug: Filename: {filename}, Column: {column}")
            
            if filename in self.csv_files:
                df = self.csv_files[filename]
                print(f"Debug: Available columns in {filename}: {list(df.columns)}")
                if column in df.columns:
                    data = df[column].dropna()
                    data_dict[col_spec] = data
                    print(f"Debug: Added {len(data)} data points for {col_spec}")
                else:
                    print(f"Debug: Column {column} not found in {filename}")
            else:
                print(f"Debug: File {filename} not found")
        
        if not data_dict:
            print("Debug: No valid data found")
            return
        
        if plot_type == "Box Plot":
            data_list = list(data_dict.values())
            
            # 提取文件名中第一个下划线之前的部分作为标签
            labels = []
            for col_spec in data_dict.keys():
                filename = col_spec.split('.')[0]  # 获取文件名部分
                # 取第一个下划线之前的部分
                if '_' in filename:
                    label = filename.split('_')[0]
                else:
                    label = filename  # 如果没有下划线，使用完整文件名
                labels.append(label)
            
            # 创建箱线图，设置patch_artist=True以便自定义样式
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            
            # 让盒子变成透明的，只保留边框
            for patch in bp['boxes']:
                patch.set_facecolor('none')  # 设置填充色为透明
                patch.set_edgecolor('black')  # 保持边框为黑色
                patch.set_alpha(1.0)  # 设置透明度
            
            # 添加数据点，显示原始数据分布
            for i, data in enumerate(data_list):
                # 在x轴位置添加随机偏移，避免点重叠
                x_pos = i + 1
                jitter = np.random.normal(0, 0.05, len(data))  # 添加小的随机偏移
                ax.scatter(x_pos + jitter, data, alpha=0.6, color=color, s=20, label=label if i == 0 else "")
            
            # 设置x轴标签旋转，防止重叠
            if any(len(label) > 10 for label in labels):
                ax.tick_params(axis='x', rotation=45)
            
            ax.set_title("Box Plot with Data Points")
            ax.set_ylabel("Value")
            
        elif plot_type == "Histogram":
            for i, (col_label, data) in enumerate(data_dict.items()):
                alpha = 0.7 if len(data_dict) > 1 else 1.0
                ax.hist(data, bins=self.bins_spinbox.value(), alpha=alpha, 
                       color=color, label=f"{label} - {col_label}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            
        elif plot_type == "Violin Plot":
            data_list = list(data_dict.values())
            
            # 提取文件名中第一个下划线之前的部分作为标签
            labels = []
            for col_spec in data_dict.keys():
                filename = col_spec.split('.')[0]  # 获取文件名部分
                # 取第一个下划线之前的部分
                if '_' in filename:
                    label = filename.split('_')[0]
                else:
                    label = filename  # 如果没有下划线，使用完整文件名
                labels.append(label)
            
            parts = ax.violinplot(data_list, positions=range(1, len(data_list)+1))
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_label(label if pc == parts['bodies'][0] else "")
            ax.set_xticks(range(1, len(data_list)+1))
            ax.set_xticklabels(labels)
            if any(len(label) > 10 for label in labels):
                ax.tick_params(axis='x', rotation=45)
        
        print(f"Debug: Plot creation completed for {plot_type} with color")
    
    def create_single_column_plot_with_color_pg(self, selected_columns, plot_type, color, label):
        """使用 pyqtgraph 创建带指定颜色的单列图表"""
        print(f"Debug: Creating pyqtgraph {plot_type} with color {color} and label {label}")
        
        # Parse selected columns to get data
        data_dict = {}
        
        for col_spec in selected_columns:
            print(f"Debug: Processing column: {col_spec}")
            
            # Handle filenames with dots properly
            last_dot_idx = col_spec.rfind('.')
            if last_dot_idx == -1:
                print(f"Debug: Invalid column spec format: {col_spec}")
                continue
            filename = col_spec[:last_dot_idx]
            column = col_spec[last_dot_idx + 1:]
            
            if filename in self.csv_files:
                df = self.csv_files[filename]
                if column in df.columns:
                    data = df[column].dropna()
                    data_dict[col_spec] = data
                    print(f"Debug: Added {len(data)} data points for {col_spec}")
        
        if not data_dict:
            print("Debug: No valid data found")
            return
        
        if plot_type == "Box Plot":
            # 对于Box Plot，绘制完整的箱线图
            all_data = []
            x_positions = []
            categories = []
            
            # 提取文件名中的日期和样品名称作为标签
            for i, (col_spec, data) in enumerate(data_dict.items()):
                filename = col_spec.split('.')[0]
                
                # 解析文件名，提取日期和样品名称
                # 文件名格式为: 20250828_samplename_analysisxxx
                if '_' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        date_part = parts[0]  # 例如: 20250828
                        sample_part = parts[1]  # 例如: samplename
                        category = f"{date_part}\n{sample_part}"  # 使用换行符分隔
                    else:
                        category = parts[0]
                else:
                    category = filename
                categories.append(category)
                
                # 计算箱线图统计数据
                q1 = data.quantile(0.25)
                q2 = data.quantile(0.5)  # 中位数
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_whisker = max(data.min(), q1 - 1.5 * iqr)
                upper_whisker = min(data.max(), q3 + 1.5 * iqr)
                
                # 绘制箱体（透明填充）
                box_width = 0.3
                box_x = [i - box_width/2, i + box_width/2, i + box_width/2, i - box_width/2, i - box_width/2]
                box_y = [q1, q1, q3, q3, q1]
                
                # 创建箱体轮廓
                box_outline = pg.PlotCurveItem(box_x, box_y, pen=pg.mkPen(color=(0, 0, 0), width=2))
                self.plot_widget.addItem(box_outline)
                
                # 绘制中位数线
                median_line = pg.PlotCurveItem([i - box_width/2, i + box_width/2], [q2, q2], 
                                             pen=pg.mkPen(color=(0, 0, 0), width=3))
                self.plot_widget.addItem(median_line)
                
                # 绘制下须线
                lower_whisker_line = pg.PlotCurveItem([i, i], [q1, lower_whisker], 
                                                    pen=pg.mkPen(color=(0, 0, 0), width=2))
                self.plot_widget.addItem(lower_whisker_line)
                
                # 绘制上须线
                upper_whisker_line = pg.PlotCurveItem([i, i], [q3, upper_whisker], 
                                                    pen=pg.mkPen(color=(0, 0, 0), width=2))
                self.plot_widget.addItem(upper_whisker_line)
                
                # 绘制须线端点
                whisker_cap_width = box_width * 0.3
                lower_cap = pg.PlotCurveItem([i - whisker_cap_width/2, i + whisker_cap_width/2], 
                                           [lower_whisker, lower_whisker], pen=pg.mkPen(color=(0, 0, 0), width=2))
                upper_cap = pg.PlotCurveItem([i - whisker_cap_width/2, i + whisker_cap_width/2], 
                                           [upper_whisker, upper_whisker], pen=pg.mkPen(color=(0, 0, 0), width=2))
                self.plot_widget.addItem(lower_cap)
                self.plot_widget.addItem(upper_cap)
                
                # 绘制异常值
                outliers = data[(data < lower_whisker) | (data > upper_whisker)]
                if len(outliers) > 0:
                    outlier_x = np.full(len(outliers), i)
                    outlier_scatter = pg.ScatterPlotItem(x=outlier_x, y=outliers.values,
                                                       pen=pg.mkPen(color=(0, 0, 0)), brush=pg.mkBrush(color=(255, 255, 255)),
                                                       size=6, symbol='o')
                    self.plot_widget.addItem(outlier_scatter)
                
                # 添加数据点（可选，带抖动）
                jitter = np.random.normal(0, 0.05, len(data))
                x_pos = np.full(len(data), i) + jitter
                
                all_data.extend(data.values)
                x_positions.extend(x_pos)
            
            # 绘制所有数据点（半透明）
            scatter = pg.ScatterPlotItem(x=x_positions, y=all_data, 
                                       pen=pg.mkPen(color, alpha=100), brush=pg.mkBrush(color, alpha=100),
                                       size=4, symbol='o', name=label)
            self.plot_widget.addItem(scatter)
            
            # 设置x轴标签
            x_ticks = [(i, cat) for i, cat in enumerate(categories)]
            ax = self.plot_widget.getAxis('bottom')
            ax.setTicks([x_ticks])
            
        elif plot_type == "Histogram":
            # 对于直方图，计算并显示频数分布
            for i, (col_label, data) in enumerate(data_dict.items()):
                y, x = np.histogram(data, bins=self.bins_spinbox.value())
                # 创建阶梯图
                curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, 
                                       brush=pg.mkBrush(color, alpha=100),
                                       pen=pg.mkPen(color, width=2), name=f"{label}")
                self.plot_widget.addItem(curve)
            
        elif plot_type == "Violin Plot":
            # 对于小提琴图，使用核密度估计
            categories = []
            for i, (col_spec, data) in enumerate(data_dict.items()):
                filename = col_spec.split('.')[0]
                
                # 解析文件名，提取日期和样品名称
                # 文件名格式为: 20250828_samplename_analysisxxx
                if '_' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        date_part = parts[0]  # 例如: 20250828
                        sample_part = parts[1]  # 例如: samplename
                        category = f"{date_part}\n{sample_part}"  # 使用换行符分隔
                    else:
                        category = parts[0]
                else:
                    category = filename
                categories.append(category)
                
                # 简化版本：使用散点图 + 密度线
                from scipy.stats import gaussian_kde
                
                kde = gaussian_kde(data)
                y_range = np.linspace(data.min(), data.max(), 100)
                density = kde(y_range)
                
                # 归一化密度到合适的宽度
                density = density / density.max() * 0.4
                
                # 绘制左右对称的密度曲线
                x_left = i - density
                x_right = i + density
                
                curve_left = pg.PlotCurveItem(x_left, y_range, pen=pg.mkPen(color, width=2))
                curve_right = pg.PlotCurveItem(x_right, y_range, pen=pg.mkPen(color, width=2))
                self.plot_widget.addItem(curve_left)
                self.plot_widget.addItem(curve_right)
                
                # 添加数据点
                jitter = np.random.normal(0, 0.05, len(data))
                scatter = pg.ScatterPlotItem(x=np.full(len(data), i) + jitter, y=data.values,
                                           pen=pg.mkPen(color), brush=pg.mkBrush(color),
                                           size=4, symbol='o', name=f"{label}" if i == 0 else None)
                self.plot_widget.addItem(scatter)
            
            # 设置x轴标签
            x_ticks = [(i, cat) for i, cat in enumerate(categories)]
            ax = self.plot_widget.getAxis('bottom')
            ax.setTicks([x_ticks])
        
        print(f"Debug: PyQtGraph plot creation completed for {plot_type}")
    
    def create_dual_column_plot(self, ax, x_column, y_column, plot_type):
        """创建双列图表"""
        print(f"Debug: Creating {plot_type} with X: {x_column}, Y: {y_column}")
        
        try:
            # Parse column specifications
            x_last_dot = x_column.rfind('.')
            y_last_dot = y_column.rfind('.')
            
            if x_last_dot == -1 or y_last_dot == -1:
                ax.text(0.5, 0.5, "列格式错误", ha='center', va='center', transform=ax.transAxes)
                return
                
            x_filename = x_column[:x_last_dot]
            x_col_name = x_column[x_last_dot + 1:]
            y_filename = y_column[:y_last_dot]
            y_col_name = y_column[y_last_dot + 1:]
            
            # Get data
            x_data = pd.Series()
            y_data = pd.Series()
            
            if x_filename in self.csv_files and x_col_name in self.csv_files[x_filename].columns:
                x_data = self.csv_files[x_filename][x_col_name].dropna()
                
            if y_filename in self.csv_files and y_col_name in self.csv_files[y_filename].columns:
                y_data = self.csv_files[y_filename][y_col_name].dropna()
            
            if len(x_data) == 0 or len(y_data) == 0:
                ax.text(0.5, 0.5, "没有可用数据", ha='center', va='center', transform=ax.transAxes)
                return
            
            # Ensure same length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]
            
            if plot_type == "Scatter Plot":
                ax.scatter(x_data, y_data, alpha=0.7)
                
                # Add trend line
                if len(x_data) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                    line_x = np.array([x_data.min(), x_data.max()])
                    line_y = slope * line_x + intercept
                    ax.plot(line_x, line_y, 'r--', alpha=0.8)
                    ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            elif plot_type == "KDE Plot":
                # Create 2D KDE plot
                import matplotlib.pyplot as plt
                from scipy.stats import gaussian_kde
                
                # Create a grid for the KDE
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                
                # Add some padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min -= x_range * 0.1
                x_max += x_range * 0.1
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
                
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x_data, y_data])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Plot contours
                ax.contour(xx, yy, f, alpha=0.7)
                ax.scatter(x_data, y_data, alpha=0.5, s=20)
            
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f"{plot_type}: {x_column} vs {y_column}")
            
            print(f"Debug: Dual column plot creation completed")
            
        except Exception as e:
            print(f"Debug: Error in dual column plot: {e}")
            ax.text(0.5, 0.5, f"创建图表时出错: {str(e)}", ha='center', va='center', transform=ax.transAxes)
    
    def create_dual_column_plot_with_color(self, ax, x_column, y_column, plot_type, color, label):
        """创建带指定颜色的双列图表"""
        print(f"Debug: Creating {plot_type} with color {color} and label {label}")
        
        try:
            # Parse column specifications
            x_last_dot = x_column.rfind('.')
            y_last_dot = y_column.rfind('.')
            
            if x_last_dot == -1 or y_last_dot == -1:
                print("Debug: Invalid column format")
                return
                
            x_filename = x_column[:x_last_dot]
            x_col_name = x_column[x_last_dot + 1:]
            y_filename = y_column[:y_last_dot]
            y_col_name = y_column[y_last_dot + 1:]
            
            # Get data
            x_data = pd.Series()
            y_data = pd.Series()
            
            if x_filename in self.csv_files and x_col_name in self.csv_files[x_filename].columns:
                x_data = self.csv_files[x_filename][x_col_name].dropna()
                
            if y_filename in self.csv_files and y_col_name in self.csv_files[y_filename].columns:
                y_data = self.csv_files[y_filename][y_col_name].dropna()
            
            if len(x_data) == 0 or len(y_data) == 0:
                print("Debug: No valid data for dual column plot with color")
                return
            
            # Ensure same length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]
            
            if plot_type == "Scatter Plot":
                ax.scatter(x_data, y_data, alpha=0.7, color=color, label=label)
                
            elif plot_type == "KDE Plot":
                # Create 2D KDE plot
                from scipy.stats import gaussian_kde
                
                # Create a grid for the KDE
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                
                # Add some padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min -= x_range * 0.1
                x_max += x_range * 0.1
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
                
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x_data, y_data])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Plot contours with specified color
                contour = ax.contour(xx, yy, f, alpha=0.7, colors=[color])
                ax.scatter(x_data, y_data, alpha=0.5, s=20, color=color, label=label)
            
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            
            print(f"Debug: Dual column plot with color creation completed")
            
        except Exception as e:
            print(f"Debug: Error in dual column plot with color: {e}")
    
    def create_dual_column_plot_with_color_pg(self, x_column, y_column, plot_type, color, label):
        """使用 pyqtgraph 创建带指定颜色的双列图表"""
        print(f"Debug: Creating pyqtgraph {plot_type} with color {color} and label {label}")
        
        try:
            # Parse column specifications
            x_last_dot = x_column.rfind('.')
            y_last_dot = y_column.rfind('.')
            
            if x_last_dot == -1 or y_last_dot == -1:
                print("Debug: Invalid column format")
                return
                
            x_filename = x_column[:x_last_dot]
            x_col_name = x_column[x_last_dot + 1:]
            y_filename = y_column[:y_last_dot]
            y_col_name = y_column[y_last_dot + 1:]
            
            # Get data
            x_data = pd.Series()
            y_data = pd.Series()
            
            if x_filename in self.csv_files and x_col_name in self.csv_files[x_filename].columns:
                x_data = self.csv_files[x_filename][x_col_name].dropna()
                
            if y_filename in self.csv_files and y_col_name in self.csv_files[y_filename].columns:
                y_data = self.csv_files[y_filename][y_col_name].dropna()
            
            if len(x_data) == 0 or len(y_data) == 0:
                print("Debug: No valid data for dual column plot with color")
                return
            
            # Ensure same length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]
            
            if plot_type == "Scatter Plot":
                # 创建散点图
                scatter = pg.ScatterPlotItem(x=x_data.values, y=y_data.values,
                                           pen=pg.mkPen(color), brush=pg.mkBrush(color),
                                           size=8, symbol='o', name=label)
                self.plot_widget.addItem(scatter)
                
                # 添加趋势线
                if len(x_data) > 1:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                    line_x = np.array([x_data.min(), x_data.max()])
                    line_y = slope * line_x + intercept
                    
                    trend_line = pg.PlotCurveItem(line_x, line_y, pen=pg.mkPen(color, width=2, style=Qt.DashLine))
                    self.plot_widget.addItem(trend_line)
                    
                    # 添加R²值标签
                    text_item = pg.TextItem(f'R² = {r_value**2:.3f}', color=color)
                    text_item.setPos(x_data.quantile(0.05), y_data.quantile(0.95))
                    self.plot_widget.addItem(text_item)
                
            elif plot_type == "KDE Plot":
                # 对于KDE图，创建等高线图的简化版本
                from scipy.stats import gaussian_kde
                
                # 创建网格
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                
                # 添加边距
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min -= x_range * 0.1
                x_max += x_range * 0.1
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
                
                # 首先绘制原始数据点
                scatter = pg.ScatterPlotItem(x=x_data.values, y=y_data.values,
                                           pen=pg.mkPen(color), brush=pg.mkBrush(color),
                                           size=6, symbol='o', name=label)
                self.plot_widget.addItem(scatter)
                
                # 简化的密度表示：使用颜色深度表示密度
                # 这里可以添加更复杂的KDE实现
            
            # 设置轴标签
            self.plot_widget.setLabel('bottom', x_column)
            self.plot_widget.setLabel('left', y_column)
            
            print(f"Debug: PyQtGraph dual column plot creation completed")
            
        except Exception as e:
            print(f"Debug: Error in pyqtgraph dual column plot: {e}")
  
    def enable_export_buttons(self, enabled):
        """启用/禁用导出按钮"""
        self.export_data_btn.setEnabled(enabled)
        self.export_stats_btn.setEnabled(enabled)
        self.export_image_btn.setEnabled(enabled)
    
    def export_current_data(self):
        """导出当前显示的数据"""
        if not self.csv_files or self.selected_columns_list.count() == 0:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "current_data.csv", "CSV files (*.csv)"
        )
        
        if filename:
            try:
                # 获取当前选中的列
                selected_columns = []
                for i in range(self.selected_columns_list.count()):
                    selected_columns.append(self.selected_columns_list.item(i).text())
                
                # 合并选中的数据
                combined_data = pd.DataFrame()
                max_length = 0
                data_info = []
                
                # 首先收集所有数据信息，找到最大长度
                for col_spec in selected_columns:
                    try:
                        # 解析文件名和列名
                        last_dot_idx = col_spec.rfind('.')
                        if last_dot_idx == -1:
                            continue
                        
                        file_name = col_spec[:last_dot_idx]
                        column = col_spec[last_dot_idx + 1:]
                        
                        if file_name in self.csv_files and column in self.csv_files[file_name].columns:
                            data = self.csv_files[file_name][column].dropna()  # 移除NaN值
                            # 使用文件名中的样品名作为列名
                            # 文件名格式为: 20250828_samplename_analysisxxx
                            if '_' in file_name:
                                parts = file_name.split('_')
                                if len(parts) >= 2:
                                    sample_name = parts[1]  # 使用第二段作为样品名
                                else:
                                    sample_name = parts[0]
                            else:
                                sample_name = file_name
                            new_col_name = f"{sample_name}_{column}"
                            
                            data_info.append({
                                'name': new_col_name,
                                'data': data,
                                'length': len(data)
                            })
                            max_length = max(max_length, len(data))
                            print(f"Debug: Found column {new_col_name} with {len(data)} data points")
                    except Exception as e:
                        print(f"Debug: Error processing column {col_spec}: {e}")
                        continue
                
                # 创建DataFrame，所有列都填充到相同长度
                for info in data_info:
                    data = info['data']
                    if len(data) < max_length:
                        # 如果数据长度不足，用NaN填充
                        padded_data = pd.Series([np.nan] * max_length)
                        padded_data[:len(data)] = data
                        combined_data[info['name']] = padded_data
                        print(f"Debug: Padded column {info['name']} from {len(data)} to {max_length} rows")
                    else:
                        combined_data[info['name']] = data
                        print(f"Debug: Added column {info['name']} with {len(data)} data points")
                
                if combined_data.empty:
                    QMessageBox.warning(self, "警告", "没有有效数据可导出")
                    return
                
                # 导出数据
                combined_data.to_csv(filename, index=False)
                
                # 统计有效数据行数（非全NaN的行）
                valid_rows = combined_data.dropna(how='all').shape[0]
                total_rows = len(combined_data)
                
                QMessageBox.information(self, "成功", 
                    f"数据已导出到 {filename}\n"
                    f"包含 {len(combined_data.columns)} 列，{total_rows} 行数据\n"
                    f"有效数据行数: {valid_rows}\n"
                    f"最大数据长度: {max_length}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                print(f"Debug: Export error: {e}")

    def export_statistics(self):
        """导出统计摘要"""
        if not self.csv_files or self.selected_columns_list.count() == 0:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存统计摘要", "statistics.csv", "CSV files (*.csv)"
        )
        
        if filename:
            try:
                # 获取当前选中的列
                selected_columns = []
                for i in range(self.selected_columns_list.count()):
                    selected_columns.append(self.selected_columns_list.item(i).text())
                
                stats_data = []
                
                for col_spec in selected_columns:
                    try:
                        # 解析文件名和列名
                        last_dot_idx = col_spec.rfind('.')
                        if last_dot_idx == -1:
                            continue
                        
                        file_name = col_spec[:last_dot_idx]
                        column = col_spec[last_dot_idx + 1:]
                        
                        if file_name in self.csv_files and column in self.csv_files[file_name].columns:
                            data = self.csv_files[file_name][column].dropna()
                            
                            if len(data) > 0:
                                # 使用文件名中的样品名
                                # 文件名格式为: 20250828_samplename_analysisxxx
                                if '_' in file_name:
                                    parts = file_name.split('_')
                                    if len(parts) >= 2:
                                        sample_name = parts[1]  # 使用第二段作为样品名
                                    else:
                                        sample_name = parts[0]
                                else:
                                    sample_name = file_name
                                
                                stats_data.append({
                                    'Sample': sample_name,
                                    'File': file_name,
                                    'Column': column,
                                    'Count': len(data),
                                    'Mean': data.mean(),
                                    'Std': data.std(),
                                    'Min': data.min(),
                                    'Max': data.max(),
                                    'Q25': data.quantile(0.25),
                                    'Q50': data.quantile(0.50),
                                    'Q75': data.quantile(0.75)
                                })
                                print(f"Debug: Added stats for {sample_name} ({file_name}.{column})")
                    except Exception as e:
                        print(f"Debug: Error processing stats for {col_spec}: {e}")
                        continue
                
                if not stats_data:
                    QMessageBox.warning(self, "警告", "没有有效数据可导出统计信息")
                    return
                
                # 创建统计摘要DataFrame并导出
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_csv(filename, index=False)
                QMessageBox.information(self, "成功", f"统计摘要已导出到 {filename}\n包含 {len(stats_data)} 个数据列的统计信息")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                print(f"Debug: Stats export error: {e}")

    def export_image(self):
        """导出图像"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "plot.png", "PNG files (*.png);;SVG files (*.svg)"
        )
        
        if filename:
            try:
                # 使用 pyqtgraph 的导出功能
                exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
                exporter.export(filename)
                print(f"图像已保存到 {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                print(f"Debug: Image export error: {e}")

class MainApplication(QMainWindow):
    """主应用程序"""
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Current Event Analyzer - 多功能版本')
        self.setGeometry(100, 100, 1400, 750)  # Reduced from 1600x900 to 1400x750
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        central_widget.setLayout(QVBoxLayout())
        central_widget.layout().addWidget(self.tab_widget)
        
        # Create tabs
        self.single_event_tab = SingleEventAnalyzer()
        self.multi_file_tab = MultiFileVisualizer()
        
        self.tab_widget.addTab(self.single_event_tab, "单事件分析")
        self.tab_widget.addTab(self.multi_file_tab, "多文件可视化")

def main():
    # 设置环境变量以减少 macOS 警告
    import os
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-logging'
    
    app = QApplication(sys.argv)
    
    # 设置应用属性以减少警告
    app.setAttribute(Qt.AA_DontShowIconsInMenus, True)
    app.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)
    
    # 抑制 Qt 的调试输出
    import logging
    logging.getLogger('qt').setLevel(logging.WARNING)
    
    window = MainApplication()
    window.show()
    sys.exit(app.exec_())


class PositionLabelDialog(QDialog):
    """Position Label编辑对话框"""
    
    def __init__(self, peaks_data, parent=None):
        super().__init__(parent)
        self.peaks_data = peaks_data
        self.setWindowTitle("编辑Position Labels")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        
        # 说明文字
        info_label = QLabel("为每个峰值设置Position Label（可选）:")
        layout.addWidget(info_label)
        
        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["峰值编号", "Prominence", "宽度 (ms)", "Position Label"])
        
        # 设置列宽
        self.table.setColumnWidth(0, 100)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 200)
        
        # 填充数据
        self.table.setRowCount(len(peaks_data))
        for i, peak_data in enumerate(peaks_data):
            # 峰值编号
            peak_num_item = QTableWidgetItem(f"峰值 {i+1}")
            peak_num_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 0, peak_num_item)
            
            # Prominence
            prominence_item = QTableWidgetItem(f"{peak_data['peak_amplitude']:.6f}")
            prominence_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 1, prominence_item)
            
            # 宽度 (μs)
            width_item = QTableWidgetItem(f"{peak_data['peak_width_us']:.2f}")
            width_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 2, width_item)
            
            # Position Label (可编辑)
            label_item = QTableWidgetItem("")
            label_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable)
            self.table.setItem(i, 3, label_item)
        
        layout.addWidget(self.table)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_labels(self):
        """获取所有position labels"""
        labels = []
        for i in range(self.table.rowCount()):
            label_item = self.table.item(i, 3)  # Position Label现在是第4列（索引3）
            labels.append(label_item.text() if label_item else "")
        return labels


if __name__ == '__main__':
    main()
