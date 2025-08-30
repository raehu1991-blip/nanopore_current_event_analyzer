import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from scipy.signal import find_peaks
from nptdms import TdmsFile
import csv

class CurrentEventAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.time_data = None
        self.sample_rate = 250000  # 250kHz as specified
        self.current_folder = None
        self.file_list = []
        self.current_file_index = 0
        self.peaks_data = []
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Current Event Analyzer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # File loading section
        file_group = QGroupBox("文件加载")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Load TDMS/NPZ File Folder")
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
        
        # Export section
        export_group = QGroupBox("导出结果")
        export_layout = QVBoxLayout()
        
        self.export_btn = QPushButton("Export Selected to CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)
        
        # Peak Review section
        review_group = QGroupBox("峰值审核")
        review_layout = QVBoxLayout()
        
        # Peak list with checkboxes
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(150)
        review_layout.addWidget(self.peak_list_widget)
        
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
        
        review_group.setLayout(review_layout)
        left_layout.addWidget(review_group)
        
        # Analysis parameters section
        param_group = QGroupBox("分析参数")
        param_layout = QVBoxLayout()
        
        # Threshold setting
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("阈值 (Threshold):"))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-10.0, 10.0)
        self.threshold_input.setValue(-0.1)  # Default as specified
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setDecimals(3)
        thresh_layout.addWidget(self.threshold_input)
        param_layout.addLayout(thresh_layout)
        
        # Peak detection height parameter
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.0, 10.0)
        self.height_input.setValue(0.02)  # As specified in requirements
        self.height_input.setSingleStep(0.001)
        self.height_input.setDecimals(4)
        height_layout.addWidget(self.height_input)
        param_layout.addLayout(height_layout)
        
        # Background color selection
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("背景颜色:"))
        self.bg_color_combo = QComboBox()
        self.bg_color_combo.addItems(["白色", "黑色", "浅灰色", "深灰色"])
        self.bg_color_combo.currentTextChanged.connect(self.change_background_color)
        bg_layout.addWidget(self.bg_color_combo)
        param_layout.addLayout(bg_layout)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)
        
        # Results section
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # Right panel for plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', '电流 (A)', 'A')
        self.plot_widget.setLabel('bottom', '时间 (s)', 's')
        self.plot_widget.showGrid(True, True)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Add crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen='y')
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Connect mouse move event
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Add coordinate label
        self.coord_label = QLabel("Position: (0, 0)")
        
        plot_panel = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addWidget(self.coord_label)
        plot_panel.setLayout(plot_layout)
        
        main_layout.addWidget(plot_panel, stretch=1)
        
    def mouse_moved(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self.coord_label.setText(f"Position: ({x:.6f}, {y:.6f})")
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
        
    def load_file_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含TDMS/NPZ文件的文件夹")
        if folder_path:
            self.current_folder = folder_path
            self.file_list = []
            
            # Find all TDMS and NPZ files
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.tdms', '.npz')):
                    self.file_list.append(os.path.join(folder_path, file))
            
            if self.file_list:
                self.file_list.sort()
                self.current_file_index = 0
                self.load_current_file()
                self.prev_btn.setEnabled(len(self.file_list) > 1)
                self.next_btn.setEnabled(len(self.file_list) > 1)
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中未找到TDMS或NPZ文件。")
    
    def load_current_file(self):
        if not self.file_list:
            return
            
        file_path = self.file_list[self.current_file_index]
        file_name = os.path.basename(file_path)
        
        try:
            if file_path.lower().endswith('.tdms'):
                    self.load_tdms_file(file_path)
            elif file_path.lower().endswith('.npz'):
                    self.load_npz_file(file_path)
                
            self.file_info_label.setText(f"文件 {self.current_file_index + 1}/{len(self.file_list)}: {file_name}")
            self.plot_data()
            self.analyze_btn.setEnabled(True)
            
            # Reset analysis results
            self.peaks_data = []
            self.results_text.clear()
            self.peak_list_widget.clear()
            self.export_btn.setEnabled(False)
            self.select_all_btn.setEnabled(False)
            self.select_none_btn.setEnabled(False)
            
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
        
        # Plot the current data
        self.plot_widget.plot(self.time_data, self.current_data, pen='b', name='电流')
        
        # Add threshold line
        threshold = self.threshold_input.value()
        threshold_line = pg.InfiniteLine(pos=threshold, angle=0, pen=pg.mkPen('r', style=Qt.DashLine))
        self.plot_widget.addItem(threshold_line)
            
    def analyze_data(self):
        if self.current_data is None or self.time_data is None:
            return
            
        threshold = self.threshold_input.value()
        height = self.height_input.value()
        
        # Find data points below threshold
        below_threshold_mask = self.current_data < threshold
        below_threshold_indices = np.where(below_threshold_mask)[0]
        
        if len(below_threshold_indices) == 0:
            self.results_text.setText("未找到低于阈值的数据点。")
            return
            
        # Extract data below threshold
        below_threshold_current = self.current_data[below_threshold_indices]
        below_threshold_time = self.time_data[below_threshold_indices]
        
        # Use find_peaks directly on the original current data for negative peaks
        # Find peaks in negative current (invert the signal)
        peaks, properties = find_peaks(-self.current_data, height=height, prominence=0.08)
        
        # Filter peaks to only include those below threshold
        threshold_peaks = []
        for peak in peaks:
            if self.current_data[peak] < threshold:
                threshold_peaks.append(peak)
        
        peaks = np.array(threshold_peaks)
        
        if len(peaks) == 0:
            self.results_text.setText("在低于阈值的数据中未找到峰值。")
            return
        
        # Calculate total time below threshold
        total_time = 0
        if len(below_threshold_indices) > 1:
            # Find continuous segments
            segments = []
            start = 0
            for i in range(1, len(below_threshold_indices)):
                if below_threshold_indices[i] - below_threshold_indices[i-1] > 1:
                    segments.append((start, i-1))
                    start = i
            segments.append((start, len(below_threshold_indices)-1))
            
            # Calculate total time for all segments
            for seg_start, seg_end in segments:
                if seg_end > seg_start:
                    time_diff = below_threshold_time[seg_end] - below_threshold_time[seg_start]
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
            
            # Calculate peak amplitude (start current - peak current)
            amplitude = start_current - peak_currents[i]
            peak_amplitudes.append(amplitude)
        
        # Store results
        current_file = os.path.basename(self.file_list[self.current_file_index]) if self.file_list else "Unknown"
        
        self.peaks_data = []
        for i in range(len(peaks)):
            self.peaks_data.append({
                'file_name': current_file,
                'peak_number': i + 1,
                'total_time': total_time,
                'peak_t': peak_times[i],
                'peak_i': peak_currents[i],
                'peak_start_i': peak_start_currents[i],
                'peak_start_t': peak_start_times[i],
                'peak_amplitude': peak_amplitudes[i]
            })
        
        # Display results
        results_text = f"分析结果:\n"
        results_text += f"文件: {current_file}\n"
        results_text += f"找到峰值: {len(peaks)}\n"
        results_text += f"总时间 (低于阈值): {total_time:.6f} s\n\n"
        
        for i, peak_data in enumerate(self.peaks_data):
            results_text += f"峰值 {i+1}:\n"
            results_text += f"  时间: {peak_data['peak_t']:.6f}s\n"
            results_text += f"  电流: {peak_data['peak_i']:.6f}A\n"
            results_text += f"  起点时间: {peak_data['peak_start_t']:.6f}s\n"
            results_text += f"  起点电流: {peak_data['peak_start_i']:.6f}A\n"
            results_text += f"  振幅: {peak_data['peak_amplitude']:.6f}A\n\n"
        
        self.results_text.setText(results_text)
        
        # Populate peak list for review
        self.peak_list_widget.clear()
        for i, peak_data in enumerate(self.peaks_data):
            item_text = f"峰值 {i+1}: 时间={peak_data['peak_t']:.6f}s, 振幅={peak_data['peak_amplitude']:.6f}A"
            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # Default to checked
            self.peak_list_widget.addItem(item)
        
        # Plot peaks on the graph
        self.plot_data()
        
        # Add peak markers
        if len(peak_times) > 0:
            self.plot_widget.plot(peak_times, peak_currents, 
                                pen=None, symbol='o', symbolBrush='r', symbolSize=8, name='峰值')
        
        # Add start point markers
        if len(peak_start_times) > 0:
            self.plot_widget.plot(peak_start_times, peak_start_currents, 
                                pen=None, symbol='t', symbolBrush='g', symbolSize=8, name='起点')
        
        # Enable review and export buttons
        self.select_all_btn.setEnabled(True)
        self.select_none_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
    
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
    
    def select_all_peaks(self):
        """选择所有峰值"""
        for i in range(self.peak_list_widget.count()):
            item = self.peak_list_widget.item(i)
            item.setCheckState(Qt.Checked)
    
    def select_none_peaks(self):
        """取消选择所有峰值"""
        for i in range(self.peak_list_widget.count()):
            item = self.peak_list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def get_selected_peaks(self):
        """获取选中的峰值数据"""
        selected_peaks = []
        for i in range(self.peak_list_widget.count()):
            item = self.peak_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_peaks.append(self.peaks_data[i])
        return selected_peaks
    
    def export_to_csv(self):
        if not self.peaks_data:
            QMessageBox.warning(self, "警告", "没有分析结果可导出。")
            return
            
        # Get selected peaks
        selected_peaks = self.get_selected_peaks()
        if not selected_peaks:
            QMessageBox.warning(self, "警告", "请至少选择一个峰值进行导出。")
            return
            
        # Create CSV filename based on folder name
        folder_name = os.path.basename(self.current_folder) if self.current_folder else "analysis"
        csv_filename = f"{folder_name}_analysis_results.csv"
        csv_path = os.path.join(self.current_folder, csv_filename)
        
        # Always check if file exists and append
        file_exists = os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['file_name', 'peak_number', 'total_time', 'peak_t', 'peak_i', 
                             'peak_start_i', 'peak_start_t', 'peak_amplitude']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header only if file doesn't exist
                if not file_exists:
                    writer.writeheader()
                
                # Write only selected peaks data
                for peak_data in selected_peaks:
                    writer.writerow(peak_data)
            
            message = f"成功导出 {len(selected_peaks)} 个峰值到 {csv_path}"
            if file_exists:
                message += "\n(数据已追加到现有文件)"
            QMessageBox.information(self, "导出成功", message)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出CSV失败: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = CurrentEventAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()