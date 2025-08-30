import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from nptdms import TdmsFile
import tkinter as tk
from tkinter import filedialog
import os

def load_tdms_file():
    """让用户选择并加载TDMS文件"""
    # 创建tkinter根窗口（隐藏）
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择TDMS文件",
        filetypes=[("TDMS files", "*.tdms"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("未选择文件，程序退出")
        return None, None
    
    try:
        # 读取TDMS文件
        with TdmsFile.read(file_path) as tdms_file:
            # 获取所有组和通道
            groups = tdms_file.groups()
            if not groups:
                raise ValueError("TDMS文件中未找到数据组")
            
            print(f"加载文件: {os.path.basename(file_path)}")
            print(f"数据组数量: {len(groups)}")
            
            # 查找包含电流数据的通道
            current_channel = None
            sample_rate = None
            
            for group in groups:
                print(f"组名: {group.name}")
                for channel in group.channels():
                    print(f"  通道名: {channel.name}")
                    
                    # 查找采样率信息
                    if hasattr(channel, 'properties') and 'wf_samples' in channel.properties:
                        sample_rate = channel.properties.get('wf_samples', 250000)
                    
                    # 查找电流数据通道
                    if 'current' in channel.name.lower() or current_channel is None:
                        current_channel = channel
            
            if current_channel is None:
                raise ValueError("未在TDMS文件中找到电流数据通道")
            
            # 提取数据
            current_data = np.array(current_channel.data)
            
            # 使用默认采样率如果未找到
            if sample_rate is None:
                sample_rate = 250000  # 250kHz 默认采样率
            
            # 生成时间数组
            time_data = np.arange(len(current_data)) / sample_rate
            
            print(f"数据点数: {len(current_data)}")
            print(f"采样率: {sample_rate} Hz")
            print(f"总时间: {time_data[-1]:.6f} 秒")
            
            return time_data, current_data
            
    except Exception as e:
        print(f"加载TDMS文件时出错: {str(e)}")
        return None, None

# 加载TDMS文件数据
print("请选择TDMS文件...")
time, current = load_tdms_file()

# 如果文件加载失败，使用示例数据
if time is None or current is None:
    print("使用示例数据...")
    time = np.linspace(0, 1, 1000)  # 示例时间
    current = np.sin(2 * np.pi * 5 * time) * -100 + np.random.normal(0, 10, time.shape)  # 示例电流数据

# 查找峰值（对于负值电流，我们查找负向峰值）
# 反转数据来查找负向峰值
peaks, properties = find_peaks(-current, height=0.01, distance=100, prominence=0.001)

print(f"找到 {len(peaks)} 个峰值")

# 提取每个峰的起点
peak_starts = []
for peak in peaks:
    # 从峰值向前寻找起点（电流开始下降的地方）
    start_idx = peak
    for i in range(peak, 0, -1):
        if i == 0:
            start_idx = 0
            break
        # 寻找电流开始显著下降的点
        if current[i] > current[i-1]:  # 找到上升趋势的结束点
            start_idx = i
            break
        start_idx = i
    peak_starts.append(start_idx)

print(f"峰值起点索引: {peak_starts}")

# 绘图显示结果
plt.figure(figsize=(15, 8))
plt.plot(time, current, label='电流 (A)', color='blue', linewidth=1)

if len(peaks) > 0:
    plt.plot(time[peaks], current[peaks], "x", label='峰值', color='red', markersize=8)
    plt.scatter(time[peak_starts], current[peak_starts], color='green', label='峰值起点', zorder=5, s=50)

plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='零线')
plt.title('电流峰值分析', fontsize=14)
plt.xlabel('时间 (s)', fontsize=12)
plt.ylabel('电流 (A)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 显示统计信息
if len(peaks) > 0:
    plt.text(0.02, 0.98, f'峰值数量: {len(peaks)}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# 输出详细分析结果
if len(peaks) > 0:
    print(f"\n=== 峰值分析结果 ===")
    print(f"总峰值数量: {len(peaks)}")
    print(f"峰值时间: {time[peaks]}")
    print(f"峰值电流: {current[peaks]}")
    print(f"起点时间: {time[peak_starts]}")
    print(f"起点电流: {current[peak_starts]}")
    
    # 计算相对值
    for i, (peak_idx, start_idx) in enumerate(zip(peaks, peak_starts)):
        relative_time = time[peak_idx] - time[start_idx]
        relative_amplitude = current[peak_idx] - current[start_idx]
        print(f"峰值 {i+1}: 相对时间={relative_time:.6f}s, 相对振幅={relative_amplitude:.6f}A")
else:
    print("未检测到峰值")