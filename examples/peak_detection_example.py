#!/usr/bin/env python3
"""
峰值检测示例脚本
演示如何使用scipy.find_peaks进行电流峰值检测

这是一个独立的示例，展示了峰值检测的基本原理。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_sample_data():
    """生成示例电流数据"""
    # 创建时间序列
    time = np.linspace(0, 2, 2000)  # 2秒，2000个点
    
    # 基础电流信号
    current = np.zeros_like(time)
    
    # 添加一些电流事件（负向峰值）
    for peak_time in [0.3, 0.7, 1.2, 1.6]:
        # 高斯形状的负向峰值
        peak_amplitude = np.random.uniform(0.5, 1.0)
        width = 0.05
        peak_profile = -peak_amplitude * np.exp(-((time - peak_time) / width) ** 2)
        current += peak_profile
    
    # 添加噪声
    noise = np.random.normal(0, 0.05, len(time))
    current += noise
    
    return time, current

def detect_peaks_example():
    """峰值检测示例"""
    # 生成示例数据
    time, current = generate_sample_data()
    
    # 设置检测参数
    threshold = -0.1  # 阈值
    prominence = 0.08  # 显著性
    height = 0.05      # 最小高度
    
    print("峰值检测参数:")
    print(f"  阈值 (threshold): {threshold}")
    print(f"  显著性 (prominence): {prominence}")
    print(f"  最小高度 (height): {height}")
    print()
    
    # 找到低于阈值的数据点
    below_threshold = current < threshold
    below_indices = np.where(below_threshold)[0]
    
    print(f"低于阈值的数据点数量: {len(below_indices)}")
    
    # 使用find_peaks检测负向峰值
    # 通过对信号取负值来检测负向峰值
    peaks, properties = find_peaks(-current, 
                                 height=height, 
                                 prominence=prominence,
                                 distance=50)  # 最小峰值间距
    
    # 过滤出低于阈值的峰值
    valid_peaks = []
    for peak in peaks:
        if current[peak] < threshold:
            valid_peaks.append(peak)
    
    valid_peaks = np.array(valid_peaks)
    
    print(f"检测到的峰值数量: {len(valid_peaks)}")
    
    # 计算峰值信息
    if len(valid_peaks) > 0:
        peak_times = time[valid_peaks]
        peak_currents = current[valid_peaks]
        
        print("\n峰值详细信息:")
        for i, (t, curr) in enumerate(zip(peak_times, peak_currents)):
            print(f"  峰值 {i+1}: 时间={t:.3f}s, 电流={curr:.3f}A")
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 主图 - 电流vs时间
    plt.subplot(2, 1, 1)
    plt.plot(time, current, 'b-', linewidth=1, label='电流信号')
    plt.axhline(threshold, color='r', linestyle='--', alpha=0.7, label=f'阈值 ({threshold})')
    
    if len(valid_peaks) > 0:
        plt.plot(time[valid_peaks], current[valid_peaks], 
                'ro', markersize=8, label=f'检测到的峰值 ({len(valid_peaks)}个)')
    
    plt.xlabel('时间 (s)')
    plt.ylabel('电流 (A)')
    plt.title('电流峰值检测示例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 - 峰值检测结果的放大视图
    plt.subplot(2, 1, 2)
    if len(valid_peaks) > 0:
        # 显示第一个峰值的详细视图
        peak_idx = valid_peaks[0]
        start_idx = max(0, peak_idx - 100)
        end_idx = min(len(time), peak_idx + 100)
        
        plt.plot(time[start_idx:end_idx], current[start_idx:end_idx], 'b-', linewidth=2)
        plt.plot(time[peak_idx], current[peak_idx], 'ro', markersize=10, label='峰值')
        plt.axhline(threshold, color='r', linestyle='--', alpha=0.7, label='阈值')
        
        plt.xlabel('时间 (s)')
        plt.ylabel('电流 (A)')
        plt.title(f'峰值详细视图 - 峰值 1')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '未检测到峰值', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    return time, current, valid_peaks

if __name__ == '__main__':
    print("电流峰值检测示例")
    print("=" * 30)
    
    try:
        time_data, current_data, detected_peaks = detect_peaks_example()
        print(f"\n示例完成！检测到 {len(detected_peaks)} 个峰值。")
        
    except ImportError as e:
        print("错误：缺少必要的依赖包")
        print("请安装所需包：pip install matplotlib scipy numpy")
        
    except Exception as e:
        print(f"运行时错误：{e}")

