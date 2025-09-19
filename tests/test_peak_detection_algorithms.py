#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
峰值检测算法测试脚本
测试三种峰值检测算法：Scipy find_peaks、小波变换、机器学习

运行测试:
    python tests/test_peak_detection_algorithms.py
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

# 导入主分析器中的峰值检测函数
try:
    from main_analyzer import (
        detect_peaks_scipy,
        detect_peaks_wavelet,
        detect_peaks_ml,
        generate_sample_data
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入错误: {e}")
    IMPORT_SUCCESS = False


class TestPeakDetectionAlgorithms(unittest.TestCase):
    """峰值检测算法测试类"""
    
    def setUp(self):
        """测试设置"""
        if not IMPORT_SUCCESS:
            self.skipTest("无法导入主分析器模块")
        
        # 生成示例数据
        self.time, self.current = generate_sample_data()
        self.threshold = -0.1
        
        print(f"生成的示例数据: {len(self.current)} 个数据点")
        print(f"电流范围: {np.min(self.current):.3f} 到 {np.max(self.current):.3f}")
    
    def test_scipy_peak_detection(self):
        """测试Scipy find_peaks算法"""
        print("\n测试Scipy find_peaks算法...")
        
        # 测试基本功能
        peaks = detect_peaks_scipy(self.current, threshold=self.threshold)
        
        self.assertIsInstance(peaks, np.ndarray)
        print(f"检测到 {len(peaks)} 个峰值")
        
        # 验证峰值确实低于阈值
        if len(peaks) > 0:
            peak_currents = self.current[peaks]
            self.assertTrue(np.all(peak_currents < self.threshold))
            print(f"所有峰值均低于阈值 {self.threshold}")
    
    def test_wavelet_peak_detection(self):
        """测试小波变换峰值检测算法"""
        print("\n测试小波变换峰值检测算法...")
        
        try:
            peaks = detect_peaks_wavelet(self.current, threshold=self.threshold)
            
            self.assertIsInstance(peaks, np.ndarray)
            print(f"检测到 {len(peaks)} 个峰值")
            
            # 验证峰值确实低于阈值
            if len(peaks) > 0:
                peak_currents = self.current[peaks]
                self.assertTrue(np.all(peak_currents < self.threshold))
                print(f"所有峰值均低于阈值 {self.threshold}")
                
        except ImportError as e:
            if "pywt" in str(e).lower():
                self.skipTest("PyWavelets未安装，跳过小波变换测试")
            else:
                raise
    
    def test_ml_peak_detection(self):
        """测试机器学习峰值检测算法"""
        print("\n测试机器学习峰值检测算法...")
        
        try:
            peaks = detect_peaks_ml(self.current, threshold=self.threshold)
            
            self.assertIsInstance(peaks, np.ndarray)
            print(f"检测到 {len(peaks)} 个峰值")
            
            # 验证峰值确实低于阈值
            if len(peaks) > 0:
                peak_currents = self.current[peaks]
                self.assertTrue(np.all(peak_currents < self.threshold))
                print(f"所有峰值均低于阈值 {self.threshold}")
                
        except ImportError as e:
            if "sklearn" in str(e).lower() or "scikit" in str(e).lower():
                self.skipTest("scikit-learn未安装，跳过机器学习测试")
            else:
                raise
    
    def test_algorithm_comparison(self):
        """测试不同算法的一致性"""
        print("\n测试不同算法的一致性...")
        
        # 获取所有可用算法的结果
        results = {}
        
        # Scipy
        scipy_peaks = detect_peaks_scipy(self.current, threshold=self.threshold)
        results['scipy'] = set(scipy_peaks)
        print(f"Scipy检测到 {len(scipy_peaks)} 个峰值")
        
        # Wavelet (如果可用)
        try:
            wavelet_peaks = detect_peaks_wavelet(self.current, threshold=self.threshold)
            results['wavelet'] = set(wavelet_peaks)
            print(f"小波变换检测到 {len(wavelet_peaks)} 个峰值")
        except ImportError:
            print("小波变换不可用")
        
        # ML (如果可用)
        try:
            ml_peaks = detect_peaks_ml(self.current, threshold=self.threshold)
            results['ml'] = set(ml_peaks)
            print(f"机器学习检测到 {len(ml_peaks)} 个峰值")
        except ImportError:
            print("机器学习不可用")
        
        # 比较算法结果
        if len(results) > 1:
            algorithms = list(results.keys())
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    algo1 = algorithms[i]
                    algo2 = algorithms[j]
                    
                    common_peaks = results[algo1] & results[algo2]
                    print(f"{algo1} 和 {algo2} 共同检测到 {len(common_peaks)} 个峰值")
                    
                    # 至少应该有部分一致性
                    self.assertGreater(len(common_peaks), 0, 
                                     f"算法 {algo1} 和 {algo2} 没有检测到共同的峰值")
    
    def test_missing_dependencies(self):
        """测试缺失依赖时的错误处理"""
        print("\n测试缺失依赖时的错误处理...")
        
        # 模拟缺失PyWavelets
        with patch.dict('sys.modules', {'pywt': None}):
            with self.assertRaises(ImportError) as context:
                detect_peaks_wavelet(self.current, threshold=self.threshold)
            self.assertIn('pywt', str(context.exception).lower())
            print("✓ 正确检测到PyWavelets缺失")
        
        # 模拟缺失scikit-learn
        with patch.dict('sys.modules', {'sklearn': None}):
            with self.assertRaises(ImportError) as context:
                detect_peaks_ml(self.current, threshold=self.threshold)
            self.assertIn('sklearn', str(context.exception).lower())
            print("✓ 正确检测到scikit-learn缺失")


class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def setUp(self):
        """性能测试设置"""
        if not IMPORT_SUCCESS:
            self.skipTest("无法导入主分析器模块")
        
        # 生成更大的数据集进行性能测试
        self.large_time = np.linspace(0, 10, 10000)  # 10秒，10000个点
        self.large_current = np.random.normal(0, 0.1, 10000)
        
        # 添加一些明显的峰值
        peak_times = [1.0, 3.0, 5.0, 7.0, 9.0]
        for peak_time in peak_times:
            peak_idx = int(peak_time * 1000)
            width = 50
            start_idx = max(0, peak_idx - width)
            end_idx = min(len(self.large_current), peak_idx + width)
            self.large_current[start_idx:end_idx] -= 0.5 * np.exp(
                -((np.arange(start_idx, end_idx) - peak_idx) ** 2) / (width ** 2)
            )
    
    def test_scipy_performance(self):
        """测试Scipy算法性能"""
        import time
        
        start_time = time.time()
        peaks = detect_peaks_scipy(self.large_current, threshold=-0.2)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"Scipy处理 {len(self.large_current)} 个数据点耗时: {processing_time:.3f}秒")
        print(f"检测到 {len(peaks)} 个峰值")
        
        # 性能应该合理
        self.assertLess(processing_time, 1.0, "Scipy算法处理时间过长")
    
    def test_wavelet_performance(self):
        """测试小波变换算法性能"""
        try:
            import time
            
            start_time = time.time()
            peaks = detect_peaks_wavelet(self.large_current, threshold=-0.2)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"小波变换处理 {len(self.large_current)} 个数据点耗时: {processing_time:.3f}秒")
            print(f"检测到 {len(peaks)} 个峰值")
            
            # 性能应该合理
            self.assertLess(processing_time, 2.0, "小波变换算法处理时间过长")
            
        except ImportError:
            self.skipTest("PyWavelets未安装，跳过小波变换性能测试")
    
    def test_ml_performance(self):
        """测试机器学习算法性能"""
        try:
            import time
            
            start_time = time.time()
            peaks = detect_peaks_ml(self.large_current, threshold=-0.2)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"机器学习处理 {len(self.large_current)} 个数据点耗时: {processing_time:.3f}秒")
            print(f"检测到 {len(peaks)} 个峰值")
            
            # 性能应该合理
            self.assertLess(processing_time, 3.0, "机器学习算法处理时间过长")
            
        except ImportError:
            self.skipTest("scikit-learn未安装，跳过机器学习性能测试")


def run_algorithm_tests():
    """运行算法测试"""
    print("运行峰值检测算法测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [TestPeakDetectionAlgorithms, TestPerformance]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print("\n" + "=" * 50)
    print("算法测试完成！")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_algorithm_tests()
    sys.exit(0 if success else 1)