#!/usr/bin/env python3
"""
基础测试模块
测试电流事件分析器的基本功能

运行测试:
    python -m pytest tests/
    或
    python tests/test_basic.py
"""

import sys
import os
import unittest
import numpy as np

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

class TestBasicFunctionality(unittest.TestCase):
    """基础功能测试类"""
    
    def setUp(self):
        """测试设置"""
        self.sample_current = np.array([0.0, -0.05, -0.15, -0.25, -0.15, -0.05, 0.0])
        self.sample_time = np.arange(len(self.sample_current)) * 0.001  # 1ms间隔
        self.threshold = -0.1
    
    def test_sample_data_generation(self):
        """测试示例数据生成"""
        self.assertEqual(len(self.sample_current), 7)
        self.assertEqual(len(self.sample_time), 7)
        self.assertAlmostEqual(self.sample_time[-1], 0.006, places=3)
    
    def test_threshold_detection(self):
        """测试阈值检测"""
        below_threshold = self.sample_current < self.threshold
        below_indices = np.where(below_threshold)[0]
        
        # 应该有3个点低于阈值（索引2, 3, 4）
        expected_indices = [2, 3, 4]
        np.testing.assert_array_equal(below_indices, expected_indices)
    
    def test_peak_detection_import(self):
        """测试峰值检测相关导入"""
        try:
            from scipy.signal import find_peaks
            # 简单的峰值检测测试
            peaks, _ = find_peaks(-self.sample_current, height=0.1)
            self.assertIsInstance(peaks, np.ndarray)
        except ImportError:
            self.fail("无法导入scipy.signal.find_peaks")
    
    def test_data_processing(self):
        """测试数据处理功能"""
        # 测试基本的数组操作
        min_current = np.min(self.sample_current)
        max_current = np.max(self.sample_current)
        
        self.assertAlmostEqual(min_current, -0.25, places=2)
        self.assertAlmostEqual(max_current, 0.0, places=2)
    
    def test_relative_calculations(self):
        """测试相对值计算"""
        peak_idx = np.argmin(self.sample_current)  # 最小值的索引
        peak_current = self.sample_current[peak_idx]
        peak_time = self.sample_time[peak_idx]
        
        # 找起点（假设为第一个非零值）
        start_idx = 1  # 索引1处开始下降
        start_current = self.sample_current[start_idx]
        start_time = self.sample_time[start_idx]
        
        # 计算相对值
        amplitude = start_current - peak_current
        relative_time = peak_time - start_time
        
        self.assertGreater(amplitude, 0)  # 振幅应该为正
        self.assertGreater(relative_time, 0)  # 相对时间应该为正


class TestFileHandling(unittest.TestCase):
    """文件处理测试类"""
    
    def test_csv_data_format(self):
        """测试CSV数据格式"""
        # 测试数据文件路径
        test_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_data.csv')
        
        if os.path.exists(test_data_path):
            # 尝试读取测试数据
            try:
                import pandas as pd
                df = pd.read_csv(test_data_path)
                
                # 检查数据结构
                self.assertGreater(len(df), 0)  # 应该有数据
                self.assertIn('column1', df.columns)  # 应该有column1列
                
                # 检查数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                self.assertGreater(len(numeric_cols), 0)  # 应该有数值列
                
            except ImportError:
                self.skipTest("pandas未安装，跳过CSV读取测试")
        else:
            self.skipTest("测试数据文件不存在，跳过CSV读取测试")


class TestDependencies(unittest.TestCase):
    """依赖包测试类"""
    
    def test_numpy_import(self):
        """测试numpy导入"""
        try:
            import numpy as np
            self.assertTrue(hasattr(np, 'array'))
        except ImportError:
            self.fail("无法导入numpy")
    
    def test_scipy_import(self):
        """测试scipy导入"""
        try:
            from scipy.signal import find_peaks
            self.assertTrue(callable(find_peaks))
        except ImportError:
            self.fail("无法导入scipy.signal.find_peaks")
    
    def test_pandas_import(self):
        """测试pandas导入"""
        try:
            import pandas as pd
            self.assertTrue(hasattr(pd, 'DataFrame'))
        except ImportError:
            self.skipTest("pandas未安装，跳过测试")
    
    def test_pyqt5_import(self):
        """测试PyQt5导入"""
        try:
            from PyQt5.QtWidgets import QApplication
            self.assertTrue(QApplication is not None)
        except ImportError:
            self.skipTest("PyQt5未安装，跳过测试")


def run_tests():
    """运行所有测试"""
    print("运行电流事件分析器基础测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [TestBasicFunctionality, TestFileHandling, TestDependencies]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print("\n" + "=" * 50)
    print(f"测试完成！")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

