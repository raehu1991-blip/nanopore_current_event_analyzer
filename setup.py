#!/usr/bin/env python3
"""
Current Event Analyzer 安装配置
电流事件分析器的安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join("docs", "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Current Event Analyzer - 电流事件分析器"

# 读取requirements文件
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="current-event-analyzer",
    version="1.0.0",
    description="电流事件分析器 - 基于PyQt5的TDMS和NPZ文件电流数据分析工具",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Current Event Analyzer Team",
    author_email="",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "current-analyzer=main_analyzer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="current analysis, tdms, npz, pyqt5, data visualization, peak detection",
    project_urls={
        "Documentation": "",
        "Source": "",
        "Tracker": "",
    },
)

