# GitHub 发布指南

## 📋 发布到GitHub的完整步骤

### 步骤1：准备GitHub账户和Git

1. **创建GitHub账户**
   - 访问 [github.com](https://github.com) 注册账户
   - 验证邮箱地址

2. **安装Git**
   - 下载并安装 [Git for Windows](https://git-scm.com/download/win)
   - 安装完成后重启命令行
   - 验证安装：`git --version`

3. **配置Git用户信息**
   ```bash
   git config --global user.name "你的用户名"
   git config --global user.email "你的邮箱@example.com"
   ```

### 步骤2：创建GitHub仓库

1. **在GitHub上创建新仓库**
   - 登录GitHub，点击右上角 "+" 按钮
   - 选择 "New repository"
   - 仓库名建议：`current-event-analyzer`
   - 描述：`电流事件分析器 - 基于PyQt5的TDMS和NPZ文件分析工具`
   - 设为Public（公开）或Private（私有）
   - ✅ **不要**初始化README、.gitignore或LICENSE（我们已经创建了）
   - 点击 "Create repository"

### 步骤3：初始化本地Git仓库

在项目目录下打开命令行，执行以下命令：

```bash
# 初始化Git仓库
git init

# 添加所有文件到版本控制
git add .

# 查看文件状态
git status

# 提交第一个版本
git commit -m "Initial commit: 电流事件分析器 v1.0.0

- 添加单事件分析功能
- 添加多文件可视化功能  
- 支持TDMS和NPZ文件格式
- 完整的项目结构和文档
- PyQt5图形界面"

# 添加远程仓库（替换为你的GitHub用户名）
git remote add origin https://github.com/你的用户名/current-event-analyzer.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

### 步骤4：GitHub仓库设置

1. **添加仓库描述和标签**
   - 在GitHub仓库页面点击右上角的设置图标（⚙️）
   - 添加描述：`电流事件分析器 - 基于PyQt5的TDMS和NPZ文件分析工具`
   - 添加标签：`python`, `pyqt5`, `data-analysis`, `tdms`, `peak-detection`, `scientific-computing`

2. **设置GitHub Pages（可选）**
   - Settings → Pages
   - Source: Deploy from a branch
   - Branch: main / docs
   - 启用后可通过URL访问文档

3. **添加项目logo（可选）**
   - 在仓库根目录添加 `logo.png` 文件
   - 在README中引用：`![Logo](logo.png)`

### 步骤5：发布Release版本

1. **创建第一个Release**
   - 在GitHub仓库页面点击 "Releases"
   - 点击 "Create a new release"
   - Tag version: `v1.0.0`
   - Release title: `Current Event Analyzer v1.0.0`
   - 描述发布内容（参考下面的模板）
   - 选择 "Set as the latest release"
   - 点击 "Publish release"

**Release描述模板：**
```markdown
## 🎉 Current Event Analyzer v1.0.0

### ✨ 新功能
- 🔬 单事件分析模块
- 📊 多文件数据可视化
- 🎯 智能峰值检测算法
- 📱 现代化PyQt5界面

### 🚀 核心特性
- 支持TDMS和NPZ文件格式
- 交互式数据可视化
- 可调参数的峰值检测
- CSV数据导出功能
- 跨平台兼容性

### 📋 系统要求
- Python 3.7+
- PyQt5 5.15.0+
- Windows/macOS/Linux

### 🚀 快速开始
1. 下载源代码：`git clone https://github.com/你的用户名/current-event-analyzer.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行程序：`python run.py`

### 📚 文档
- [用户手册](docs/README.md)
- [需求规格](docs/requirements.md)
- [API文档](docs/api.md)
```

### 步骤6：持续维护

1. **更新代码**
   ```bash
   # 修改代码后
   git add .
   git commit -m "描述你的更改"
   git push origin main
   ```

2. **版本管理**
   ```bash
   # 创建新版本标签
   git tag v1.0.1
   git push origin v1.0.1
   ```

3. **分支管理**
   ```bash
   # 创建功能分支
   git checkout -b feature/new-feature
   # 开发完成后合并
   git checkout main
   git merge feature/new-feature
   ```

### 步骤7：增强项目可见性

1. **添加徽章到README**
   - 参考README.md中的徽章示例
   - 可以添加构建状态、覆盖率等徽章

2. **创建详细文档**
   - API文档
   - 用户手册
   - 开发者指南
   - 贡献指南

3. **添加示例和截图**
   - 在docs/images/中添加界面截图
   - 创建使用示例视频
   - 添加数据样本

4. **社区功能**
   - 启用Issues用于bug报告
   - 启用Discussions用于讨论
   - 创建贡献指南
   - 添加行为准则

## 🛠️ 故障排除

### Git命令不存在
```bash
# Windows: 下载并安装Git for Windows
# macOS: brew install git  
# Ubuntu: sudo apt-get install git
```

### 推送失败（认证问题）
```bash
# 使用个人访问令牌
# GitHub Settings → Developer settings → Personal access tokens
# 生成token并用作密码
```

### 文件太大
```bash
# 使用Git LFS处理大文件
git lfs install
git lfs track "*.tdms"
git add .gitattributes
```

## 📚 相关资源

- [Git官方文档](https://git-scm.com/doc)
- [GitHub帮助文档](https://docs.github.com)
- [Markdown语法指南](https://guides.github.com/features/mastering-markdown/)
- [开源许可证选择](https://choosealicense.com/)

## 🎯 下一步

1. 完善项目文档
2. 添加单元测试
3. 设置CI/CD流程
4. 创建用户反馈渠道
5. 考虑发布到PyPI

---

🎉 恭喜！你的项目现在已经在GitHub上了！
