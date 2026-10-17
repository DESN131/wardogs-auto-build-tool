# 本地验证截图

此目录中的截图用于开发验证，默认不提交到 git。

需要运行离线检测或单元测试时，请在本地放置：

```text
assets/reference/1.png
assets/reference/2.png
assets/reference/3.png
assets/reference/4.png
assets/validate/*.png
assets/validate/white/*.png
```

然后从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe auto_build.py --test-dir assets/validate --expect yellow
.\.venv\Scripts\python.exe auto_build.py --test-dir assets/validate/white --expect white
.\.venv\Scripts\python.exe auto_build.py --test-dir assets/reference --expect any
```

也可以把自己的截图放进任意目录，用 `--test-dir <目录> --expect yellow|white|any` 验证；脚本支持 png/jpg/jpeg/bmp/webp 和 `--recursive`。截图仅作为本地回归数据；`tests/` 中的测试代码仍然由 git 管理。
