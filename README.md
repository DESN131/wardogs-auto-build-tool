# WARDOGS 自动建造工具

项目代号：`wardogs-auto-build`

面向 Windows + 无边框窗口的 WARDOGS 自动建造工具：识别黄色建造叉，移动准星，按住左键起锤，并在抡锤期间继续瞄准。默认使用“确认新黄叉后再起锤”的稳定模式，也可以通过 `--chain` 启用链式挥锤。

> 自动输入可能违反游戏服务条款。请先在离线/私人环境验证，不要直接用于正式账号。本项目不修改游戏文件、不读取游戏内存、不绕过反作弊。

## 快速开始

### 首次安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 直接运行

```powershell
.\.venv\Scripts\python.exe auto_build.py
```

进入游戏后：

| 键 | 作用 |
|---|---|
| `F8` | 开始 / 暂停 / 恢复 |
| `F9` | 停止并退出 |

默认会在确认新黄叉后开始下一锤，适合稳定连续建造。

### 常用可选命令

```powershell
# 只检测，不发送鼠标输入
.\.venv\Scripts\python.exe auto_build.py --dry-run --verbose

# 只建一次，确认完整流程
.\.venv\Scripts\python.exe auto_build.py --max-builds 1

# 带逐步计时日志运行
.\.venv\Scripts\python.exe auto_build.py --debug 2>&1 |
  Tee-Object -FilePath logs/debug_run.log

# 链式挥锤实验模式
.\.venv\Scripts\python.exe auto_build.py --chain
```

## 功能

- 黄色/白色建造叉检测，兼容不同建筑和视角。
- 两级快速检测：大画面降采样找候选，再回全分辨率局部确认。
- 拟人化鼠标移动：弯曲路径、随机抖动、短延迟和末端过冲修正。
- 510 ms 起锤，前约 450 ms 继续瞄准，最后 60 ms 停止移动。
- 白叉锁定、新黄叉验证、陈旧白叉过滤。
- 支持连续建造、按建造数/真实落锤数停止、离线截图验证。

## 项目结构

```text
wardogs-auto-build/
├─ auto_build.py            # 启动入口
├─ wardogs/
│  ├─ auto_build.py         # 主循环、状态机、CLI、日志
│  ├─ detector.py           # 黄色/白色建造叉检测
│  └─ win_input.py          # SendInput、热键、鼠标轨迹
├─ config.yaml              # 运行配置
├─ requirements.txt
├─ docs/方案.md             # 设计与实验记录
├─ assets/                  # 本地验证截图（不提交）
├─ tests/                   # 单元测试
├─ logs/                    # 运行日志（不提交）
├─ artifacts/               # 调试图/标注图（不提交）
└─ .gitignore
```

## 建造模式

### 默认模式

`swing.chain: false`。每次落锤后先确认新黄叉，再进入下一锤；每锤建成率更高、波动更小。

### 链式模式

```powershell
.\.venv\Scripts\python.exe auto_build.py --chain
```

- 松开左键后立即起下一锤；
- 在 510 ms 抡锤期间等待并瞄准新黄叉；
- 用新黄叉相对预测旧位置的位移判断上一锤是否建成；
- 未瞄准落锤仍会获得约 1/5 建造进度；
- `chain_probe_ms: 450` 保证这一锤能落锤；设为 200 会在看不到新叉时提前松开，但那是 0 进度。

实测中链式每锤更快，但每锤建成率更低；综合效率旧模式更稳定，因此默认仍是旧模式。

### 停止条件

```powershell
# 完成 30 次已验证建造后退出
.\.venv\Scripts\python.exe auto_build.py --max-builds 30

# 完成 30 次真实落锤后退出（提前松开不计）
.\.venv\Scripts\python.exe auto_build.py --max-swings 30
```

退出时会打印真实落锤次数、提前松开次数、建造阶段耗时、`ms/件` 和 `ms/锤`。

## 检测方案

当前检测使用“颜色 + 形状 + 白叉锁定”的经典方案：

1. 启动/丢失目标时全屏搜索；瞄准和点击后使用中心 `1000×1000` ROI。
2. BGR→HSV，黄色范围 `H 22–36 / S 130–255 / V 110–255`，再用色度约束筛选。
3. 形态学去噪、连通域、面积/宽高/填充率/宽高比过滤。
4. 多个候选时取离屏幕中心最近的一个。
5. 中心白叉优先于黄色残影。
6. `detector.fast_mode: true`：
   - 大画面先 2 倍降采样找黄色候选；
   - 回全分辨率候选附近精确确认；
   - 白叉锁定只在中心 320×320 小图确认；
   - 找不到目标或需要完整白叉候选时自动回退到完整 detector。

实测纯检测耗时：

| 场景 | 完整 detector | 快速检测 |
|---|---:|---:|
| 全屏 2560×1440 黄叉 | 80–130 ms | **约 10 ms** |
| 1000×1000 ROI | 约 27–37 ms | **约 3–11 ms** |
| 白叉中心锁定 | 约 100–130 ms | **约 9–12 ms** |

抓屏仍由 `mss` 决定：ROI 约 8–15 ms，全屏约 25–45 ms。

## 离线验证

验证不要求固定文件名。把截图放进任意目录，脚本会自动发现 `png/jpg/jpeg/bmp/webp` 并按预期判定：

```powershell
# 目录内每张都应有黄叉
.\.venv\Scripts\python.exe auto_build.py --test-dir my_shots --expect yellow

# 目录内每张都应白叉锁定
.\.venv\Scripts\python.exe auto_build.py --test-dir my_white_shots --expect white

# 只报告结果，不要求每张都有目标
.\.venv\Scripts\python.exe auto_build.py --test-dir my_shots --expect any

# 递归搜索子目录；可同时传多个目录
.\.venv\Scripts\python.exe auto_build.py `
  --test-dir my_shots my_white_shots --recursive --expect any
```

每行输出 `PASS/FAIL`，末尾输出：

```text
Summary: images=6 yellow=6 white=0 empty=0 failed=0 expect=yellow
```

有失败、文件无法读取或目录中没有图片时返回非零退出码。

项目自带的回归截图位于 `assets/`，默认不提交；需要时在本地放置，然后运行：

```powershell
.\.venv\Scripts\python.exe auto_build.py --test-dir assets/validate --expect yellow
.\.venv\Scripts\python.exe auto_build.py --test-dir assets/validate/white --expect white
```

## Debug 日志

```powershell
.\.venv\Scripts\python.exe auto_build.py --debug --max-swings 30 2>&1 |
  Tee-Object -FilePath logs/debug_run30_old.log

.\.venv\Scripts\python.exe auto_build.py --debug --max-swings 30 --chain 2>&1 |
  Tee-Object -FilePath logs/debug_run30_chain.log
```

主要标签：

- `DET`：抓屏、检测耗时、ROI、黄/白候选、锁定、V/S、bbox、误差；
- `AIM` / `CALIB`：移动、稳定、counts、增益；
- `SWING`：抡锤期间的瞄准次数、剩余时间、快速补瞄；
- `CHAIN`：链式立即起锤、新目标位移、1/5 部分进度；
- `WHITE`：白叉确认；
- `CLICK`：点击前/按住/点击后耗时；
- `VERIFY`：新黄叉轮询和位移；
- `BUILD`：建造完成计数。

## 关键配置

完整参数见 [config.yaml](config.yaml)。

| 参数 | 说明 |
|---|---|
| `game.window_title` | 默认 `WARDOGS`，无需修改 |
| `detector.fast_mode` | 两级快速检测，默认 `true` |
| `detector.roi_half` | 瞄准/轮询 ROI 半径，默认 500 |
| `click.hold_ms` | 左键保持时间，默认 510 ms |
| `click.poll_ms` | 新黄叉轮询间隔，默认 10 ms |
| `swing.chain` | 链式挥锤，默认 `false` |
| `swing.chain_probe_ms` | 链式新叉探测时间，默认 450 ms |
| `aim.gain_x/gain_y` | 屏幕像素/鼠标 counts 增益 |
| `human.*` | 弯曲、抖动、延迟、过冲参数 |
| `miss.every_n_builds` | 每 N 次建造插入一次非瞄准点击，0 关闭 |

## 测试与开发

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  auto_build.py wardogs\auto_build.py wardogs\detector.py wardogs\win_input.py

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 28 项测试通过。缺少本地 `assets/` 截图时，依赖素材的回归会自动 skip，不会导致测试失败。

## 常见问题

- **截到黑屏**：使用无边框窗口，不要用独占全屏。
- **鼠标/左键无效**：脚本与游戏保持相同权限级别；先运行 `--test-click 3` 在记事本验证输入。
- **F8 有时不能暂停**：已使用按下锁存位，暂停会在当前挥锤/检测步骤结束后生效。
- **黄叉变白后停住**：检查白叉阈值和 `WHITE` 日志，HDR 可能改变亮度/饱和度。
- **原地挥锤**：检查日志中的 `stale=True` 或 `AIM stale lock -> real target`。
- **落锤后有空档**：看 `DET detect_ms`、`VERIFY` 和 `WAIT`；快速检测下 ROI 通常低于约 12 ms。
- **DPI/画面缩放错位**：确认客户区尺寸与游戏渲染尺寸一致。

