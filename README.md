# PostureGuard / 姿态守护

Android 本地坐姿监测应用，使用 Kotlin、CameraX 和 MediaPipe Pose Landmarker。新版采用浅色薄荷绿界面，支持个人校准、实时姿态指标和持续偏离提醒。

## 改版功能

- 薄荷绿卡片界面、坐姿舒适度圆环与明确的取景引导。
- 正面肩颈平衡、侧面头背姿态两种监测模式，支持切换前后镜头。
- 至少三秒个人坐姿校准，显示实时姿态指标与关键点可信度。
- 仅统计有效观测时间和良好占比，支持声音提醒开关。
- 低置信度过滤、按时间平滑、阈值滞回和异常持续时间确认。

**验证状态：** 当前改版已完成语法、XML 和资源引用的静态检查，尚未编译或进行设备测试。详细范围见 [检查报告](CHECK_REPORT.md)。

## 使用

1. 选择正面或侧面模式，启动监测并允许相机权限。
2. 正面模式需要双耳、双肩入镜；侧面模式需要同侧的耳朵、肩膀和髋部入镜。
3. 稳定放置手机，让镜头接近肩部高度。保持自然坐直，点击“校准坐姿”。
4. 校准需要至少 3 秒、20 帧清晰且稳定的姿态。移动、遮挡或明显倾斜会重新计时。
5. 姿态持续偏离时显示纠正建议，稳定异常状态持续 15 秒后提醒一次。声音可关闭。

切换模式、镜头或身体观察侧会清除个人基准，需要重新校准；暂停保留当前基准，重新开始监测会清空本次统计。退出页面会暂停相机。校准只保留在当前 Activity 生命周期内，模式和声音偏好保存在本机。

画面仅在设备上分析，不保存、不上传。

## 新算法

- **可靠性过滤**：关键点的 visibility 与 presence 均需至少 0.65，且位置必须在画面内。关键点缺失、画面太小或身体方向不适合当前模式时，显示取景引导。
- **比例修正**：将单独归一化的 x、y 坐标换算为相同尺度，避免画面长宽比影响角度。
- **正面模式**：计算有符号的肩部、头部倾斜角，相对个人基准判定。通用进入阈值分别为 5°、8°。
- **侧面模式**：按同侧耳朵、肩膀和髋部的最低置信度选择观察侧；头前伸需水平位移和颈部弯折两个指标同时超限。水平位移按躯干长度归一化，通用阈值为 0.22，颈部弯折阈值为 25°；躯干倾斜阈值为 15°。校准后使用相对基准偏差。
- **平滑和恢复**：使用时间相关 EMA（350 毫秒时间常数），退出阈值为进入阈值的 75%。异常确认需 1.2 秒，恢复确认需 0.8 秒。
- **计时重置**：低置信度、人物离开画面、采样间隔超过 1 秒、暂停或切换模式时，清空异常计时与滤波状态，避免旧结果触发提醒。
- **个人校准**：对稳定样本取中位数。通用合理范围检查用于拒绝明显偏斜的校准姿态；这些数值仍是启发式阈值。
- **舒适度和统计**：舒适度分数是基准偏离程度的 UI 表达；有效监测与良好占比仅统计清晰、可判断的观测时间，不包括校准和取景时间。

原实现依赖三维估计角度。新版使用经画面比例修正的二维几何与个人基准，使用户在固定机位下更容易理解和校准；尚未通过实际数据集比较证明识别准确率提升。

MediaPipe 的坐标与运行模式参考[官方 Android 指南](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android)。所有帧在单独的分析线程中同步处理，相机使用 KEEP_ONLY_LATEST 丢弃积压帧。

## 代码结构

```text
app/src/main/java/com/postureguard/
  MainActivity.kt    # 界面、相机、权限、音频、会话与统计
  PoseAnalyzer.kt    # MediaPipe 帧处理与资源释放
  PostureEngine.kt   # 独立几何、校准、平滑和状态判定
  ScoreRingView.kt   # 舒适度圆环
app/src/main/res/
  layout/activity_main.xml
  drawable/         # 本地矢量插画与图标
  values/           # 薄荷绿主题与文案
tools/check_source.py # 无编译的语法、XML 与资源引用检查
CHECK_REPORT.md      # 本次检查结果与范围
```

## 静态检查（不编译）

基础 XML 与资源检查只需要 Python：

```powershell
python tools/check_source.py
git diff --check
```

Kotlin 语法解析可使用 tree-sitter 和 tree-sitter-kotlin 的二进制 wheel。安装后即可同时检查 Kotlin 语法：

```powershell
python -m pip install --only-binary=:all: tree-sitter tree-sitter-kotlin
python tools/check_source.py
```

如果将解析器安装在单独目录，可传入目录路径：

```powershell
python tools/check_source.py --parser-path <解析器安装目录>
```

脚本不调用 Gradle，不做 Kotlin 类型检查，不运行 Android 应用。详细结果见 [CHECK_REPORT.md](CHECK_REPORT.md)。

## Android 项目

项目保留原有构建配置和依赖版本：AGP 9.1.0、Gradle 9.3.1、Java 21、Android SDK 36.1、CameraX 1.5.3 和 MediaPipe 0.10.32。模型文件已包含在 `app/src/main/assets/pose_landmarker_lite.task`。

在 Android Studio 中打开仓库根目录，并准备 Java 21、Android SDK 36.1 与 Android 8.0（API 26）或更高版本的设备。模型已随源码提供。

界面实际显示、相机兼容性、音频和识别阈值仍需后续设备验证。

## English

PostureGuard monitors sitting posture on-device. The refreshed mint interface provides front/side monitoring, a three-second personal calibration, confidence-gated 2D metrics, time-based smoothing, hysteresis, sustained-state confirmation, and an optional reminder after 15 seconds of confirmed deviation. Source checks are documented in CHECK_REPORT.md; no build or device test was performed.
