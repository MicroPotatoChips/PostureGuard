# PostureGuard / 姿态守护

PostureGuard is an Android app that monitors sitting posture in real time using MediaPipe pose landmarks and provides visual/audio reminders when posture degrades.

PostureGuard 是一个基于 Android 的实时坐姿监测应用，使用 MediaPipe 姿态关键点进行识别，并在姿态持续不良时提供界面与声音提醒。

---

## Features / 功能

### English
- Real-time posture analysis from camera frames.
- Front-view mode: detects shoulder tilt and head tilt.
- Side-view mode: detects hunching / forward-head and trunk lean.
- Smoothing + debounce to reduce jitter and false alerts.
- Audio alert after prolonged bad posture.

### 中文
- 基于摄像头画面的实时姿态分析。
- 正面模式：检测歪肩、歪头。
- 侧面模式：检测驼背/头前伸、躯干前倾。
- 使用平滑与去抖逻辑，减少抖动和误报。
- 不良姿态持续一段时间后触发语音提醒。

---

## Tech Stack / 技术栈

### English
- Kotlin
- Android CameraX
- Google MediaPipe Tasks (Pose Landmarker)

### 中文
- Kotlin
- Android CameraX
- Google MediaPipe Tasks（Pose Landmarker）

---

## Project Structure / 项目结构

```text
app/src/main/java/com/postureguard/
  MainActivity.kt      # Camera, UI, mode switching / 相机与界面控制
  PoseAnalyzer.kt      # Core posture algorithm / 姿态算法核心
```

---

## How to Run / 运行方式

### English
1. Open the project in Android Studio.
2. Ensure `pose_landmarker_lite.task` exists in `app/src/main/assets/`.
3. Build and run on an Android device.
4. Grant camera permission.
5. Select front/side mode and start monitoring.

### 中文
1. 使用 Android Studio 打开项目。
2. 确认 `app/src/main/assets/` 下存在 `pose_landmarker_lite.task`。
3. 编译并运行到 Android 设备。
4. 授予相机权限。
5. 选择正面/侧面模式并开始监控。

---

## Algorithm Notes / 算法说明

### English
- The analyzer first checks landmark reliability (visibility/presence).
- Front mode uses normalized shoulder/head roll ratios.
- Side mode combines:
  - 3D ear-shoulder-hip angle,
  - 2D normalized forward-head ratio,
  - 2D normalized trunk-lean ratio.
- EMA smoothing and 1-second state debounce improve stability.

### 中文
- 识别前先进行关键点可靠性判断（visibility/presence）。
- 正面模式使用归一化的肩部/头部偏斜比例。
- 侧面模式结合以下指标：
  - 3D 耳-肩-胯夹角，
  - 2D 头前伸归一化比例，
  - 2D 躯干前倾归一化比例。
- 通过 EMA 平滑与 1 秒状态去抖提升稳定性。

---

## Roadmap / 后续计划

### English
- Per-user calibration for thresholds.
- Better UX for guidance and recovery actions.
- Historical trend reports and weekly summaries.

### 中文
- 加入个体化阈值校准。
- 优化纠正引导与恢复动作提示。
- 增加历史趋势统计与周报。

---

## License / 许可证

### English
No license file is currently included. Add one before public distribution.

### 中文
当前仓库未包含许可证文件，公开发布前建议补充。
