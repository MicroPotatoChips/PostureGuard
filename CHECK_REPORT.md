# PostureGuard 改版检查

检查日期：2026-10-01。代码来源：MicroPotatoChips/PostureGuard，master 的 eeec194368138da8a3459b4ca7cf4afb8cdab319。改版分支：feat/mint-ui-posture-engine。

按用户要求仅做静态检查，未编译、未执行 Android 应用、未生成 APK，未运行 JUnit、Lint 或设备测试。开始检查前只运行过 Gradle 的 --version 命令，用于查看工具信息，没有执行任何构建任务。

## 自动检查

| 检查项目 | 结果 |
| --- | --- |
| Kotlin 语法树解析 | 6 个源文件通过，包括 4 个应用源文件 |
| Android XML 解析 | 19 个 XML 文件通过 |
| 本地资源与代码引用 | 124 个资源名、26 个主界面视图 ID 检查通过 |
| 重复资源名与主界面重复 ID | 未发现 |
| 模型路径与文件存在性 | pose_landmarker_lite.task 存在 |
| git diff --check | 通过 |

检查脚本：tools/check_source.py。Kotlin 解析采用 tree-sitter 0.26.0 和 tree-sitter-kotlin 1.1.0，只读取语法，不执行 Kotlin 编译或类型检查。Material 组件样式、开关类和 MediaPipe 调用方式另以官方源码/文档核对，见下方参考。

## 人工逻辑检查与修正

- 关键点不清晰或人物离开画面时，清空平滑、异常确认和提醒计时，界面不显示“姿态良好”。
- 用画面长宽比修正二维角度；侧面观察侧按耳、肩、髋最低置信度选择，加入切换差值，避免反复跳侧。
- 保留观察侧对应的个人基准信息；换侧或换镜头时清除旧基准，避免镜像方向导致错误偏差。
- 初始异常确认阶段显示“正在确认坐姿”，避免在异常确认的 1.2 秒内显示良好。
- 校准只收集稳定、可信的样本，丢失画面、移动或采样间隔过长会重新计时；校准期间不会触发异常提醒。
- 同一段确认异常只发出一次提醒事件；恢复确认、离开画面、暂停与模式切换会重置提醒状态。
- 相机启动、切换模式、校准与暂停使用会话编号过滤旧回调；启动期间切换模式会取消旧启动并重新请求，避免按钮卡住。
- 后台退出时关闭相机、清空分析器与暂停声音；分析、重置、模型创建和释放排在同一个分析线程上。
- 修正 MPImage.close() 后访问位图尺寸的问题：尺寸在释放前保存；最终清理检查位图是否已回收。[MediaPipe 位图容器源码](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/java/com/google/mediapipe/framework/image/BitmapImageContainer.java)
- 页面使用滚动容器并处理系统栏与刘海区域。声音偏好和检测模式保存在本机，个人基准保留在本次页面生命周期内。

## 检查范围的限制

上述结果确认语法结构、资源连接和人工检查的分支逻辑。由于未编译，尚未确认 Kotlin 类型、完整依赖解析与 Android 资源链接；由于未运行，也未验证不同屏幕和大字体的实际布局、镜头设备兼容性、生命周期时序及声音播放。

算法是几何与个人基准结合的启发式实现。阈值和实际识别准确率尚未经过真实坐姿数据集或设备实验验证。原有 ExampleUnitTest 与 ExampleInstrumentedTest 为模板测试，不能作为新算法验证结果。

## 核对参考

- [MediaPipe Android Pose Landmarker 指南](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android)：VIDEO 模式、单独线程和 detectForVideo 调用。
- [MediaPipe Java PoseLandmarker API](https://developers.google.com/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarker)：递增时间戳与方法签名。
- [CameraX 图像分析](https://developer.android.com/media/camera/camerax/analyze)：生命周期绑定、KEEP_ONLY_LATEST 和 ImageProxy 关闭。
- [Material 1.10.0 按钮样式](https://github.com/material-components/material-components-android/blob/1.10.0/lib/java/com/google/android/material/button/res/values/styles.xml)：本次使用的 Material3 样式存在。
- [Material 1.10.0 MaterialSwitch](https://github.com/material-components/material-components-android/blob/1.10.0/lib/java/com/google/android/material/materialswitch/MaterialSwitch.java)：开关类存在。
