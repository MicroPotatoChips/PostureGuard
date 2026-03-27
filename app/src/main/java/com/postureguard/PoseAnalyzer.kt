package com.postureguard

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaPlayer
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Landmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt

class PoseAnalyzer(
    private val context: Context,
    private val onResult: (status: String, isBad: Boolean, debugInfo: String) -> Unit
) : ImageAnalysis.Analyzer {

    var isFrontMode: Boolean = true // 对应 MainActivity 的默认选择
    private var poseLandmarker: PoseLandmarker? = null

    // --- 检测可靠性阈值 ---
    private val MIN_VISIBILITY = 0.55f
    private val MIN_PRESENCE = 0.50f
    private val MIN_SHOULDER_WIDTH = 0.04f

    // --- 算法阈值 (比例/角度) ---
    private val THRESHOLD_SHOULDER_TILT = 0.12f // 肩膀高度差/肩宽
    private val THRESHOLD_HEAD_ROLL = 0.13f // 耳朵高度差/肩宽
    private val THRESHOLD_HEAD_FORWARD = 0.45f // 头前伸比例
    private val THRESHOLD_TRUNK_FORWARD = 0.28f // 躯干前倾比例
    private val THRESHOLD_SIDE_ANGLE = 146f // 耳-肩-胯夹角 (越小越驼)

    // --- 滤波平滑 (EMA) ---
    private var emaSideAngle: Float? = null
    private var emaHeadForward: Float? = null
    private var emaTrunkForward: Float? = null

    // --- 报警与状态锁 ---
    private var stableState = "good"
    private var stableIssues: List<String> = emptyList()
    private var pendingState: String? = null
    private var pendingSince = 0L
    private var badPostureStartTime = 0L
    private var hasAlerted = false
    private val mediaPlayer: MediaPlayer by lazy {
        MediaPlayer.create(context, R.raw.sound).apply { isLooping = false }
    }

    init {
        setupPoseLandmarker()
    }

    private fun setupPoseLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .build()
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result, _ -> processPose(result) }
            .build()
        poseLandmarker = PoseLandmarker.createFromOptions(context, options)
    }

    override fun analyze(image: ImageProxy) {
        val rotationDegrees = image.imageInfo.rotationDegrees
        val bitmap = image.toBitmap()

        // 关键：处理 Bitmap 旋转，确保 MediaPipe 看到的是正向的人
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        val mpImage = BitmapImageBuilder(rotatedBitmap).build()
        poseLandmarker?.detectAsync(mpImage, SystemClock.uptimeMillis())
        image.close()
    }

    private fun processPose(result: PoseLandmarkerResult) {
        val landmarks = result.landmarks() // 归一化坐标 (用于计算比例)
        val worldLandmarks = result.worldLandmarks() // 3D 世界坐标 (用于计算角度)

        if (landmarks.isNullOrEmpty() || worldLandmarks.isNullOrEmpty()) {
            onResult("未检测到人", false, "")
            return
        }

        val lm = landmarks[0]
        val wlm = worldLandmarks[0]
        val issues = mutableListOf<String>()
        var debugStr = ""

        if (!isFrontMode) {
            // ================= 侧面模式（混合判据） =================
            val leftScore = reliabilityScore(lm[7]) + reliabilityScore(lm[11]) + reliabilityScore(lm[23])
            val rightScore = reliabilityScore(lm[8]) + reliabilityScore(lm[12]) + reliabilityScore(lm[24])
            val useLeft = leftScore >= rightScore

            val ear = if (useLeft) wlm[7] else wlm[8]
            val shoulder = if (useLeft) wlm[11] else wlm[12]
            val hip = if (useLeft) wlm[23] else wlm[24]

            val ear2d = if (useLeft) lm[7] else lm[8]
            val shoulder2d = if (useLeft) lm[11] else lm[12]
            val hip2d = if (useLeft) lm[23] else lm[24]

            if (!isReliable(ear2d) || !isReliable(shoulder2d) || !isReliable(hip2d)) {
                onResult("请将侧身完整置于画面中", false, "关键点可信度低")
                return
            }

            val sideAngle = calculate3DAngle(ear, shoulder, hip)
            emaSideAngle = smooth(emaSideAngle, sideAngle)

            val torsoLength = max(distance2D(shoulder2d, hip2d), 1e-4f)
            val headForwardRatio = abs(ear2d.x() - shoulder2d.x()) / torsoLength
            val trunkForwardRatio = abs(shoulder2d.x() - hip2d.x()) / torsoLength
            emaHeadForward = smooth(emaHeadForward, headForwardRatio)
            emaTrunkForward = smooth(emaTrunkForward, trunkForwardRatio)

            if ((emaSideAngle ?: sideAngle) < THRESHOLD_SIDE_ANGLE ||
                (emaHeadForward ?: headForwardRatio) > THRESHOLD_HEAD_FORWARD
            ) {
                issues.add("驼背/头前伸")
            }

            if ((emaTrunkForward ?: trunkForwardRatio) > THRESHOLD_TRUNK_FORWARD) {
                issues.add("躯干前倾")
            }

            debugStr = "夹角:${(emaSideAngle ?: sideAngle).toInt()}° 头前伸:${"%.2f".format(emaHeadForward ?: headForwardRatio)} 躯干前倾:${"%.2f".format(emaTrunkForward ?: trunkForwardRatio)}"
        } else {
            // ================= 正面模式 (比例归一化法) =================
            // 计算肩宽作为基准单位 (解决远近误差)
            val shoulderWidth = hypot(lm[11].x() - lm[12].x(), lm[11].y() - lm[12].y())

            if (shoulderWidth < MIN_SHOULDER_WIDTH || !isReliable(lm[11]) || !isReliable(lm[12])) {
                onResult("请保持上半身完整正对镜头", false, "肩部关键点不稳定")
                return
            }

            // 歪肩：y轴差 / 肩宽
            val sDiffRatio = abs(lm[11].y() - lm[12].y()) / shoulderWidth
            // 歪头：耳朵y轴差 / 肩宽
            val eDiffRatio = if (isReliable(lm[7]) && isReliable(lm[8])) {
                abs(lm[7].y() - lm[8].y()) / shoulderWidth
            } else {
                0f
            }

            if (sDiffRatio > THRESHOLD_SHOULDER_TILT) issues.add("歪肩")
            if (eDiffRatio > THRESHOLD_HEAD_ROLL) issues.add("歪头")

            debugStr = "肩偏:${"%.2f".format(sDiffRatio)} 头偏:${"%.2f".format(eDiffRatio)}"
        }

        // --- 状态判定逻辑 ---
        val now = SystemClock.elapsedRealtime()
        val currentState = if (issues.isEmpty()) "good" else "bad"

        if (currentState == stableState) {
            pendingState = null
            pendingSince = 0L
        } else {
            if (pendingState != currentState) {
                pendingState = currentState
                pendingSince = now
            } else if (now - pendingSince >= 1000L) {
                stableState = currentState
                stableIssues = issues.toList()
                pendingState = null
                pendingSince = 0L
            }
        }

        // --- 报警逻辑 ---
        if (stableState == "bad") {
            if (badPostureStartTime == 0L) badPostureStartTime = now
            if ((now - badPostureStartTime) > 15000 && !hasAlerted) { // 15秒提醒
                mediaPlayer.start()
                hasAlerted = true
            }
        } else {
            badPostureStartTime = 0L
            hasAlerted = false
        }

        val visibleIssues = if (stableState == "bad") stableIssues else emptyList()
        val statusText = if (stableState == "good") "姿态良好 ✨" else "${visibleIssues.joinToString("/")} ⚠️"
        onResult(statusText, stableState == "bad", debugStr)
    }

    private fun calculate3DAngle(a: Landmark, b: Landmark, c: Landmark): Float {
        val v1 = floatArrayOf(a.x() - b.x(), a.y() - b.y(), a.z() - b.z())
        val v2 = floatArrayOf(c.x() - b.x(), c.y() - b.y(), c.z() - b.z())
        val dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        val mag1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
        val mag2 = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        if (mag1 < 1e-6f || mag2 < 1e-6f) return 180f

        val cosine = (dot / (mag1 * mag2)).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(cosine).toDouble()).toFloat()
    }

    private fun distance2D(a: Landmark, b: Landmark): Float {
        return hypot(a.x() - b.x(), a.y() - b.y())
    }

    private fun smooth(prev: Float?, now: Float, alpha: Float = 0.3f): Float {
        return if (prev == null) now else prev * (1f - alpha) + now * alpha
    }

    private fun reliabilityScore(lm: Landmark): Float {
        val visibility = lm.visibility().orElse(0f)
        val presence = lm.presence().orElse(0f)
        return 0.6f * visibility + 0.4f * presence
    }

    private fun isReliable(lm: Landmark): Boolean {
        return lm.visibility().orElse(0f) >= MIN_VISIBILITY &&
            lm.presence().orElse(0f) >= MIN_PRESENCE
    }

    fun close() {
        poseLandmarker?.close()
        if (mediaPlayer.isPlaying) mediaPlayer.stop()
        mediaPlayer.release()
    }
}
