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
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

class PoseAnalyzer(
    private val context: Context,
    private val onResult: (status: String, isBad: Boolean, debugInfo: String) -> Unit
) : ImageAnalysis.Analyzer {

    var isFrontMode: Boolean = false // 对应 MainActivity 的默认选择
    @Volatile private var isClosed = false
    private var poseLandmarker: PoseLandmarker? = null
    private var lastInferenceTimestampMs = 0L

    // --- 算法阈值（比例/角度） ---
    private val THRESHOLD_SHOULDER_TILT = 0.08f // 肩膀高度差/肩宽
    private val THRESHOLD_HEAD_TILT = 0.06f // 头部高度差/肩宽
    private val THRESHOLD_SIDE_ANGLE = 142f // 驼背角度（越小越驼）
    private val THRESHOLD_TORSO_TILT = 160f // 躯干前倾角度

    // --- 滤波平滑（EMA） ---
    private var emaAngle: Float? = null
    private var emaTorso: Float? = null

    // --- 报警与状态锁 ---
    private var lastState = "good"
    private var lastStateTime = 0L
    private var lastBadIssues: List<String> = emptyList()
    private var badPostureStartTime = 0L
    private var hasAlerted = false

    private var mediaPlayer: MediaPlayer? = null

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
        if (isClosed) {
            image.close()
            return
        }

        try {
            val rotationDegrees = image.imageInfo.rotationDegrees
            val bitmap = image.toBitmap()

            // 关键：处理 Bitmap 旋转，确保 MediaPipe 看到的是正向人体。
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

            val timestamp = SystemClock.uptimeMillis().let { now ->
                if (now <= lastInferenceTimestampMs) lastInferenceTimestampMs + 1 else now
            }
            lastInferenceTimestampMs = timestamp

            val mpImage = BitmapImageBuilder(rotatedBitmap).build()
            poseLandmarker?.detectAsync(mpImage, timestamp)
        } catch (_: Exception) {
            // 吞掉单帧异常，避免分析链路崩溃。
        } finally {
            image.close()
        }
    }

    private fun processPose(result: PoseLandmarkerResult) {
        if (isClosed) return

        val landmarks = result.landmarks()
        val worldLandmarks = result.worldLandmarks()

        val lm = landmarks.firstOrNull()
        val wlm = worldLandmarks.firstOrNull()
        if (lm.isNullOrEmpty() || wlm.isNullOrEmpty()) {
            onResult("未检测到人", false, "")
            return
        }

        val issues = mutableListOf<String>()
        var debugStr = ""

        if (!isFrontMode) {
            // ================= 侧面模式（3D 向量法） =================
            val leftShoulder2d = lm.getOrNull(11)
            val rightShoulder2d = lm.getOrNull(12)
            if (leftShoulder2d == null || rightShoulder2d == null) {
                onResult("未检测到人", false, "")
                return
            }

            val useLeft = (leftShoulder2d.visibility().orElse(0f) >= rightShoulder2d.visibility().orElse(0f))

            val ear = (if (useLeft) wlm.getOrNull(7) else wlm.getOrNull(8))
            val shoulder = (if (useLeft) wlm.getOrNull(11) else wlm.getOrNull(12))
            val hip = (if (useLeft) wlm.getOrNull(23) else wlm.getOrNull(24))

            if (ear == null || shoulder == null || hip == null) {
                onResult("请将上半身完整置于画面中", false, "肩部关键点缺失")
                return
            }

            // 1) 3D 骨骼夹角（耳-肩-胯）
            val angle = calculate3DAngle(ear, shoulder, hip)
            emaAngle = if (emaAngle == null) angle else emaAngle!! * 0.7f + angle * 0.3f

            // 2) 躯干垂直倾斜度
            val dx = shoulder.x() - hip.x()
            val dy = shoulder.y() - hip.y()
            val torsoTilt = abs(Math.toDegrees(atan2(dx.toDouble(), dy.toDouble()))).toFloat()
            emaTorso = if (emaTorso == null) torsoTilt else emaTorso!! * 0.7f + torsoTilt * 0.3f

            if ((emaAngle ?: angle) < THRESHOLD_SIDE_ANGLE) issues.add("⚠️驼背")
            if ((emaTorso ?: torsoTilt) < THRESHOLD_TORSO_TILT) issues.add("⚠️背前倾")

            debugStr = "角度:${(emaAngle ?: angle).toInt()}° 倾斜:${(emaTorso ?: torsoTilt).toInt()}°"
        } else {
            // ================= 正面模式（比例归一化法） =================
            val leftShoulder = lm.getOrNull(11)
            val rightShoulder = lm.getOrNull(12)
            if (leftShoulder == null || rightShoulder == null) {
                onResult("请保持正对镜头", false, "肩部关键点缺失")
                return
            }

            val shoulderWidth = hypot(leftShoulder.x() - rightShoulder.x(), leftShoulder.y() - rightShoulder.y())
            if (shoulderWidth < 1e-4f) {
                onResult("请保持正对镜头", false, "肩宽过小")
                return
            }

            // 歪肩：y 轴差 / 肩宽
            val sDiffRatio = abs(leftShoulder.y() - rightShoulder.y()) / shoulderWidth
            // 歪头：耳朵 y 轴差 / 肩宽（耳朵关键点缺失则不判定歪头）
            val leftEar = lm.getOrNull(7)
            val rightEar = lm.getOrNull(8)
            val eDiffRatio = if (leftEar != null && rightEar != null) {
                abs(leftEar.y() - rightEar.y()) / shoulderWidth
            } else {
                0f
            }

            if (sDiffRatio > THRESHOLD_SHOULDER_TILT) issues.add("⚠️歪肩")
            if (eDiffRatio > THRESHOLD_HEAD_TILT) issues.add("⚠️歪头")

            debugStr = "歪肩:${"%.2f".format(sDiffRatio)} 头偏:${"%.2f".format(eDiffRatio)}"
        }

        // --- 状态判定逻辑 ---
        val now = SystemClock.elapsedRealtime()
        val currentState = if (issues.isEmpty()) "good" else "bad"

        // 状态抖动保护：只有持续 1 秒的状态改变才生效。
        if (currentState != lastState) {
            if (lastStateTime == 0L) {
                lastStateTime = now
            } else if (now - lastStateTime > 1000L) {
                lastState = currentState
                lastStateTime = 0L
                if (lastState == "good") {
                    lastBadIssues = emptyList()
                }
            }
        } else {
            lastStateTime = 0L
        }

        if (currentState == "bad") {
            lastBadIssues = issues.toList()
        }

        // --- 报警逻辑 ---
        if (lastState == "bad") {
            if (badPostureStartTime == 0L) badPostureStartTime = now
            if ((now - badPostureStartTime) > 15000L && !hasAlerted) { // 15 秒提醒
                playAlertSound()
                hasAlerted = true
            }
        } else {
            badPostureStartTime = 0L
            hasAlerted = false
        }

        val statusText = if (lastState == "good") {
            "😘姿态良好"
        } else {
            "${lastBadIssues.joinToString("/")}"
        }
        onResult(statusText, lastState == "bad", debugStr)
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

    private fun playAlertSound() {
        val player = mediaPlayer ?: MediaPlayer.create(context, R.raw.sound)?.apply {
            isLooping = false
        }?.also {
            mediaPlayer = it
        } ?: return

        runCatching {
            if (player.isPlaying) {
                player.seekTo(0)
            } else {
                player.start()
            }
        }
    }

    fun close() {
        isClosed = true
        poseLandmarker?.close()
        poseLandmarker = null

        mediaPlayer?.let { player ->
            runCatching {
                if (player.isPlaying) player.stop()
            }
            player.release()
        }
        mediaPlayer = null
    }
}
