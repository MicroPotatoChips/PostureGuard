package com.postureguard

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

/** Owned by the single CameraX analysis executor, including model creation and disposal. */
class PoseAnalyzer(
    private val context: Context,
    private val onResult: (PostureResult, Long) -> Unit
) : ImageAnalysis.Analyzer {
    @Volatile var enabled = false
    @Volatile var sessionId = 0L
    private var closed = false
    private var landmarker: PoseLandmarker? = null
    private val engine = PostureEngine()
    private var lastInference = 0L

    fun prepare() {
        check(!closed)
        if (landmarker != null) return
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(.6f)
            .setMinPosePresenceConfidence(.6f)
            .setMinTrackingConfidence(.6f)
            .build()
        landmarker = PoseLandmarker.createFromOptions(context, options)
    }

    fun reset(clearCalibration: Boolean = false) = engine.reset(clearCalibration)
    fun setMode(mode: PostureMode) = engine.setMode(mode)
    fun calibrate() = engine.startCalibration()

    override fun analyze(image: ImageProxy) {
        var bitmap: Bitmap? = null
        var rotated: Bitmap? = null
        val token = sessionId
        try {
            if (!enabled || closed || landmarker == null) return
            val now = SystemClock.uptimeMillis()
            if (now - lastInference < 66L) return
            lastInference = maxOf(now, lastInference + 1)
            val source = image.toBitmap().also { bitmap = it }
            val frame = if (image.imageInfo.rotationDegrees == 0) source else {
                val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            }
            rotated = frame
            val width = frame.width
            val height = frame.height
            val mpImage = BitmapImageBuilder(frame).build()
            val pose = try {
                landmarker!!.detectForVideo(mpImage, lastInference)
            } finally {
                mpImage.close()
            }
            if (!enabled || sessionId != token) return
            val points = pose.landmarks().firstOrNull()?.map {
                PosePoint(it.x(), it.y(), it.visibility().orElse(0f), it.presence().orElse(0f))
            } ?: emptyList()
            onResult(engine.update(points, width, height, lastInference), token)
        } catch (error: Exception) {
            Log.e("PostureGuard", "Frame analysis failed", error)
            engine.reset()
            if (enabled && sessionId == token) onResult(PostureResult(PostureState.ERROR, engine.mode), token)
        } finally {
            if (rotated !== bitmap) rotated?.let { if (!it.isRecycled) it.recycle() }
            bitmap?.let { if (!it.isRecycled) it.recycle() }
            image.close()
        }
    }

    fun close() {
        enabled = false
        closed = true
        engine.reset(clearCalibration = true)
        landmarker?.close()
        landmarker = null
    }
}
