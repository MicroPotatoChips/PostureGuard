package com.postureguard

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.media.MediaPlayer
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.card.MaterialCardView
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.progressindicator.LinearProgressIndicator
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity() {
    private lateinit var viewFinder: PreviewView
    private lateinit var btnToggle: MaterialButton
    private lateinit var btnCalibrate: MaterialButton
    private lateinit var soundSwitch: MaterialSwitch
    private lateinit var analyzer: PoseAnalyzer
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var mediaPlayer: MediaPlayer? = null
    private var modelReady = false
    private var isRunning = false
    private var isStarting = false
    private var revision = 0L
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var selectedMode = PostureMode.SIDE
    private var calibrated = false
    private var validTimeMs = 0L
    private var goodTimeMs = 0L
    private var lastValidAt: Long? = null
    private var previousWasGood = false
    private var latestResult: PostureResult? = null

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted && lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) startCamera()
        else if (!granted) toast(R.string.camera_permission_denied)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        viewFinder = findViewById(R.id.viewFinder)
        btnToggle = findViewById(R.id.btnToggle)
        btnCalibrate = findViewById(R.id.btnCalibrate)
        soundSwitch = findViewById(R.id.switchSound)
        val preferences = getSharedPreferences("posture_preferences", MODE_PRIVATE)
        selectedMode = if (preferences.getBoolean("front_mode", false)) PostureMode.FRONT else PostureMode.SIDE
        soundSwitch.isChecked = preferences.getBoolean("sound", true)
        analyzer = PoseAnalyzer(applicationContext) { result, token ->
            runOnUiThread {
                if (isRunning && !isDestroyed && token == revision) renderResult(result)
            }
        }
        val modeGroup = findViewById<MaterialButtonToggleGroup>(R.id.modeGroup)
        modeGroup.check(if (selectedMode == PostureMode.FRONT) R.id.btnFront else R.id.btnSide)
        modeGroup.addOnButtonCheckedListener { _, id, checked ->
            if (checked) {
                val mode = if (id == R.id.btnFront) PostureMode.FRONT else PostureMode.SIDE
                if (mode != selectedMode) {
                    selectedMode = mode
                    preferences.edit().putBoolean("front_mode", mode == PostureMode.FRONT).apply()
                    val restart = isStarting
                    if (restart) stopCamera()
                    resetTracking(clearCalibration = true)
                    if (restart) startCamera(resetStatistics = false)
                }
            }
        }
        btnToggle.setOnClickListener {
            if (isRunning || isStarting) stopCamera()
            else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
        findViewById<MaterialButton>(R.id.btnSwitch).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) CameraSelector.LENS_FACING_BACK
                else CameraSelector.LENS_FACING_FRONT
            text(R.id.tvCamera).setText(if (lensFacing == CameraSelector.LENS_FACING_FRONT) R.string.camera_front else R.string.camera_back)
            val restart = isRunning || isStarting
            if (restart) stopCamera()
            resetTracking(clearCalibration = true)
            if (restart) startCamera(resetStatistics = false)
        }
        btnCalibrate.setOnClickListener {
            if (isRunning) {
                val token = invalidateResults()
                btnCalibrate.isEnabled = false
                renderResult(PostureResult(PostureState.CALIBRATING, selectedMode, calibrated = calibrated))
                cameraExecutor.execute {
                    analyzer.calibrate()
                    runOnUiThread {
                        if (token == revision && isRunning) analyzer.enabled = true
                    }
                }
            }
        }
        soundSwitch.setOnCheckedChangeListener { _, enabled ->
            preferences.edit().putBoolean("sound", enabled).apply()
            soundSwitch.setText(if (enabled) R.string.sound_on else R.string.sound_off)
            if (!enabled) stopSound()
            renderAlert(latestResult)
        }
        soundSwitch.setText(if (soundSwitch.isChecked) R.string.sound_on else R.string.sound_off)
        renderIdle()
        btnToggle.setText(R.string.loading_model)
        cameraExecutor.execute {
            try {
                analyzer.prepare()
                runOnUiThread {
                    if (!isDestroyed) {
                        modelReady = true
                        btnToggle.isEnabled = true
                        btnToggle.setText(R.string.start_monitoring)
                    }
                }
            } catch (error: Exception) {
                Log.e("PostureGuard", "Model initialization failed", error)
                runOnUiThread {
                    if (!isDestroyed) {
                        text(R.id.tvStatus).setText(R.string.analysis_failed)
                        text(R.id.tvHint).setText(R.string.model_failed)
                        btnToggle.setText(R.string.start_monitoring)
                    }
                }
            }
        }
    }

    private fun startCamera(resetStatistics: Boolean = true) {
        if (!modelReady || isStarting || !lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) return
        isStarting = true
        val token = invalidateResults()
        btnToggle.isEnabled = false
        btnToggle.setText(R.string.starting_camera)
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (token != revision || isDestroyed || !lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) return@addListener
            try {
                val provider = future.get().also { cameraProvider = it }
                val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                check(provider.hasCamera(selector))
                val rotation = viewFinder.display?.rotation ?: Surface.ROTATION_0
                val preview = Preview.Builder().setTargetRotation(rotation).build().also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetRotation(rotation)
                    .build().also { it.setAnalyzer(cameraExecutor, analyzer) }
                imageAnalysis?.clearAnalyzer()
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
                imageAnalysis = analysis
                isStarting = false
                isRunning = true
                if (resetStatistics) {
                    validTimeMs = 0L
                    goodTimeMs = 0L
                    updateStatistics()
                }
                lastValidAt = null
                btnToggle.isEnabled = true
                btnToggle.setText(R.string.pause_monitoring)
                btnToggle.backgroundTintList = ColorStateList.valueOf(color(R.color.ink))
                viewFinder.visibility = View.VISIBLE
                findViewById<View>(R.id.previewPlaceholder).visibility = View.GONE
                text(R.id.tvLive).setText(R.string.live_badge)
                renderResult(PostureResult(PostureState.SEARCHING, selectedMode, calibrated = calibrated))
                val mode = selectedMode
                cameraExecutor.execute {
                    analyzer.setMode(mode)
                    analyzer.reset()
                    runOnUiThread {
                        if (token == revision && isRunning) analyzer.enabled = true
                    }
                }
            } catch (error: Exception) {
                Log.e("PostureGuard", "Camera binding failed", error)
                stopCamera()
                text(R.id.tvHint).setText(R.string.camera_failed)
                toast(R.string.camera_failed)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun invalidateResults(): Long {
        revision += 1
        analyzer.enabled = false
        analyzer.sessionId = revision
        lastValidAt = null
        previousWasGood = false
        stopSound()
        return revision
    }

    private fun resetTracking(clearCalibration: Boolean) {
        val token = invalidateResults()
        if (clearCalibration) calibrated = false
        val mode = selectedMode
        cameraExecutor.execute {
            analyzer.setMode(mode)
            analyzer.reset(clearCalibration)
            runOnUiThread {
                if (token == revision && isRunning) analyzer.enabled = true
            }
        }
        if (isRunning) renderResult(PostureResult(PostureState.SEARCHING, mode, calibrated = calibrated))
        else renderIdle()
    }

    private fun stopCamera() {
        invalidateResults()
        imageAnalysis?.clearAnalyzer()
        imageAnalysis = null
        cameraProvider?.unbindAll()
        isRunning = false
        isStarting = false
        cameraExecutor.execute { analyzer.reset() }
        btnToggle.isEnabled = modelReady
        btnToggle.setText(R.string.start_monitoring)
        btnToggle.backgroundTintList = ColorStateList.valueOf(color(R.color.mint_primary))
        viewFinder.visibility = View.INVISIBLE
        findViewById<View>(R.id.previewPlaceholder).visibility = View.VISIBLE
        text(R.id.tvLive).setText(R.string.ready_badge)
        renderIdle()
    }

    private fun renderIdle() {
        latestResult = null
        text(R.id.tvStatus).setText(R.string.status_idle)
        text(R.id.tvHint).setText(R.string.hint_idle)
        text(R.id.tvStatus).setTextColor(color(R.color.ink))
        findViewById<MaterialCardView>(R.id.statusCard).setCardBackgroundColor(color(R.color.surface))
        findViewById<ScoreRingView>(R.id.scoreRing).setScore(null, false)
        renderMetrics(null)
        renderCalibration(false, 0)
        btnCalibrate.isEnabled = false
        renderAlert(null)
    }

    private fun renderResult(result: PostureResult) {
        latestResult = result
        calibrated = result.calibrated
        val bad = result.state == PostureState.ADJUST
        val status = when (result.state) {
            PostureState.SEARCHING -> R.string.status_searching
            PostureState.CHECKING -> R.string.status_checking
            PostureState.GOOD -> R.string.status_good
            PostureState.ADJUST -> R.string.status_adjust
            PostureState.CALIBRATING -> R.string.status_calibrating
            PostureState.ERROR -> R.string.analysis_failed
        }
        text(R.id.tvStatus).setText(status)
        text(R.id.tvStatus).setTextColor(color(if (bad) R.color.warning else R.color.ink))
        val issueText = result.issues.joinToString("；") { issue ->
            getString(when (issue) {
                PostureIssue.SHOULDER_TILT -> R.string.issue_shoulder
                PostureIssue.HEAD_TILT -> R.string.issue_head
                PostureIssue.FORWARD_HEAD -> R.string.issue_forward
                PostureIssue.TRUNK_LEAN -> R.string.issue_trunk
            })
        }
        text(R.id.tvHint).text = if ((bad || result.state == PostureState.CHECKING) && issueText.isNotEmpty()) issueText else getString(when (result.state) {
            PostureState.SEARCHING -> if (selectedMode == PostureMode.FRONT) R.string.hint_front else R.string.hint_side
            PostureState.CHECKING -> R.string.hint_adjust
            PostureState.GOOD -> R.string.hint_good
            PostureState.ADJUST -> R.string.hint_adjust
            PostureState.CALIBRATING -> R.string.hint_calibrating
            PostureState.ERROR -> R.string.analysis_retry
        })
        findViewById<MaterialCardView>(R.id.statusCard).setCardBackgroundColor(
            color(if (bad) R.color.warning_light else R.color.surface))
        findViewById<ScoreRingView>(R.id.scoreRing).setScore(result.score, bad)
        renderMetrics(result.metrics)
        renderCalibration(result.state == PostureState.CALIBRATING, result.calibrationProgress)
        btnCalibrate.isEnabled = result.metrics != null && result.state != PostureState.CALIBRATING
        val valid = result.metrics != null &&
            result.state in setOf(PostureState.CHECKING, PostureState.GOOD, PostureState.ADJUST)
        val now = SystemClock.uptimeMillis()
        val good = result.state == PostureState.GOOD && result.issues.isEmpty()
        if (valid) {
            lastValidAt?.let { previous ->
                val elapsed = now - previous
                if (elapsed in 1..1000) {
                    val interval = elapsed.coerceAtMost(500)
                    validTimeMs += interval
                    if (good && previousWasGood) goodTimeMs += interval
                }
            }
            lastValidAt = now
        } else lastValidAt = null
        previousWasGood = good
        updateStatistics()
        renderAlert(result)
        if (result.shouldAlert && soundSwitch.isChecked) playSound()
    }

    private fun renderMetrics(metrics: PostureMetrics?) {
        val front = selectedMode == PostureMode.FRONT
        text(R.id.tvMetricOneLabel).setText(if (front) R.string.metric_shoulder else R.string.metric_head_forward)
        text(R.id.tvMetricTwoLabel).setText(if (front) R.string.metric_head else R.string.metric_trunk)
        text(R.id.tvMetricOne).text = metrics?.let {
            if (front) getString(R.string.metric_degrees, abs(it.shoulderRoll))
            else getString(R.string.metric_ratio, abs(it.forwardHead))
        } ?: getString(R.string.metric_missing)
        text(R.id.tvMetricTwo).text = metrics?.let {
            getString(R.string.metric_degrees, abs(if (front) it.headRoll else it.trunkLean))
        } ?: getString(R.string.metric_missing)
        text(R.id.tvSignal).text = metrics?.let { getString(R.string.signal_quality, (it.confidence * 100).toInt()) }
            ?: getString(R.string.signal_waiting)
    }

    private fun renderCalibration(inProgress: Boolean, progress: Int) {
        text(R.id.tvCalibration).text = if (inProgress) getString(R.string.calibration_progress, progress)
            else getString(if (calibrated) R.string.calibration_done else R.string.calibration_default)
        btnCalibrate.setText(if (calibrated) R.string.recalibrate else R.string.calibrate)
        findViewById<LinearProgressIndicator>(R.id.calibrationProgress).apply {
            visibility = if (inProgress) View.VISIBLE else View.GONE
            setProgressCompat(progress, false)
        }
    }

    private fun updateStatistics() {
        val seconds = validTimeMs / 1000
        text(R.id.tvSession).text = String.format(Locale.ROOT, "%02d:%02d", seconds / 60, seconds % 60)
        text(R.id.tvGoodRate).text = if (validTimeMs == 0L) getString(R.string.metric_missing)
            else getString(R.string.percent_value, (goodTimeMs * 100 / validTimeMs).toInt())
    }

    private fun renderAlert(result: PostureResult?) {
        val badDuration = result?.badDurationMs ?: 0L
        text(R.id.tvAlert).text = when {
            !soundSwitch.isChecked -> getString(R.string.alert_muted)
            result?.state != PostureState.ADJUST -> getString(R.string.alert_caption)
            badDuration >= PostureEngine.ALERT_DELAY_MS -> getString(R.string.alert_sent)
            else -> getString(R.string.alert_countdown,
                ((PostureEngine.ALERT_DELAY_MS - badDuration + 999) / 1000).toInt())
        }
    }

    private fun playSound() {
        try {
            val player = mediaPlayer ?: MediaPlayer.create(this, R.raw.sound)?.also { mediaPlayer = it } ?: return
            player.seekTo(0)
            player.start()
        } catch (error: Exception) { Log.w("PostureGuard", "Reminder sound unavailable", error) }
    }

    private fun stopSound() { mediaPlayer?.let { runCatching { it.pause(); it.seekTo(0) } } }
    private fun text(id: Int) = findViewById<TextView>(id)
    private fun color(id: Int) = ContextCompat.getColor(this, id)
    private fun toast(message: Int) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    override fun onStop() {
        stopCamera()
        super.onStop()
    }

    override fun onDestroy() {
        // The queued close runs after any in-flight inference on the same executor.
        analyzer.enabled = false
        cameraExecutor.execute { analyzer.close() }
        cameraExecutor.shutdown()
        mediaPlayer?.release()
        mediaPlayer = null
        cameraProvider = null
        super.onDestroy()
    }
}
