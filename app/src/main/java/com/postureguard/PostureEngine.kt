package com.postureguard

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

enum class PostureMode { FRONT, SIDE }
enum class PostureState { SEARCHING, CHECKING, GOOD, ADJUST, CALIBRATING, ERROR }
enum class PostureIssue { SHOULDER_TILT, HEAD_TILT, FORWARD_HEAD, TRUNK_LEAN }

data class PosePoint(
    val x: Float,
    val y: Float,
    val visibility: Float = 1f,
    val presence: Float = 1f
)

data class PostureMetrics(
    val shoulderRoll: Float = 0f,
    val headRoll: Float = 0f,
    val forwardHead: Float = 0f,
    val neckBend: Float = 0f,
    val trunkLean: Float = 0f,
    val confidence: Float = 0f
)

data class PostureResult(
    val state: PostureState,
    val mode: PostureMode,
    val metrics: PostureMetrics? = null,
    val score: Int? = null,
    val issues: Set<PostureIssue> = emptySet(),
    val calibrated: Boolean = false,
    val calibrationProgress: Int = 0,
    val badDurationMs: Long = 0,
    val shouldAlert: Boolean = false
)

/** Pure geometry and temporal decisions. All timestamps are monotonic milliseconds. */
class PostureEngine {
    var mode = PostureMode.SIDE
        private set
    private var baseline: PostureMetrics? = null
    private var baselineLeft: Boolean? = null
    private var smooth: PostureMetrics? = null
    private var lastFrameAt: Long? = null
    private var selectedLeft: Boolean? = null
    private var stableBad = false
    private var pendingBad: Boolean? = null
    private var pendingSince = 0L
    private var badSince: Long? = null
    private var alerted = false
    private var activeIssues = emptySet<PostureIssue>()
    private var calibrating = false
    private val calibrationSamples = mutableListOf<PostureMetrics>()
    private var calibrationSince: Long? = null

    fun setMode(value: PostureMode) {
        if (value == mode) return
        mode = value
        reset(clearCalibration = true)
    }

    fun reset(clearCalibration: Boolean = false) {
        clearTracking()
        calibrating = false
        calibrationSamples.clear()
        calibrationSince = null
        if (clearCalibration) {
            baseline = null
            baselineLeft = null
        }
    }

    fun startCalibration() {
        clearTracking()
        calibrating = true
        calibrationSamples.clear()
        calibrationSince = null
    }

    private fun clearTracking() {
        smooth = null
        lastFrameAt = null
        selectedLeft = null
        stableBad = false
        pendingBad = null
        badSince = null
        alerted = false
        activeIssues = emptySet()
    }

    fun update(points: List<PosePoint>, width: Int, height: Int, now: Long): PostureResult {
        val previousTime = lastFrameAt
        // A gap must never count as observed bad posture or calibration time.
        if (previousTime != null && (now <= previousTime || now - previousTime > 1000)) {
            clearTracking()
            calibrationSamples.clear()
            calibrationSince = null
        }
        val raw = measure(points, width, height)
        if (raw == null) {
            clearTracking()
            calibrationSamples.clear()
            calibrationSince = null
            return result(if (calibrating) PostureState.CALIBRATING else PostureState.SEARCHING)
        }
        val delta = lastFrameAt?.let { now - it } ?: 0L
        lastFrameAt = now
        val alpha = if (smooth == null) 1f else (1.0 - exp(-delta / 350.0)).toFloat()
        val filtered = blend(smooth ?: raw, raw, alpha)
        smooth = filtered

        if (calibrating) return calibrate(raw, filtered, now)

        val issues = classify(filtered, baseline, activeIssues)
        activeIssues = issues
        val bad = issues.isNotEmpty()
        if (bad == stableBad) {
            pendingBad = null
        } else {
            if (pendingBad != bad) {
                pendingBad = bad
                pendingSince = now
            }
            val dwellMs = if (bad) 1200L else 800L
            if (now - pendingSince >= dwellMs) {
                stableBad = bad
                pendingBad = null
                badSince = if (bad) now else null
                alerted = false
            }
        }
        val duration = badSince?.let { now - it } ?: 0L
        val alert = stableBad && bad && duration >= ALERT_DELAY_MS && !alerted
        if (alert) alerted = true
        return result(
            if (stableBad) PostureState.ADJUST else if (bad) PostureState.CHECKING else PostureState.GOOD,
            filtered,
            score(filtered),
            issues,
            duration = duration,
            alert = alert
        )
    }

    private fun calibrate(raw: PostureMetrics, filtered: PostureMetrics, now: Long): PostureResult {
        // Avoid accepting an obviously tilted or folded pose as the personal neutral pose.
        val plausible = if (mode == PostureMode.FRONT) {
            abs(raw.shoulderRoll) <= 18f && abs(raw.headRoll) <= 22f
        } else {
            abs(raw.forwardHead) <= .45f && abs(raw.trunkLean) <= 28f && raw.neckBend <= 50f
        }
        if (!plausible || !isSteady(raw)) {
            calibrationSamples.clear()
            calibrationSince = null
            return result(PostureState.CALIBRATING, filtered)
        }
        if (calibrationSince == null) calibrationSince = now
        calibrationSamples.add(raw)
        val elapsed = now - (calibrationSince ?: now)
        val progress = min((elapsed * 100 / 3000).toInt(), calibrationSamples.size * 100 / 20).coerceIn(0, 100)
        if (progress < 100) return result(PostureState.CALIBRATING, filtered, progress = progress)
        baseline = PostureMetrics(
            median { it.shoulderRoll }, median { it.headRoll }, median { it.forwardHead },
            median { it.neckBend }, median { it.trunkLean }, median { it.confidence }
        )
        baselineLeft = selectedLeft
        calibrating = false
        calibrationSamples.clear()
        calibrationSince = null
        clearTracking()
        smooth = filtered
        lastFrameAt = now
        return result(PostureState.GOOD, filtered, 100)
    }

    private fun isSteady(raw: PostureMetrics): Boolean {
        val first = calibrationSamples.firstOrNull() ?: return true
        return if (mode == PostureMode.FRONT) {
            abs(raw.shoulderRoll - first.shoulderRoll) < 4 && abs(raw.headRoll - first.headRoll) < 4
        } else {
            abs(raw.forwardHead - first.forwardHead) < .06 &&
                abs(raw.trunkLean - first.trunkLean) < 5 && abs(raw.neckBend - first.neckBend) < 6
        }
    }

    private fun median(value: (PostureMetrics) -> Float): Float {
        val sorted = calibrationSamples.map(value).sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2 else sorted[middle]
    }

    private fun classify(m: PostureMetrics, b: PostureMetrics?, previous: Set<PostureIssue>): Set<PostureIssue> {
        val issues = mutableSetOf<PostureIssue>()
        fun exceeds(issue: PostureIssue, value: Float, threshold: Float) {
            val limit = if (issue in previous) threshold * .75f else threshold
            if (value > limit) issues.add(issue)
        }
        if (mode == PostureMode.FRONT) {
            exceeds(PostureIssue.SHOULDER_TILT, abs(m.shoulderRoll - (b?.shoulderRoll ?: 0f)), 5f)
            exceeds(PostureIssue.HEAD_TILT, abs(m.headRoll - (b?.headRoll ?: 0f)), 8f)
        } else {
            // Require agreement between head displacement and the ear-shoulder-hip angle.
            val displacement = abs(m.forwardHead - (b?.forwardHead ?: 0f)) / .22f
            val bend = max(0f, m.neckBend - (b?.neckBend ?: 0f)) / 25f
            exceeds(PostureIssue.FORWARD_HEAD, min(displacement, bend), 1f)
            exceeds(PostureIssue.TRUNK_LEAN, abs(m.trunkLean - (b?.trunkLean ?: 0f)), 15f)
        }
        return issues
    }

    private fun score(m: PostureMetrics): Int {
        val b = baseline
        val severity = if (mode == PostureMode.FRONT) {
            max(abs(m.shoulderRoll - (b?.shoulderRoll ?: 0f)) / 5f,
                abs(m.headRoll - (b?.headRoll ?: 0f)) / 8f)
        } else {
            max(min(abs(m.forwardHead - (b?.forwardHead ?: 0f)) / .22f,
                max(0f, m.neckBend - (b?.neckBend ?: 0f)) / 25f),
                abs(m.trunkLean - (b?.trunkLean ?: 0f)) / 15f)
        }
        return (100 - min(severity, 3f) * 25).toInt().coerceIn(0, 100)
    }

    private fun result(
        state: PostureState, metrics: PostureMetrics? = null, score: Int? = null,
        issues: Set<PostureIssue> = emptySet(), progress: Int = 0,
        duration: Long = 0, alert: Boolean = false
    ) = PostureResult(state, mode, metrics, score, issues, baseline != null, progress, duration, alert)

    private fun quality(point: PosePoint?) = point?.let { min(it.visibility, it.presence) } ?: 0f

    private fun reliable(p: PosePoint?) = p != null && p.x.isFinite() && p.y.isFinite() &&
        p.x in 0f..1f && p.y in 0f..1f && quality(p) >= .65f

    private fun measure(points: List<PosePoint>, width: Int, height: Int): PostureMetrics? {
        if (width <= 0 || height <= 0) return null
        // x and y are normalized separately by MediaPipe: convert to the same scale.
        val aspect = width.toFloat() / height
        fun x(p: PosePoint) = p.x * aspect
        fun roll(left: PosePoint, right: PosePoint): Float {
            val dy = left.y - right.y
            return Math.toDegrees(atan2(dy.toDouble(), abs(x(left) - x(right)).toDouble())).toFloat()
        }
        if (mode == PostureMode.FRONT) {
            val ids = listOf(11, 12, 7, 8)
            if (!ids.all { reliable(points.getOrNull(it)) }) return null
            val ls = points[11]; val rs = points[12]; val le = points[7]; val re = points[8]
            if (abs(x(ls) - x(rs)) < .08f || abs(x(le) - x(re)) < .025f) return null
            return PostureMetrics(shoulderRoll = roll(ls, rs), headRoll = roll(le, re),
                confidence = ids.minOf { quality(points[it]) })
        }
        val leftIds = listOf(7, 11, 23)
        val rightIds = listOf(8, 12, 24)
        val leftQuality = if (leftIds.all { reliable(points.getOrNull(it)) }) leftIds.minOf { quality(points[it]) } else 0f
        val rightQuality = if (rightIds.all { reliable(points.getOrNull(it)) }) rightIds.minOf { quality(points[it]) } else 0f
        if (max(leftQuality, rightQuality) < .65f) return null
        val useLeft = when (selectedLeft) {
            true -> leftQuality >= .65f && rightQuality < leftQuality + .15f
            false -> rightQuality < .65f || leftQuality >= rightQuality + .15f
            null -> leftQuality >= rightQuality
        }
        if (selectedLeft != null && useLeft != selectedLeft) {
            // A side change invalidates smoothing and a calibration collected from that side.
            clearTracking()
            calibrationSamples.clear()
            calibrationSince = null
            baseline = null
        }
        selectedLeft = useLeft
        if (baselineLeft != null && baselineLeft != useLeft) {
            baseline = null
            baselineLeft = null
        }
        val ids = if (useLeft) leftIds else rightIds
        val ear = points[ids[0]]; val shoulder = points[ids[1]]; val hip = points[ids[2]]
        val tx = x(shoulder) - x(hip); val ty = shoulder.y - hip.y
        val nx = x(ear) - x(shoulder); val ny = ear.y - shoulder.y
        val torso = hypot(tx, ty); val neck = hypot(nx, ny)
        if (torso < .12f || neck < .035f || ty >= -.05f || ny >= 0f) return null
        val leftShoulder = points.getOrNull(11)
        val rightShoulder = points.getOrNull(12)
        if (reliable(leftShoulder) && reliable(rightShoulder) &&
            abs(x(leftShoulder!!) - x(rightShoulder!!)) / torso > .65f) return null
        val cosine = ((nx * tx + ny * ty) / (neck * torso)).coerceIn(-1f, 1f)
        return PostureMetrics(
            forwardHead = nx / torso,
            neckBend = Math.toDegrees(acos(cosine.toDouble())).toFloat(),
            trunkLean = Math.toDegrees(atan2(tx.toDouble(), -ty.toDouble())).toFloat(),
            confidence = min(leftQuality.takeIf { useLeft } ?: rightQuality, 1f)
        )
    }

    private fun blend(a: PostureMetrics, b: PostureMetrics, alpha: Float): PostureMetrics {
        fun mix(x: Float, y: Float) = x + alpha * (y - x)
        return PostureMetrics(mix(a.shoulderRoll, b.shoulderRoll), mix(a.headRoll, b.headRoll),
            mix(a.forwardHead, b.forwardHead), mix(a.neckBend, b.neckBend),
            mix(a.trunkLean, b.trunkLean), b.confidence)
    }

    companion object { const val ALERT_DELAY_MS = 15_000L }
}
