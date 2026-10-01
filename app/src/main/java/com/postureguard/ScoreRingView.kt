package com.postureguard

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class ScoreRingView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var score: Int? = null
    private var warning = false
    private val ring = RectF()

    init { contentDescription = context.getString(R.string.score_waiting) }

    fun setScore(value: Int?, isWarning: Boolean) {
        if (score == value && warning == isWarning) return
        score = value
        warning = isWarning
        contentDescription = value?.let { context.getString(R.string.score_description, it) }
            ?: context.getString(R.string.score_waiting)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val size = minOf(width, height).toFloat()
        val left = (width - size) / 2f
        val top = (height - size) / 2f
        val stroke = size * .065f
        ring.set(left + stroke, top + stroke, left + size - stroke, top + size - stroke)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = stroke
        paint.strokeCap = Paint.Cap.ROUND
        paint.color = ContextCompat.getColor(context, R.color.line)
        canvas.drawArc(ring, -90f, 360f, false, paint)
        paint.color = ContextCompat.getColor(context, if (warning) R.color.warning else R.color.mint_primary)
        score?.let { canvas.drawArc(ring, -90f, it * 3.6f, false, paint) }
        paint.style = Paint.Style.FILL
        paint.textAlign = Paint.Align.CENTER
        paint.textSize = size * .31f
        paint.typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        val center = height / 2f
        canvas.drawText(score?.toString() ?: "—", width / 2f, center + size * .025f, paint)
        paint.textSize = size * .12f
        paint.typeface = Typeface.create("sans-serif", Typeface.NORMAL)
        paint.color = ContextCompat.getColor(context, R.color.ink_muted)
        canvas.drawText(context.getString(R.string.score_caption), width / 2f, center + size * .23f, paint)
    }
}
