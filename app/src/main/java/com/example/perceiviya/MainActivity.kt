package com.example.perceiviya

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.*
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val label: String,
    val score: Float,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float, // Normalized 0..1
    val zone: String,      // "left", "ahead", "right"
    val proximity: String  // "URGENT", "NEAR", "FAR"
)

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: BoxOverlayView
    private lateinit var statusText: TextView
    private lateinit var tts: TextToSpeech
    private var ttsReady = false

    private val depthInputSize = 256
    private var depthSession: OrtSession? = null

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    private val inputSize = 320
    private val numAnchors = 2100 // For imgsz=320: (40*40) + (20*20) + (10*10) = 2100
    private val numClasses = 80
    private var lastAlertTime = 0L
    private var lastAlertSignature = ""
    @Volatile private var latestDetections: List<Detection> = emptyList()

    private val cocoLabels = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
        overlayView = BoxOverlayView(this)
        statusText = TextView(this).apply {
            textSize = 18f
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xCC000000.toInt())
            setPadding(24, 110, 24, 24)
            text = "Initializing offline ONNX model..."
        }
        val root = FrameLayout(this).apply {
            addView(previewView)
            addView(overlayView)
            addView(statusText)
            setOnClickListener { speakSceneDescription() }
        }
        setContentView(root)

        tts = TextToSpeech(this, this)
        initOnnxModel()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initOnnxModel() {
        cameraExecutor.execute {
            try {
                val env = OrtEnvironment.getEnvironment()
                val options = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(4)
                }

                val modelBytes = assets.open("yolo11n.onnx").readBytes()
                val yolo = env.createSession(modelBytes, options)

                // Optional depth model: safely falls back to box heuristic if missing or unsupported
                val depth = try {
                    val depthBytes = assets.open("depth.onnx").readBytes()
                    env.createSession(depthBytes, options)
                } catch (_: Exception) {
                    null
                }

                ortEnv = env
                ortSession = yolo
                depthSession = depth

                val modeLabel = if (depth != null) "YOLO + Depth ready" else "YOLO ready (box proximity)"
                runOnUiThread { statusText.text = "$modeLabel. Tap to describe." }
            } catch (e: Exception) {
                runOnUiThread { statusText.text = "ONNX load error: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val env = ortEnv
        val session = ortSession
        if (env == null || session == null) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            val rotated = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
            val depthInfo = runDepthMap(env, rotated)

            val resized = Bitmap.createScaledBitmap(rotated, inputSize, inputSize, true)
            val inputName = session.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val floatBuffer = FloatBuffer.wrap(bitmapToNCHWWithSize(resized, inputSize))

            OnnxTensor.createTensor(env, floatBuffer, shape).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { results ->
                    val flatOutput = extractFloats(results[0].value)
                    val detections = decodeAndNms(
                        flatOutput,
                        depthInfo = depthInfo,
                        confThreshold = 0.45f,
                        iouThreshold = 0.45f
                    )
                    latestDetections = detections
                    runOnUiThread {
                        overlayView.detections = detections
                        overlayView.invalidate()
                    }
                    evaluateObstacleWarnings(detections)
                }
            }
        } finally {
            imageProxy.close()
        }
    }

    private fun runDepthMap(env: OrtEnvironment, rotatedBitmap: Bitmap): Pair<FloatArray, Int>? {
        val session = depthSession ?: return null
        return try {
            val resized = Bitmap.createScaledBitmap(rotatedBitmap, depthInputSize, depthInputSize, true)
            val inputName = session.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, depthInputSize.toLong(), depthInputSize.toLong())
            val buffer = FloatBuffer.wrap(bitmapToNCHWWithSize(resized, depthInputSize))

            OnnxTensor.createTensor(env, buffer, shape).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { res ->
                    val raw = extractFloats(res[0].value)
                    if (raw.isEmpty()) return null

                    var minVal = Float.POSITIVE_INFINITY
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (v in raw) {
                        if (v < minVal) minVal = v
                        if (v > maxVal) maxVal = v
                    }
                    val span = (maxVal - minVal).coerceAtLeast(1e-6f)
                    val norm = FloatArray(raw.size) { idx ->
                        ((raw[idx] - minVal) / span).coerceIn(0f, 1f)
                    }
                    Pair(norm, depthInputSize)
                }
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun medianDepthInBox(
        depthMap: FloatArray,
        size: Int,
        x1: Float, y1: Float, x2: Float, y2: Float
    ): Float {
        val startX = (x1 * size).toInt().coerceIn(0, size - 1)
        val endX = (x2 * size).toInt().coerceIn(startX + 1, size)
        val startY = (y1 * size).toInt().coerceIn(0, size - 1)
        val endY = (y2 * size).toInt().coerceIn(startY + 1, size)

        val samples = ArrayList<Float>((endX - startX) * (endY - startY) / 4 + 1)
        var y = startY
        while (y < endY) {
            var x = startX
            while (x < endX) {
                samples.add(depthMap[y * size + x])
                x += 2
            }
            y += 2
        }
        if (samples.isEmpty()) return 0.5f
        samples.sort()
        return samples[samples.size / 2]
    }

    // Flat layout for [1, 84, 2100]: index = channel * numAnchors + anchorIdx
    private fun decodeAndNms(
        flatData: FloatArray,
        depthInfo: Pair<FloatArray, Int>?,
        confThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        if (flatData.size < (4 + numClasses) * numAnchors) return emptyList()
        val candidates = mutableListOf<Detection>()

        for (i in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = confThreshold
            for (c in 0 until numClasses) {
                val score = flatData[(4 + c) * numAnchors + i]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            if (bestClass == -1) continue

            val cx = flatData[0 * numAnchors + i] / inputSize
            val cy = flatData[1 * numAnchors + i] / inputSize
            val w = flatData[2 * numAnchors + i] / inputSize
            val h = flatData[3 * numAnchors + i] / inputSize

            val x1 = (cx - w / 2f).coerceIn(0f, 1f)
            val y1 = (cy - h / 2f).coerceIn(0f, 1f)
            val x2 = (cx + w / 2f).coerceIn(0f, 1f)
            val y2 = (cy + h / 2f).coerceIn(0f, 1f)

            val centerX = (x1 + x2) / 2f
            val boxArea = (x2 - x1) * (y2 - y1)

            val zone = when {
                centerX < 0.35f -> "left"
                centerX > 0.65f -> "right"
                else -> "ahead"
            }

            val proximity = if (depthInfo != null) {
                val closeScore = medianDepthInBox(depthInfo.first, depthInfo.second, x1, y1, x2, y2)
                when {
                    closeScore > 0.72f || (closeScore > 0.58f && y2 > 0.85f) -> "URGENT"
                    closeScore > 0.45f || y2 > 0.72f -> "NEAR"
                    else -> "FAR"
                }
            } else {
                when {
                    (y2 > 0.85f && boxArea > 0.15f) || boxArea > 0.35f -> "URGENT"
                    y2 > 0.68f || boxArea > 0.12f -> "NEAR"
                    else -> "FAR"
                }
            }

            val label = cocoLabels.getOrElse(bestClass) { "obstacle" }
            candidates.add(Detection(label, bestScore, x1, y1, x2, y2, zone, proximity))
        }

        val sorted = candidates.sortedByDescending { it.score }
        val kept = mutableListOf<Detection>()
        for (item in sorted) {
            val overlaps = kept.any { existing -> iou(existing, item) > iouThreshold }
            if (!overlaps) {
                kept.add(item)
            }
        }
        return kept
    }

    private fun extractFloats(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is FloatBuffer -> {
                val copy = FloatArray(value.remaining())
                value.duplicate().get(copy)
                copy
            }
            is Array<*> -> {
                val out = ArrayList<Float>()
                flattenFloats(value, out)
                out.toFloatArray()
            }
            else -> FloatArray(0)
        }
    }

    private fun flattenFloats(node: Any?, out: ArrayList<Float>) {
        when (node) {
            is Float -> out.add(node)
            is FloatArray -> for (v in node) out.add(v)
            is Array<*> -> for (child in node) flattenFloats(child, out)
        }
    }

    private fun iou(a: Detection, b: Detection): Float {
        val interX1 = max(a.x1, b.x1)
        val interY1 = max(a.y1, b.y1)
        val interX2 = min(a.x2, b.x2)
        val interY2 = min(a.y2, b.y2)
        val interArea = max(0f, interX2 - interX1) * max(0f, interY2 - interY1)
        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        val union = areaA + areaB - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    private fun evaluateObstacleWarnings(detections: List<Detection>) {
        val urgent = detections.firstOrNull { it.proximity == "URGENT" && it.zone == "ahead" }
            ?: detections.firstOrNull { it.proximity == "URGENT" }

        val now = System.currentTimeMillis()
        if (urgent != null) {
            val sig = "${urgent.label}_${urgent.zone}"
            if (now - lastAlertTime > 3500L || sig != lastAlertSignature) {
                lastAlertTime = now
                lastAlertSignature = sig
                val msg = "Warning! ${urgent.label} very close ${urgent.zone}."
                triggerHapticWarning()
                speak(msg, urgent = true)
                runOnUiThread { statusText.text = msg }
            }
        } else {
            val summary = if (detections.isEmpty()) "Path clear"
            else detections.take(3).joinToString(", ") { "${it.label} (${it.zone})" }
            runOnUiThread { statusText.text = summary }
        }
    }

    private fun speakSceneDescription() {
        val current = latestDetections
        if (current.isEmpty()) {
            speak("The path ahead appears clear.", urgent = false)
            return
        }
        val grouped = current.take(4).joinToString(", ") {
            val distWord = if (it.proximity == "FAR") "in the distance" else "nearby"
            "a ${it.label} $distWord ${it.zone}"
        }
        val description = "I see $grouped."
        runOnUiThread { statusText.text = description }
        speak(description, urgent = true)
    }

    private fun speak(text: String, urgent: Boolean) {
        if (!ttsReady) return
        val mode = if (urgent) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts.speak(text, mode, null, System.nanoTime().toString())
    }

    private fun triggerHapticWarning() {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 120, 80, 120), -1))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(250)
        }
    }

    private fun bitmapToNCHWWithSize(bitmap: Bitmap, size: Int): FloatArray {
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        val result = FloatArray(3 * size * size)
        val channelSize = size * size
        for (i in pixels.indices) {
            val p = pixels[i]
            result[i] = ((p shr 16) and 0xFF) / 255.0f
            result[channelSize + i] = ((p shr 8) and 0xFF) / 255.0f
            result[2 * channelSize + i] = (p and 0xFF) / 255.0f
        }
        return result
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        if (angle == 0f) return source
        val matrix = Matrix().apply { postRotate(angle) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setSpeechRate(1.1f)
            ttsReady = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
        depthSession?.close()
        ortSession?.close()
        ortEnv?.close()
        cameraExecutor.shutdown()
    }
}

class BoxOverlayView(context: Context) : View(context) {
    var detections: List<Detection> = emptyList()

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = 0xFF00FF00.toInt()
    }

    private val textPaint = Paint().apply {
        color = 0xFFFFFF00.toInt()
        textSize = 42f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        for (d in detections) {
            canvas.drawRect(d.x1 * w, d.y1 * h, d.x2 * w, d.y2 * h, boxPaint)
            canvas.drawText(
                "${d.label} (${d.proximity})",
                d.x1 * w,
                (d.y1 * h - 10f).coerceAtLeast(45f),
                textPaint
            )
        }
    }
}