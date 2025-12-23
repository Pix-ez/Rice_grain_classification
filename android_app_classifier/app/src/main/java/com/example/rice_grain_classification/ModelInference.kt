package com.example.rice_grain_classification



import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.InputStream
class ModelInference(private val context: Context) {

    private var module: Module? = null
    private var currentModelName: String? = null

    // Default ImageNet classes - update with your actual classes
    private val classNames = listOf(
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight"
    )

    fun getAvailableModels(): List<String> {
        return try {
            context.assets.list("")?.filter { it.endsWith(".pte") } ?: emptyList()
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }

    fun loadModel(modelName: String): Boolean {
        return try {
            // Clean up previous model
            module?.destroy()

            val modelPath = assetFilePath(context, modelName)
            module = Module.load(modelPath)
            currentModelName = modelName
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
//
//    fun runInference(imageUri: Uri, inputSize: Int = 224): List<ClassificationResult>? {
//        if (module == null) return null
//        return try {
//            // 1. Load Bitmap
//            val bitmap = loadAndPreprocessImage(imageUri, inputSize)
//
//            // 2. Convert using our new robust Helper (Uses DirectByteBuffer)
//            val floatBuffer = TensorUtils.bitmapToFloat32Tensor(
//                bitmap,
//                TensorUtils.NORM_MEAN_RGB,
//                TensorUtils.NORM_STD_RGB
//            )
//
//            // 3. Create Tensor from the Buffer
//            // Shape: [1, 3, 224, 224] matches your Python export
//            val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
//            val inputTensor = Tensor.fromBlob(floatBuffer, inputShape)
//
//            // 4. Run Inference
//            val inputEValue = EValue.from(inputTensor)
//            val output = module?.forward(inputEValue)
//
//            val scores = output?.get(0)?.toTensor()?.dataAsFloatArray
//            scores?.let { processOutput(it) }
//        } catch (e: Exception) {
//            e.printStackTrace()
//            null
//        }
//    }


    fun runInference(imageUri: Uri, inputSize: Int = 224): List<ClassificationResult>? {
        if (module == null) return null

        try {
            val bitmap = loadAndPreprocessImage(imageUri, inputSize)

            // 1. Prepare Direct Buffer (Native Order)
            // 4 bytes per float * 3 channels * H * W
            val tensorBuffer = ByteBuffer.allocateDirect(4 * 3 * inputSize * inputSize)
            tensorBuffer.order(ByteOrder.nativeOrder()) // CRITICAL for XNNPACK
            val floatBuffer = tensorBuffer.asFloatBuffer()

            // 2. Extract Pixels & Normalize
            val pixels = IntArray(inputSize * inputSize)
            bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

            val meanR = 0.485f;
            val meanG = 0.456f;
            val meanB = 0.406f
            val stdR = 0.229f;
            val stdG = 0.224f;
            val stdB = 0.225f

            // Fill buffer in NCHW format (Channels First: RRR...GGG...BBB...)
            // 1.0.1 XNNPACK export usually expects standard PyTorch NCHW unless you changed it.

            // Red Channel
            for (pixel in pixels) {
                floatBuffer.put(((pixel shr 16 and 0xFF) / 255.0f - meanR) / stdR)
            }
            // Green Channel
            for (pixel in pixels) {
                floatBuffer.put(((pixel shr 8 and 0xFF) / 255.0f - meanG) / stdG)
            }
            // Blue Channel
            for (pixel in pixels) {
                floatBuffer.put(((pixel and 0xFF) / 255.0f - meanB) / stdB)
            }

            floatBuffer.rewind() // Reset position to 0

            // 3. Create Tensor
            val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val inputTensor = Tensor.fromBlob(floatBuffer, inputShape)

            // 4. Forward
            val inputEValue = EValue.from(inputTensor)
            val output = module?.forward(inputEValue) // Crash usually happens here

            val scores = output?.get(0)?.toTensor()?.dataAsFloatArray
            return scores?.let { processOutput(it) }

        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }


    private fun loadAndPreprocessImage(imageUri: Uri, targetSize: Int): Bitmap {
        val inputStream = context.contentResolver.openInputStream(imageUri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream?.close()
        return Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
    }


    private fun processOutput(scores: FloatArray): List<ClassificationResult> {
        val expScores = scores.map { Math.exp(it.toDouble()).toFloat() }
        val sumExp = expScores.sum()
        val probabilities = expScores.map { it / sumExp }

        val results = probabilities.mapIndexed { index, prob ->
            val name = if (index < classNames.size) classNames[index] else "Class $index"
            ClassificationResult(className = name, probability = prob)
        }
        return results.sortedByDescending { it.probability }
    }


    fun cleanup() {
        module?.destroy()
        module = null
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists()) {
            file.delete()
        }
//        if (file.exists() && file.length() > 0) {
//            return file.absolutePath
//        }

        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return file.absolutePath
    }

}