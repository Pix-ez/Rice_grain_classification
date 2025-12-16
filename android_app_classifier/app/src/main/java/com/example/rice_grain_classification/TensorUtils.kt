package com.example.rice_grain_classification



import android.graphics.Bitmap
import android.graphics.Color
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

object TensorUtils {
    // ImageNet Normalization Constants
    val NORM_MEAN_RGB = floatArrayOf(0.485f, 0.456f, 0.406f)
    val NORM_STD_RGB = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun bitmapToFloat32Tensor(
        bitmap: Bitmap,
        mean: FloatArray,
        std: FloatArray
    ): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height

        // Allocate direct buffer (Native memory) - crucial for avoiding copies/crashes
        val buffer = ByteBuffer.allocateDirect(4 * 3 * width * height)
        buffer.order(ByteOrder.nativeOrder())
        val floatBuffer = buffer.asFloatBuffer()

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // CHW (Channels-First) Loop Optimization
        // 0 = R, 1 = G, 2 = B
        for (i in 0 until 3) {
            for (j in 0 until width * height) {
                val pixel = pixels[j]

                // Extract channel value (0-255)
                val value = when (i) {
                    0 -> (pixel shr 16) and 0xFF // Red
                    1 -> (pixel shr 8) and 0xFF  // Green
                    else -> pixel and 0xFF       // Blue
                }

                // Normalize: (val/255 - mean) / std
                val normalizedValue = ((value / 255.0f) - mean[i]) / std[i]
                floatBuffer.put(normalizedValue)
            }
        }

        floatBuffer.rewind()
        return floatBuffer
    }
}