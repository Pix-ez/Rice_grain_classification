package com.example.rice_grain_classification


import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process
import android.util.Log
import java.io.File
import java.io.RandomAccessFile

data class PerformanceStats(
    val inferenceTimeMs: Long,
    val cpuUsagePercent: Float,
    val memoryUsageMB: Float,
    val peakMemoryMB: Float
) {
    // A nice formatted string for the UI
    fun getFormattedStats(): String {
        return "Time: ${inferenceTimeMs}ms  |  CPU: %.1f%%  |  Mem: %.1f MB".format(cpuUsagePercent, memoryUsageMB)
    }
}

class PerformanceMonitor(private val context: Context) {

    private var startTime: Long = 0
    private var startMemory: Long = 0
    private var cpuUsageStart: Long = 0
    private var totalCpuTimeStart: Long = 0

    fun startMonitoring() {
        startTime = System.currentTimeMillis()
        startMemory = getUsedMemoryMB().toLong()

        // Get CPU stats
        val cpuStats = getCpuStats()
        cpuUsageStart = cpuStats.first
        totalCpuTimeStart = cpuStats.second
    }

    fun stopMonitoring(): PerformanceStats {
        val endTime = System.currentTimeMillis()
        val inferenceTime = endTime - startTime

        // Memory stats
        val currentMemory = getUsedMemoryMB()
        val peakMemory = getPeakMemoryMB()

        // CPU stats
        val cpuStats = getCpuStats()
        val cpuUsageEnd = cpuStats.first
        val totalCpuTimeEnd = cpuStats.second

        val cpuUsage = calculateCpuUsage(
            cpuUsageStart, cpuUsageEnd,
            totalCpuTimeStart, totalCpuTimeEnd
        )

        return PerformanceStats(
            inferenceTimeMs = inferenceTime,
            cpuUsagePercent = cpuUsage,
            memoryUsageMB = currentMemory,
            peakMemoryMB = peakMemory
        )
    }

    private fun getUsedMemoryMB(): Float {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return memoryInfo.totalPss / 1024f // Convert KB to MB
    }

    private fun getPeakMemoryMB(): Float {
        val runtime = Runtime.getRuntime()
        val used = (runtime.totalMemory() - runtime.freeMemory())
        return used / (1024f * 1024f)
    }

    private fun getCpuStats(): Pair<Long, Long> {
        try {
            val pid = Process.myPid()
            val statFile = File("/proc/$pid/stat")

            if (statFile.exists()) {
                val reader = RandomAccessFile(statFile, "r")
                val stat = reader.readLine()
                reader.close()

                val stats = stat.split(" ")
                val utime = stats[13].toLongOrNull() ?: 0L
                val stime = stats[14].toLongOrNull() ?: 0L
                val processCpuTime = utime + stime
                val totalCpuTime = getTotalCpuTime()

                return Pair(processCpuTime, totalCpuTime)
            }
        } catch (e: Exception) {
            // Permission denied on some newer Androids
        }
        return Pair(0L, 0L)
    }

    private fun getTotalCpuTime(): Long {
        try {
            val reader = RandomAccessFile("/proc/stat", "r")
            val load = reader.readLine()
            reader.close()
            val toks = load.split(" ").filter { it.isNotEmpty() }
            return toks.slice(1..4).sumOf { it.toLongOrNull() ?: 0L }
        } catch (e: Exception) { }
        return 0L
    }

    private fun calculateCpuUsage(start: Long, end: Long, totalStart: Long, totalEnd: Long): Float {
        val delta = end - start
        val totalDelta = totalEnd - totalStart
        return if (totalDelta > 0) (delta.toFloat() / totalDelta.toFloat()) * 100f else 0f
    }
}