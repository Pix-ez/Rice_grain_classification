package com.example.rice_grain_classification


data class ClassificationResult(
    val className: String,
    val probability: Float
) {
    val probabilityPercent: String
        get() = String.format("%.2f%%", probability * 100)
}