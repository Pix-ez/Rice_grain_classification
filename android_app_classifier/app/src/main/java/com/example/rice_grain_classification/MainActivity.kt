package com.example.rice_grain_classification

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.rice_grain_classification.ui.theme.Rice_grain_classificationTheme

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.rice_grain_classification.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext



class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var modelInference: ModelInference
    private lateinit var resultAdapter: ResultAdapter

    private var selectedImageUri: Uri? = null
    private var currentModel: String? = null
    private lateinit var performanceMonitor: PerformanceMonitor
    companion object {
        init {
            try {
                // Force load the XNNPACK JNI library
                System.loadLibrary("executorch_xnnpack")
            } catch (e: UnsatisfiedLinkError) {
                // If this fails, it might be named slightly differently in 1.0.1
                // or already loaded. It is safe to ignore if Module.load works.
                e.printStackTrace()
            }
        }
    }

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            result.data?.data?.let { uri ->
                selectedImageUri = uri
                binding.imageView.setImageURI(uri)
                binding.predictButton.isEnabled = true
                binding.noImageText.visibility = View.GONE
                binding.imageView.visibility = View.VISIBLE

                // Clear previous results
                resultAdapter.updateResults(emptyList())
                binding.resultsCard.visibility = View.GONE
            }
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openImagePicker()
        } else {
            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        modelInference = ModelInference(this)
        performanceMonitor = PerformanceMonitor(this)
        setupRecyclerView()
        setupModelSpinner()
        setupClickListeners()
    }

    private fun setupRecyclerView() {
        resultAdapter = ResultAdapter()
        binding.resultsRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = resultAdapter
        }
    }

    private fun setupModelSpinner() {
        val models = modelInference.getAvailableModels()
        // Sort models so "portable" (safer) ones come first if possible
        val sortedModels = models.sortedBy { if (it.contains("portable")) 0 else 1 }

        if (sortedModels.isEmpty()) {
            Toast.makeText(this, "No models found", Toast.LENGTH_LONG).show()
            binding.modelSpinner.isEnabled = false
            return
        }

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, sortedModels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.modelSpinner.adapter = adapter

        binding.modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            // Add a flag to prevent the initial auto-crash if needed,
            // OR just ensure the sortedModels[0] is the safe portable one.
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedModel = sortedModels[position]
                loadModel(selectedModel)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupClickListeners() {
        binding.selectImageButton.setOnClickListener {
            checkPermissionAndOpenPicker()
        }

        binding.predictButton.setOnClickListener {
            selectedImageUri?.let { uri ->
                runInference(uri)
            }
        }
    }

    private fun checkPermissionAndOpenPicker() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        when {
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED -> {
                openImagePicker()
            }
            else -> {
                permissionLauncher.launch(permission)
            }
        }
    }

    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        imagePickerLauncher.launch(intent)
    }

    private fun loadModel(modelName: String) {
        binding.progressBar.visibility = View.VISIBLE
        binding.statusTextView.text = "Loading model..."
        binding.selectImageButton.isEnabled = false

        CoroutineScope(Dispatchers.IO).launch {
            val success = modelInference.loadModel(modelName)

            withContext(Dispatchers.Main) {
                binding.progressBar.visibility = View.GONE

                if (success) {
                    currentModel = modelName
                    binding.statusTextView.text = "Model loaded: $modelName"
                    binding.selectImageButton.isEnabled = true
                } else {
                    binding.statusTextView.text = "Failed to load model"
                    Toast.makeText(this@MainActivity, "Failed to load model", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun runInference(imageUri: Uri) {
        binding.progressBar.visibility = View.VISIBLE
        binding.statusTextView.text = "Running inference..."
        binding.predictButton.isEnabled = false
        binding.selectImageButton.isEnabled = false


        binding.resultsCard.visibility = View.GONE

        CoroutineScope(Dispatchers.IO).launch {


            // --- START MONITORING ---
            performanceMonitor.startMonitoring()

            val startTime = System.currentTimeMillis()
            val results = modelInference.runInference(imageUri)
            val inferenceTime = System.currentTimeMillis() - startTime

            // --- STOP MONITORING ---
            val stats = performanceMonitor.stopMonitoring()

            withContext(Dispatchers.Main) {
                binding.progressBar.visibility = View.GONE
                binding.predictButton.isEnabled = true
                binding.selectImageButton.isEnabled = true

                if (results != null && results.isNotEmpty()) {
                    binding.statusTextView.text = "Inference completed in ${inferenceTime}ms"
                    resultAdapter.updateResults(results)

                    // Show Prediction Results
                    resultAdapter.updateResults(results)

                    // Show Performance Stats
                    binding.performanceStatsText.text = stats.getFormattedStats()

                    binding.resultsCard.visibility = View.VISIBLE
                } else {
                    binding.statusTextView.text = "Inference failed"
                    Toast.makeText(this@MainActivity, "Inference failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        modelInference.cleanup()
    }
}