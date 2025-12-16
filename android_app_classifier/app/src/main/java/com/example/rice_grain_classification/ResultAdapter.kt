package com.example.rice_grain_classification
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ProgressBar
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ResultAdapter : RecyclerView.Adapter<ResultAdapter.ResultViewHolder>() {

    private var results: List<ClassificationResult> = emptyList()

    fun updateResults(newResults: List<ClassificationResult>) {
        results = newResults
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ResultViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_result, parent, false)
        return ResultViewHolder(view)
    }

    override fun onBindViewHolder(holder: ResultViewHolder, position: Int) {
        holder.bind(results[position], position + 1)
    }

    override fun getItemCount() = results.size

    class ResultViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val rankTextView: TextView = itemView.findViewById(R.id.rankTextView)
        private val classNameTextView: TextView = itemView.findViewById(R.id.classNameTextView)
        private val probabilityTextView: TextView = itemView.findViewById(R.id.probabilityTextView)
        private val progressBar: ProgressBar = itemView.findViewById(R.id.probabilityProgressBar)

        fun bind(result: ClassificationResult, rank: Int) {
            rankTextView.text = "#$rank"
            classNameTextView.text = result.className
            probabilityTextView.text = result.probabilityPercent
            progressBar.progress = (result.probability * 100).toInt()
        }
    }
}