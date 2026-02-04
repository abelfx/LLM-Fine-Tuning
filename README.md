# Efficient LLM Fine-Tuning: Microsoft Phi-2 with QLoRA

A professional demonstration of fine-tuning the **Microsoft Phi-2** (1.5B parameters) Large Language Model for targeted task adaptation. This project utilizes **QLoRA** (Quantized Low-Rank Adaptation) to achieve high-performance results on consumer hardware, focusing on specific **Business & IT scenarios** (Agile standups, Incident reports, HR interviews).

## Project Highlights

*   **Model:** [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
*   **Technique:** 4-bit Quantization (BitsAndBytes) + LoRA (PEFT)
*   **Task:** Dialogue Summarization
*   **Dataset:** Hugging Face `dialogsum` + Custom Enterprise Scenarios
*   **Result:** Reduced training loss from **~1.67** to **~1.10**, demonstrating significant capability gains in context retention and summary conciseness.

## Evaluation & Metrics

The model was evaluated using both quantitative metrics and qualitative "stress tests".

### 1. Quantitative Improvement
We compared the zero-shot performance of the Base Phi-2 model against our Fine-Tuned adapter using ROUGE, BLEU, METEOR, and BERTScore (Semantic Similarity).

| Metric | Improvement | Notes |
|:---|:---|:---|
| **BERTScore F1** | **+2.58%** | Higher semantic understanding of the summary content. |
| **ROUGE-1** | **+3.98%** | Significant improvement in unigram overlap. |
| **ROUGE-L** | **+2.72%** | Better capturing of the longest common subsequence structure. |
| **Length Ratio**| **-22.14%** | The fine-tuned model produces more concise summaries (less verbose). |
| **Training Loss** | **1.67 -> 1.10** | Smooth convergence over 1000 steps without overfitting artifacts. |

### 2. Qualitative "Stress Test" (The Demo)
We tested the model on complex logic puzzles where simple extraction fails.

**Scenario:** A meeting scheduling conflict with multiple shifts (Monday -> Tuesday -> Wednesday).
*   **Base Model:** Often gets confused, listing cancelled times or failing to find the final agreement.
*   **Fine-Tuned Model:** Correctly identifies the **final agreed time** (Wednesday @ 11 AM) and captures the specific action item (Alice bringing the Q2 report), ignoring the "red herring" cancelled appointments.

## Technical Implementation

### Optimized Training Pipeline
The training arguments were specifically tuned for a balance of speed and granular reporting:
*   **`logging_steps=25`**: To capture the critical early-stage learning curve (the initial "drop").
*   **`eval_steps=100`**: Decoupled from logging to reduced validation downtime while maintaining oversight.
*   **`group_by_length=True`**: Dramatically increases training speed by batching similar-length sequences (loss "zigzag" artifacts were smoothed in visualization using a rolling window of 6).

### Visualization
Custom Matplotlib implementation handles the `group_by_length` oscillation, presenting a clean, professional learning curve suitable for stakeholder reporting.

## Tech Stack

*   **Base Framework:** `transformers` (Hugging Face), `pytorch`
*   **Optimization:** `peft` (Parameter-Efficient Fine-Tuning), `bitsandbytes` (Quantization)
*   **Training Loop:** `trl` (Transformer Reinforcement Learning - SFTTrainer)
*   **Metrics:** `evaluate`, `rouge_score`, `bert_score`

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -q -U bitsandbytes transformers peft accelerate datasets scipy einops evaluate trl rouge_score bert_score
    ```

2.  **Run the Notebook:**
    Open `llm-finetuning-lora.ipynb` and execute all cells. 
    *   *Note:* The "Interactive Demo" cell at the end allows you to input custom conversations to test the model live.

3.  **View Results:**
    *   Training loss graphs will generate inline.
    *   Evaluation tables (Pandas DataFrames) show side-by-side comparisons of Human vs. Base vs. Fine-Tuned summaries.

---
