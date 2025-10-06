# Pipeline Architecture Diagrams

## 1. Original Pipeline (Incomplete)

```
┌──────────────────┐
│  Input Images    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Vision Encoder   │
│  (BiomedCLIP)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Image Embeddings │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ FAISS Retrieval  │ ❌ BROKEN: No text embeddings!
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Retrieved Texts  │ ❌ Cannot retrieve without embeddings
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Synthesis   │ ❌ Poor quality without good retrieval
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Captions     │ ❌ No confidence, no explainability
└──────────────────┘
```

**Problems**:
- ❌ Missing text embedding step
- ❌ No fallback when corpus insufficient
- ❌ Hardcoded for specific dataset
- ❌ No confidence reporting
- ❌ No source attribution

---

## 2. Enhanced Pipeline (Complete)

```
┌──────────────────────────────────────────────────────────────┐
│                    CONFIGURATION                             │
│  • Dataset: Data-agnostic loading                            │
│  • Models: Modular (easy to swap)                            │
│  • RAG: Online fallback enabled                              │
│  • Thresholds: Configurable                                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────┐
         │                                │
    ┌────▼─────┐                    ┌────▼────────┐
    │  Images  │                    │ Text Corpus │
    └────┬─────┘                    └────┬────────┘
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌──────────────────┐
│ Vision Encoder  │           │ Text Encoder     │
│  (BiomedCLIP)   │           │  (BiomedCLIP)    │
└────────┬────────┘           └────────┬─────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐           ┌──────────────────┐
│ Image Embeddings│           │ Text Embeddings  │ ✅ NEW!
│   (Cached)      │           │   (Cached)       │
└────────┬────────┘           └────────┬─────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
           ┌─────────────────┐
           │  FAISS Index    │ ✅ NOW WORKS!
           │  (Cosine Sim)   │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Local Retrieval │
           │   (Top-K texts) │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────────────┐
           │ Check Similarity Score  │
           └────────┬────────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
    High Score            Low Score
    (> threshold)        (< threshold)
          │                    │
          │                    ▼
          │           ┌──────────────────┐
          │           │ Online RAG       │ ✅ NEW!
          │           │ Fallback         │
          │           │ - PubMed Query   │
          │           │ - Wikipedia      │
          │           └────────┬─────────┘
          │                    │
          └─────────┬──────────┘
                    │
                    ▼
           ┌─────────────────────────┐
           │ Combined Texts          │
           │ (Local + Online)        │
           │ + Source Attribution    │ ✅ NEW!
           └────────┬────────────────┘
                    │
                    ▼
           ┌─────────────────────────┐
           │ LLM Synthesis           │
           │ (Flan-T5 / GPT)         │
           │ + Confidence Calc       │ ✅ ENHANCED!
           └────────┬────────────────┘
                    │
                    ▼
           ┌─────────────────────────┐
           │ Explainable Caption     │ ✅ ENHANCED!
           │ - Generated text        │
           │ - Confidence level      │
           │ - Source attribution    │
           │ - Uncertainty notes     │
           └────────┬────────────────┘
                    │
                    ▼
           ┌─────────────────────────┐
           │ Output Files            │
           │ - JSONL (detailed)      │
           │ - CSV (summary)         │
           │ - Visualizations        │
           │ - Metadata              │
           └─────────────────────────┘
```

**Improvements**:
- ✅ Complete text embedding step
- ✅ Online RAG fallback system
- ✅ Data-agnostic design
- ✅ Confidence reporting at every level
- ✅ Full source attribution
- ✅ Enhanced explainability

---

## 3. Online RAG Fallback Decision Flow

```
┌─────────────────────────────────────┐
│  Image Retrieved from Local Corpus  │
│  Top-K Texts + Similarity Scores    │
└────────────────┬────────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ Max Score >= 0.3?    │ (Configurable threshold)
      └──────┬───────┬───────┘
             │       │
         YES │       │ NO
             │       │
             ▼       ▼
    ┌─────────┐   ┌─────────────────────┐
    │ Use     │   │ Trigger Online RAG  │
    │ Local   │   │ Fallback            │
    │ Only    │   └──────────┬──────────┘
    └────┬────┘              │
         │                   ▼
         │          ┌──────────────────────┐
         │          │ Generate Query       │
         │          │ (from image path +   │
         │          │  domain keywords)    │
         │          └──────────┬───────────┘
         │                     │
         │                     ▼
         │          ┌──────────────────────┐
         │          │ Select Source:       │
         │          │ - PubMed             │
         │          │ - Wikipedia          │
         │          └──────────┬───────────┘
         │                     │
         │                     ▼
         │          ┌──────────────────────┐
         │          │ API Request          │
         │          │ (with timeout/retry) │
         │          └──────────┬───────────┘
         │                     │
         │          ┌──────────┴──────────┐
         │          │                     │
         │       Success              Failure
         │          │                     │
         │          ▼                     ▼
         │  ┌───────────────┐    ┌──────────────┐
         │  │ Extract       │    │ Log Error    │
         │  │ Top-N         │    │ Continue     │
         │  │ Abstracts     │    │ with Local   │
         │  └───────┬───────┘    └──────┬───────┘
         │          │                    │
         └──────────┴────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Combine:             │
         │ - Local texts        │
         │ - Online texts       │
         │ - Source tags        │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Send to LLM          │
         │ with full context    │
         └──────────────────────┘
```

---

## 4. Confidence Level Calculation

```
Retrieval Similarity Score (Max of Top-K)
│
├─ [0.0 - 0.15)  → very_low    🔴
│                   "Unable to generate caption..."
│
├─ [0.15 - 0.3)  → low         🟠
│                   "[Low confidence] ..."
│                   Triggers online RAG
│
├─ [0.3 - 0.5)   → moderate    🟡
│                   May use online RAG
│
├─ [0.5 - 0.7)   → high        🟢
│                   Good local match
│
└─ [0.7 - 1.0]   → very_high   🟢
                    Strong local match

Special Cases:
- no_retrieval   → No texts retrieved at all
- error          → Processing error occurred
```

---

## 5. Data Flow Through Cells

```
Cell 0: Install Dependencies
  ↓
Cell 1: Imports & Setup
  ↓
Cell 2: Configuration
  ↓
Cell 3: Load Dataset (Data-Agnostic)
  ├─ Collect images recursively
  ├─ Validate image files
  └─ Returns: image_paths[]
  ↓
Cell 4: Load BiomedCLIP
  └─ Returns: model, preprocess
  ↓
Cell 5: Compute Image Embeddings
  ├─ Check cache
  ├─ Batch encode images
  ├─ Normalize embeddings
  └─ Returns: image_embeddings (N × D)
  ↓
Cell 6: Load Text Corpus
  ├─ Load from file/directory
  └─ Returns: prompt_corpus[]
  ↓
Cell 7: Compute Text Embeddings ⭐ NEW
  ├─ Check cache
  ├─ Batch encode texts
  ├─ Normalize embeddings
  └─ Returns: text_embeddings (M × D)
  ↓
Cell 8: Setup Online RAG ⭐ NEW
  ├─ Configure PubMed API
  ├─ Configure Wikipedia API
  └─ Returns: online_rag_fallback()
  ↓
Cell 9: FAISS Retrieval + Online Fallback
  ├─ Build FAISS index
  ├─ Query all images
  ├─ Check similarity threshold
  ├─ Trigger online fallback if needed
  └─ Returns: retrieval_results.jsonl
  ↓
Cell 10: LLM Synthesis with Explainability
  ├─ Calculate confidence
  ├─ Build context (local + online)
  ├─ Generate caption
  ├─ Add uncertainty statements
  └─ Returns: synthesized_captions.jsonl
  ↓
Cell 11: Re-score Captions
  ├─ Compute caption-image similarity
  └─ Returns: synthesized_captions_rescored.csv
  ↓
Cell 12: Visualizations
  ├─ Confidence distribution
  ├─ Sample gallery
  └─ Returns: PNG files
  ↓
Cell 13: Save Metadata
  └─ Returns: run_metadata.json
```

---

## 6. File Outputs Structure

```
vlm_rag_outputs/
│
├── Embeddings (Cached)
│   ├── image_embeddings.npy       (N × 512 float32)
│   ├── image_paths.json           (List of paths)
│   ├── text_embeddings.npy        (M × 512 float32) ⭐ NEW
│   └── text_corpus.json           (List of texts)  ⭐ NEW
│
├── Retrieval Results
│   ├── retrieval_results.jsonl    (Detailed, with online info)
│   └── retrieval_summary.csv      (Quick view)
│
├── Generated Captions
│   ├── synthesized_captions.jsonl           (With confidence)
│   └── synthesized_captions_rescored.csv    (With scores)
│
├── Visualizations
│   ├── sample_gallery_with_confidence.png   (Color-coded)
│   └── confidence_distribution.png
│
└── Metadata
    └── run_metadata.json          (Complete statistics)
```

---

## 7. Key Configuration Parameters

```
┌─────────────────────────────────────────────────────────┐
│                  CONFIG PARAMETERS                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Dataset Settings (Data-Agnostic)                       │
│  ├─ DATASET_DIR           → Path to any image folder   │
│  ├─ SUPPORTED_FORMATS     → File extensions to process │
│  └─ RECURSIVE_SEARCH      → Search subdirectories?     │
│                                                         │
│  Model Settings (Modular)                               │
│  ├─ BIOMEDCLIP_HF_ID     → Vision-language model       │
│  ├─ LLM_ID               → Text generation model       │
│  └─ DEVICE               → "cuda" or "cpu"             │
│                                                         │
│  RAG Settings (Online Fallback)                         │
│  ├─ ENABLE_ONLINE_RAG    → true/false                  │
│  ├─ ONLINE_RAG_SOURCE    → "pubmed" or "wikipedia"     │
│  ├─ SIMILARITY_THRESHOLD → When to trigger (0-1)       │
│  ├─ MAX_ONLINE_RESULTS   → # of online abstracts       │
│  └─ PUBMED_EMAIL         → Required for PubMed API     │
│                                                         │
│  Performance Settings                                   │
│  ├─ BATCH_IMAGE          → Image batch size            │
│  ├─ BATCH_TEXT           → Text batch size             │
│  └─ TOP_K                → # of retrieval results      │
│                                                         │
│  Corpus Settings                                        │
│  └─ CORPUS_PATH          → Custom corpus (optional)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Error Handling Flow

```
At Each Stage:
│
├─ Image Loading
│  ├─ Try: Load image
│  ├─ Except: Skip image, log error
│  └─ Continue: with valid images
│
├─ Image Embedding
│  ├─ Try: Encode batch
│  ├─ Except: Add zero embedding, log error
│  └─ Continue: with available embeddings
│
├─ Text Embedding
│  ├─ Try: Encode batch
│  ├─ Except: Add zero embedding, log error
│  └─ Continue: with available embeddings
│
├─ Online RAG
│  ├─ Try: API request
│  ├─ Except: Log error, use local only
│  └─ Continue: with available texts
│
└─ LLM Synthesis
   ├─ Try: Generate caption
   ├─ Except: Return error message with score info
   └─ Continue: mark as "error" confidence
```

---

## 9. Monitoring & Metrics

```
Key Metrics to Track:
│
├─ Retrieval Quality
│  ├─ Average Top-1 Score     (Should be > 0.4)
│  ├─ Score Distribution      (Check histogram)
│  └─ Low Score Count         (< threshold)
│
├─ Online RAG Usage
│  ├─ Fallback Rate          (% images using online)
│  ├─ Fallback Success Rate  (% successful queries)
│  └─ Sources Used           (PubMed vs Wikipedia)
│
├─ Caption Quality
│  ├─ Caption-Image Similarity   (Should be > 0.4)
│  ├─ Confidence Distribution    (More high/moderate)
│  └─ Uncertainty Rate           (% with disclaimers)
│
└─ Performance
   ├─ Processing Time per Image
   ├─ Cache Hit Rate
   └─ Batch Efficiency
```

All tracked in: `run_metadata.json`

---

## 10. Comparison: Before vs After

| Feature                    | Original | Enhanced |
|----------------------------|----------|----------|
| Text Embedding             | ❌ Missing | ✅ Cell 7 |
| Online Fallback            | ❌ No      | ✅ Cell 8 |
| Data-Agnostic              | ❌ No      | ✅ Cell 3 |
| Confidence Reporting       | ❌ No      | ✅ Cell 10 |
| Source Attribution         | ❌ No      | ✅ Yes |
| Uncertainty Handling       | ❌ No      | ✅ Yes |
| Error Recovery             | ⚠️ Basic   | ✅ Robust |
| Caching                    | ⚠️ Partial | ✅ Complete |
| Logging                    | ⚠️ Basic   | ✅ Detailed |
| Modularity                 | ⚠️ Limited | ✅ High |
| Explainability             | ❌ None    | ✅ Full |

**Result**: Production-ready pipeline with complete RAG capability
