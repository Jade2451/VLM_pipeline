# Quick Reference Guide - Enhanced VLM Pipeline

## 📋 What's Been Fixed

✅ **5 Critical Issues Resolved:**

1. ⭐ **Missing Text Embedding Step** (Cell 7)
   - Original notebook could NOT retrieve because text corpus wasn't embedded
   - FAISS would fail without this
   - **NOW FIXED** with complete text embedding implementation

2. ⭐ **No Online RAG Fallback** (Cell 8)
   - Original had no way to get info when local corpus failed
   - **NOW IMPLEMENTED** with PubMed/Wikipedia integration

3. ⭐ **Dataset-Specific Code** (Cell 3)
   - Original only worked with BreakHis dataset
   - **NOW DATA-AGNOSTIC** works with any medical images

4. ⭐ **No Explainability** (Cell 10)
   - Original gave no confidence or uncertainty info
   - **NOW ENHANCED** with confidence levels and clear uncertainty statements

5. ⭐ **Limited Modularity** (Cell 2)
   - Original hard to adapt
   - **NOW MODULAR** with centralized config

---

## 📁 Files Created

1. **UPDATED_CELL_BLOCKS.md** ⭐ MAIN FILE
   - Complete Python code for all 13+ cells
   - Copy-paste ready
   - Heavily commented

2. **README.md**
   - Quick start guide
   - Configuration instructions
   - Troubleshooting

3. **CHANGES_SUMMARY.md**
   - Detailed migration guide
   - Before/after comparison
   - Testing instructions

4. **ARCHITECTURE_DIAGRAMS.md**
   - Visual pipeline flows
   - Decision trees
   - Metrics tracking

5. **This file (QUICK_REFERENCE.md)**
   - Fast lookup for common tasks

---

## 🚀 Quick Start (3 Steps)

### Step 1: Copy Updated Cells
Open `UPDATED_CELL_BLOCKS.md` and copy all cells into your notebook

### Step 2: Configure
In Cell 2, update:
```python
CONFIG = {
    "DATASET_DIR": "/path/to/your/images/",
    "PUBMED_EMAIL": "your.email@example.com",
    "ENABLE_ONLINE_RAG": True,
    "SIMILARITY_THRESHOLD": 0.3,
}
```

### Step 3: Run All Cells
Execute cells 0-13 in order. Done!

---

## 🔑 Key New Features

### 1. Text Embedding (Cell 7) - CRITICAL
```python
# This was completely missing in original!
text_embeddings = embed_texts_batch(prompt_corpus, model, device)
# Returns: (M, 512) normalized embeddings
```

### 2. Online RAG Fallback (Cell 8) - NEW
```python
# Automatically queries PubMed when local corpus insufficient
online_texts, source = online_rag_fallback(img_path, scores, CONFIG)
# Returns: additional context from online sources
```

### 3. Confidence Levels (Cell 10) - NEW
```python
caption, confidence = synthesize_caption_with_explainability(...)
# Returns: caption with confidence level
# Levels: very_high, high, moderate, low, very_low, no_retrieval, error
```

### 4. Data-Agnostic Loading (Cell 3) - NEW
```python
# Works with ANY directory structure
image_paths = load_dataset_generic(CONFIG["DATASET_DIR"], CONFIG)
# Automatically validates and collects images
```

---

## ⚙️ Configuration Cheat Sheet

### Essential Settings
```python
CONFIG = {
    # Required
    "DATASET_DIR": "/path/to/images/",
    "PUBMED_EMAIL": "your@email.com",
    
    # Online RAG
    "ENABLE_ONLINE_RAG": True,          # Enable fallback
    "SIMILARITY_THRESHOLD": 0.3,        # Lower = less fallback
    "ONLINE_RAG_SOURCE": "pubmed",      # or "wikipedia"
    
    # Models (swap as needed)
    "BIOMEDCLIP_HF_ID": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "LLM_ID": "google/flan-t5-small",   # or larger model
    
    # Performance
    "DEVICE": "cuda",                    # or "cpu"
    "BATCH_IMAGE": 8,                    # Reduce if OOM
    "BATCH_TEXT": 128,
}
```

### Adjustment Guide
| Want to...                    | Change...                      | To...        |
|-------------------------------|--------------------------------|--------------|
| Use less online fallback      | `SIMILARITY_THRESHOLD`         | Higher (0.5) |
| Use more online fallback      | `SIMILARITY_THRESHOLD`         | Lower (0.2)  |
| Disable online RAG            | `ENABLE_ONLINE_RAG`            | False        |
| Use Wikipedia instead         | `ONLINE_RAG_SOURCE`            | "wikipedia"  |
| Better captions               | `LLM_ID`                       | Larger model |
| Faster processing             | `BATCH_IMAGE`                  | Larger batch |
| Less memory usage             | `BATCH_IMAGE`, `BATCH_TEXT`    | Smaller      |

---

## 📊 Output Files Reference

After running, check:

```
vlm_rag_outputs/
├── image_embeddings.npy              ← Cached image features
├── text_embeddings.npy               ← Cached text features (NEW!)
├── retrieval_results.jsonl           ← Detailed retrieval with online info
├── synthesized_captions.jsonl        ← Captions with confidence (NEW!)
├── synthesized_captions_rescored.csv ← Easy to review in Excel
└── run_metadata.json                 ← Statistics and metrics
```

### Key Columns in CSV
- `generated_caption`: The actual caption text
- `confidence_level`: very_high, high, moderate, low, very_low
- `caption_image_similarity`: How well caption matches image (0-1)
- `local_max_score`: Best retrieval score from local corpus
- `online_source`: Was online fallback used? Which source?

---

## 🎯 Confidence Levels Explained

| Level      | Score Range | Meaning                           | Action Needed?    |
|------------|-------------|-----------------------------------|-------------------|
| very_high  | > 0.7       | Excellent corpus match            | ✅ Trust it       |
| high       | 0.5 - 0.7   | Good corpus match                 | ✅ Trust it       |
| moderate   | 0.3 - 0.5   | Acceptable match                  | ✅ Review if critical |
| low        | 0.15 - 0.3  | Weak match, online likely used    | ⚠️ Manual review  |
| very_low   | < 0.15      | Very uncertain                    | ❌ Manual review  |
| no_retrieval | N/A       | No relevant info found            | ❌ Manual caption |
| error      | N/A         | Processing error                  | ❌ Check logs     |

---

## 🔍 Troubleshooting Quick Fixes

### Issue: "text_embeddings is None"
**Fix**: Make sure Cell 7 is included and run
```python
# Check if Cell 7 exists in your notebook
text_embeddings = np.load("vlm_rag_outputs/text_embeddings.npy")
```

### Issue: "PubMed search error"
**Fix**: Check these in order:
1. `CONFIG["PUBMED_EMAIL"]` is set
2. Internet access available
3. Biopython installed: `pip install biopython`
4. Try Wikipedia instead: `CONFIG["ONLINE_RAG_SOURCE"] = "wikipedia"`

### Issue: All captions have "very_low" confidence
**Fix**: Adjust threshold or corpus:
```python
# Lower threshold to reduce fallback
CONFIG["SIMILARITY_THRESHOLD"] = 0.2

# Or improve corpus by adding more relevant texts
CONFIG["CORPUS_PATH"] = "/path/to/better/corpus.txt"
```

### Issue: "No image files found"
**Fix**: Check path and settings:
```python
# Verify path
import os
os.path.exists(CONFIG["DATASET_DIR"])  # Should be True

# Enable recursive search
CONFIG["RECURSIVE_SEARCH"] = True

# Check supported formats
CONFIG["SUPPORTED_IMAGE_FORMATS"]  # Does it include your format?
```

### Issue: Out of memory
**Fix**: Reduce batch sizes:
```python
CONFIG["BATCH_IMAGE"] = 4   # Reduce from 8
CONFIG["BATCH_TEXT"] = 64   # Reduce from 128
CONFIG["DEVICE"] = "cpu"    # Use CPU if GPU OOM
```

---

## 📝 Cell-by-Cell Checklist

Use this to verify you have all components:

```
□ Cell 0:  Install Dependencies (updated with biopython)
□ Cell 1:  Imports (updated with online RAG imports)
□ Cell 2:  Configuration (ENHANCED - modular config)
□ Cell 3:  Dataset Loading (ENHANCED - data-agnostic)
□ Cell 4:  Load BiomedCLIP (updated with better errors)
□ Cell 5:  Image Embeddings (updated with better error handling)
□ Cell 6:  Load Text Corpus (updated for data-agnostic)
□ Cell 7:  Text Embeddings ⭐ CRITICAL - WAS MISSING
□ Cell 8:  Online RAG Setup ⭐ NEW FEATURE
□ Cell 9:  FAISS Retrieval (ENHANCED - with online fallback)
□ Cell 10: LLM Synthesis (ENHANCED - with explainability)
□ Cell 11: Re-score Captions (updated with confidence tracking)
□ Cell 12: Visualizations (ENHANCED - with confidence colors)
□ Cell 13: Save Metadata (ENHANCED - with new metrics)
```

**Missing Cell 7 or 8?** Pipeline won't work correctly!

---

## 🎨 Understanding Visualizations

### Confidence-Coded Gallery
After Cell 12, you'll see images with color-coded titles:
- 🟢 **Green**: very_high or high confidence
- 🟡 **Yellow**: moderate confidence  
- 🟠 **Orange**: low confidence
- 🔴 **Red**: very_low confidence
- 🟣 **Purple**: error

Review red/orange ones manually!

### Histogram Interpretation
- Most scores > 0.5: Good corpus quality ✅
- Many scores < 0.3: Need better corpus or more online fallback ⚠️
- Red line: Similarity threshold (where fallback triggers)

---

## 🧪 Testing Your Setup

### Quick Test (5 minutes)
```python
# 1. Set small test dataset
CONFIG["DATASET_DIR"] = "/path/to/10-images/"

# 2. Enable online RAG
CONFIG["ENABLE_ONLINE_RAG"] = True

# 3. Run all cells

# 4. Check outputs
assert Path("vlm_rag_outputs/text_embeddings.npy").exists()  # NEW!
assert Path("vlm_rag_outputs/synthesized_captions.jsonl").exists()

# 5. Review CSV
df = pd.read_csv("vlm_rag_outputs/synthesized_captions_rescored.csv")
print(df[["confidence_level", "online_source"]].value_counts())
```

### Verification Checklist
✅ All cells run without errors  
✅ `text_embeddings.npy` exists (was missing before!)  
✅ Some captions have confidence levels  
✅ `online_source` column shows fallback usage  
✅ Visualizations show color-coded confidence  

---

## 📞 Getting Help

1. **Check error logs** in notebook output
2. **Review** `run_metadata.json` for metrics
3. **Read detailed docs**:
   - `UPDATED_CELL_BLOCKS.md` - Full code
   - `CHANGES_SUMMARY.md` - Migration guide
   - `ARCHITECTURE_DIAGRAMS.md` - Visual flows
4. **Open GitHub issue** with:
   - Error message
   - Config settings used
   - Dataset size/structure

---

## 🎯 Common Use Cases

### Use Case 1: Process New Dataset
```python
# Just change path - everything else is automatic
CONFIG["DATASET_DIR"] = "/path/to/new/dataset/"
# Run all cells - that's it!
```

### Use Case 2: Improve Caption Quality
```python
# Option A: Better corpus
CONFIG["CORPUS_PATH"] = "/path/to/domain-specific-corpus.txt"

# Option B: Better LLM
CONFIG["LLM_ID"] = "google/flan-t5-large"  # Larger model

# Option C: More online context
CONFIG["MAX_ONLINE_RESULTS"] = 5
```

### Use Case 3: Debug Low Confidence
```python
# Enable online fallback
CONFIG["ENABLE_ONLINE_RAG"] = True

# Lower threshold to use fallback more
CONFIG["SIMILARITY_THRESHOLD"] = 0.2

# Check which images have issues
df = pd.read_csv("vlm_rag_outputs/synthesized_captions_rescored.csv")
low_conf = df[df["confidence_level"].isin(["low", "very_low"])]
print(low_conf[["image_path", "local_max_score", "online_source"]])
```

### Use Case 4: Disable Online RAG (Offline Mode)
```python
CONFIG["ENABLE_ONLINE_RAG"] = False
# All processing stays local
```

---

## 📈 Metrics to Monitor

After running, check these in `run_metadata.json`:

```python
meta = json.load(open("vlm_rag_outputs/run_metadata.json"))

# Key metrics:
print(f"Fallback rate: {meta['fallback_rate']:.1f}%")
# < 20%: Good corpus
# > 50%: Consider better corpus or lower threshold

print(f"Avg caption similarity: {meta['avg_caption_similarity']:.3f}")
# > 0.4: Good quality
# < 0.3: Review captions

print(f"Confidence distribution: {meta['confidence_distribution']}")
# Want: mostly "high" and "moderate"
# Bad: mostly "low" or "very_low"
```

---

## ✨ Quick Win Features

### 1. Automatic Caching
Everything is cached automatically:
```python
# First run: Computes embeddings (slow)
# Second run: Loads from cache (fast!)
# To force recompute: delete .npy files
```

### 2. Source Attribution
Every caption tracks its sources:
```python
with open("vlm_rag_outputs/synthesized_captions.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        print(f"Image: {obj['image_path']}")
        print(f"  Local texts: {len(obj['local_prompts'])}")
        print(f"  Online texts: {len(obj.get('online_prompts', []))}")
        print(f"  Confidence: {obj['confidence_level']}")
```

### 3. Uncertainty Detection
Model explicitly states when unsure:
```python
# Examples of uncertainty statements:
"Unable to generate caption: No relevant reference material found"
"[Low confidence retrieval] ..."
"[Note: Limited information available. Consider manual review.]"
```

---

## 🔄 Migration from Original

If you have the original notebook:

1. **Backup first**: Save a copy
2. **Open** `UPDATED_CELL_BLOCKS.md`
3. **Replace cells** one by one (or all at once)
4. **Update config** in Cell 2
5. **Run all** cells in order
6. **Verify** outputs look correct

**Critical**: Don't skip Cell 7 (text embeddings) or Cell 8 (online RAG)!

---

## 📚 Documentation Map

- **QUICK_REFERENCE.md** (this file) - Fast lookup
- **UPDATED_CELL_BLOCKS.md** - Complete code (MAIN FILE)
- **README.md** - Project overview & quick start
- **CHANGES_SUMMARY.md** - Detailed before/after comparison
- **ARCHITECTURE_DIAGRAMS.md** - Visual pipeline flows

**Start here**: UPDATED_CELL_BLOCKS.md → Copy cells → Configure → Run!

---

## ✅ Final Checklist Before Running

- [ ] Copied all cells from UPDATED_CELL_BLOCKS.md
- [ ] Updated `DATASET_DIR` in CONFIG
- [ ] Updated `PUBMED_EMAIL` in CONFIG
- [ ] Installed dependencies (`biopython`, `requests`)
- [ ] Have internet access (if using online RAG)
- [ ] Have GPU available (or set `DEVICE = "cpu"`)
- [ ] Verified Cell 7 (text embeddings) is present ⭐
- [ ] Verified Cell 8 (online RAG) is present ⭐

**Ready to run!** Execute all cells in order.

---

**Last Updated**: 2024
**Pipeline Version**: 2.0 (Enhanced)
**Status**: Production Ready ✅
