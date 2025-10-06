# VLM Pipeline - Enhanced Medical Image Captioning

A data-agnostic, explainable, and adaptable Vision-Language Model (VLM) pipeline for medical image captioning with RAG (Retrieval-Augmented Generation) capabilities.

## Overview

This pipeline processes medical images and generates informed captions by:
1. Encoding images with BiomedCLIP
2. Retrieving relevant biomedical text from local corpus
3. Falling back to online sources (PubMed/Wikipedia) when needed
4. Synthesizing grounded captions with an LLM
5. Providing confidence levels and explainability

## Key Features

### ✓ Data-Agnostic
- Works with **any medical image dataset**
- Auto-detects image formats and directory structures
- No hardcoded dataset assumptions

### ✓ Online RAG Fallback
- Automatically queries PubMed or Wikipedia when local corpus is insufficient
- Configurable similarity threshold for triggering fallback
- Tracks and reports fallback usage

### ✓ Enhanced Explainability
- Confidence levels for every caption (very_high, high, moderate, low, very_low)
- Explicit uncertainty statements when features cannot be recognized
- Source attribution (local corpus vs. online retrieval)
- Clear error messaging

### ✓ Modular & Adaptable
- Easy to swap vision encoders, LLMs, and retrieval sources
- Configurable batch sizes, thresholds, and parameters
- Supports custom text corpora

## Quick Start

### Installation

```bash
pip install open-clip-torch==2.23.0 faiss-cpu transformers sentence-transformers tqdm matplotlib scikit-learn biopython requests
```

### Configuration

Update the `CONFIG` dictionary in Cell 2 of the notebook:

```python
CONFIG = {
    "DATASET_DIR": "/path/to/your/medical/images/",
    "CORPUS_PATH": None,  # or path to custom corpus
    "ENABLE_ONLINE_RAG": True,
    "ONLINE_RAG_SOURCE": "pubmed",  # or "wikipedia"
    "SIMILARITY_THRESHOLD": 0.3,
    "PUBMED_EMAIL": "your.email@example.com",
    # ... other settings
}
```

### Usage

1. Open `VLM_Deploy2.ipynb` in Jupyter/Kaggle
2. Replace cells with corresponding cells from `UPDATED_CELL_BLOCKS.md`
3. Run cells sequentially
4. Review outputs in the configured `OUT_DIR`

## Updated Cell Blocks

See `UPDATED_CELL_BLOCKS.md` for the complete, enhanced cell implementations that address:

1. **Cell 7 - Text Embedding Step** (CRITICAL - was missing)
   - Embeds text corpus using BiomedCLIP
   - Required for FAISS retrieval to work

2. **Cell 8 - Online RAG Fallback** (NEW FEATURE)
   - PubMed/Wikipedia integration
   - Automatic fallback on low similarity

3. **Cell 3 - Data-Agnostic Dataset Loader** (ENHANCED)
   - Generic image collection
   - Format validation

4. **Cell 10 - Enhanced LLM Synthesis** (ENHANCED)
   - Confidence-aware caption generation
   - Uncertainty handling
   - Source attribution

5. **Other Cells** (UPDATED)
   - Improved error handling
   - Better logging and progress reporting
   - Enhanced visualizations with confidence indicators

## Pipeline Flow

```
Input Images
    ↓
[Vision Encoder: BiomedCLIP]
    ↓
Image Embeddings
    ↓
[FAISS Retrieval: Local Corpus]
    ↓
Similarity < Threshold? → [Online RAG: PubMed/Wikipedia]
    ↓
Retrieved Texts (Local + Online)
    ↓
[LLM Synthesis: Flan-T5]
    ↓
Grounded Captions with Confidence
    ↓
Output Files (JSONL, CSV, Visualizations)
```

## Output Files

After running the pipeline, you'll find:

- `image_embeddings.npy` - Cached image embeddings
- `text_embeddings.npy` - Cached text corpus embeddings
- `retrieval_results.jsonl` - Raw retrieval results with scores
- `synthesized_captions.jsonl` - Generated captions with confidence
- `synthesized_captions_rescored.csv` - Captions with similarity scores
- `run_metadata.json` - Complete run statistics
- `sample_gallery_with_confidence.png` - Visual results with confidence

## Configuration Options

### Dataset Settings
- `DATASET_DIR`: Path to your image dataset
- `SUPPORTED_IMAGE_FORMATS`: Image extensions to process
- `RECURSIVE_SEARCH`: Search subdirectories

### Model Settings
- `BIOMEDCLIP_HF_ID`: Vision-language model ID
- `LLM_ID`: Text generation model ID
- `DEVICE`: "cuda" or "cpu"

### RAG Settings
- `ENABLE_ONLINE_RAG`: Enable/disable online fallback
- `ONLINE_RAG_SOURCE`: "pubmed" or "wikipedia"
- `SIMILARITY_THRESHOLD`: Minimum score to skip fallback (0-1)
- `MAX_ONLINE_RESULTS`: Number of online articles to retrieve
- `CORPUS_PATH`: Path to custom text corpus (optional)

### Performance Settings
- `BATCH_IMAGE`: Image embedding batch size
- `BATCH_TEXT`: Text embedding batch size
- `TOP_K`: Number of retrieval results per image

## Explainability Features

The pipeline provides multiple levels of transparency:

### Confidence Levels
- **very_high** (score > 0.7): Strong corpus match
- **high** (score > 0.5): Good corpus match
- **moderate** (score > 0.3): Acceptable match
- **low** (score > 0.15): Weak match, online fallback used
- **very_low** (score < 0.15): Very weak match, uncertain results
- **no_retrieval**: No relevant information found
- **error**: Processing error occurred

### Uncertainty Statements
When the model cannot confidently describe features, it explicitly states:
- "Unable to generate caption: No relevant reference material found"
- "[Low confidence retrieval] ..."
- "[Note: Limited information available from corpus. Consider manual review.]"

### Source Attribution
Every caption tracks:
- Which texts came from local corpus (with similarity scores)
- Which texts came from online sources (with source name)
- Whether online fallback was used

## Adapting for Your Use Case

### Custom Vision Encoder
```python
CONFIG["BIOMEDCLIP_HF_ID"] = "hf-hub:your/custom-vision-model"
```

### Custom LLM
```python
CONFIG["LLM_ID"] = "your/custom-llm-model"
```

### Custom Text Corpus
```python
CONFIG["CORPUS_PATH"] = "/path/to/corpus.txt"
# Or directory of .txt files
CONFIG["CORPUS_PATH"] = "/path/to/corpus_directory/"
```

### Adjust Fallback Sensitivity
```python
# Higher threshold = more fallback usage
CONFIG["SIMILARITY_THRESHOLD"] = 0.5

# Lower threshold = less fallback usage
CONFIG["SIMILARITY_THRESHOLD"] = 0.2
```

## Production Deployment

For production use, consider:

1. **Vector Database**: Replace FAISS with Pinecone/Weaviate for scale
2. **Stronger LLM**: Use larger models (GPT-4, Claude) for better captions
3. **Caching**: Enable aggressive caching of embeddings
4. **Batch Processing**: Process large datasets in batches
5. **Monitoring**: Log confidence distributions and fallback rates
6. **Validation**: Have domain experts review low-confidence captions

## Troubleshooting

### Issue: Text embeddings not created
**Solution**: Ensure Cell 7 (Text Embedding Step) is included and run

### Issue: Online RAG not working
**Solution**: Check:
- `ENABLE_ONLINE_RAG = True` in config
- `PUBMED_EMAIL` is set correctly
- Internet access is available
- Biopython is installed

### Issue: Low caption quality
**Solution**: Try:
- Increase `TOP_K` for more context
- Lower `SIMILARITY_THRESHOLD` to reduce fallback
- Use larger LLM model
- Provide custom corpus specific to your domain

### Issue: Slow processing
**Solution**:
- Reduce `BATCH_IMAGE` and `BATCH_TEXT` if OOM
- Use GPU with `DEVICE = "cuda"`
- Enable caching (embeddings saved automatically)
- Process smaller subset first with `use_sample = True`

## Citation

If you use this pipeline in your research, please cite:
```
@software{vlm_pipeline_enhanced,
  title={Enhanced VLM Pipeline for Medical Image Captioning},
  author={Your Name},
  year={2024},
  url={https://github.com/Jade2451/VLM_pipeline}
}
```

## License

[Specify your license]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Check `UPDATED_CELL_BLOCKS.md` for detailed documentation
- Review output `run_metadata.json` for diagnostics

---

**Note**: This is an enhanced version of the original VLM pipeline with critical missing features implemented (text embedding, online RAG fallback, explainability, data-agnostic design).