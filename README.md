# polyreader

An LLM agent that extracts structured, provenance-aware polymer data (identity, synthesis, molecular weight, processing, properties) from scientific papers and their supplementary information. PDFs are parsed into a navigable index (no vector store); an LLM reads the paper and writes one validated JSON record per sample.

## Install

```bash
conda create -n polyreader python=3.11 && conda activate polyreader
pip install -e .
```

Set an API key (or put it in a `.env` file): `DEEPSEEK_API_KEY` for the [DeepSeek API](https://api-docs.deepseek.com) or `OPENROUTER_API_KEY` for [OpenRouter](https://openrouter.ai).

## Usage

```bash
polyreader paper.pdf                                    # index + extract
polyreader paper.pdf si.pdf --provider deepseek         # main paper + SI
polyreader paper.pdf --extraction-mode auto             # whole paper in context when it fits (faster, cheaper)
polyreader paper.pdf --pdf-parser mineru --mineru-executable /path/to/mineru-kit
polyreader paper.pdf --skip-index                      # reuse an existing index (also --index-only, --extract-only)
```

Results go to `output/<provider>/<paper>/`: one JSON per sample in `samples/` and a `<paper>_pipeline_report.json`. Run `polyreader --help` for all options (model, thinking/reasoning effort, vision, concurrency).

```python
import asyncio
from polyreader import process_document, save_index, load_index, extract, ExtractionConfig

DEEPSEEK = "https://api.deepseek.com"  # omit base_url/model to use OpenRouter

index = asyncio.run(process_document("paper.pdf", base_url=DEEPSEEK, model="deepseek-flash"))
save_index(index, "output/paper.json")

config = ExtractionConfig(base_url=DEEPSEEK, model="deepseek-flash", extraction_mode="auto")
result = asyncio.run(extract(load_index("output/paper_tree.json"), config))
for sample in result.samples:
    print(sample.sample_id, sample.quality_tier, sample.molecular_weight.mn)
```

## Reference

Y. Shi, X. Zhong, Y. Wei, H. Choi, T. Zhou, Z. Wu. *PolyReader: an agentic LLM pipeline for extracting structured data from unstructured polymer scientific literature.* ChemRxiv (2026). https://doi.org/10.26434/chemrxiv.15008614/v1

```bibtex
@article{shi2026polyreader,
  title   = {PolyReader: an agentic LLM pipeline for extracting structured data from unstructured polymer scientific literature},
  author  = {Shi, Yuwei and Zhong, Xinyan and Wei, Yuqi and Choi, Heechae and Zhou, Tianhang and Wu, Zhenghao},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15008614/v1},
  note    = {Preprint}
}
```
