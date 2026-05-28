# Semantic Working Set: Selective KV Transfer for PD-Disaggregated LLM Serving (Preliminary Draft)

## Authors

**Pengju Yang**

## Affiliation

Jilin University (Incoming graduate student, Fall 2026)

## Version Notice

This is a preliminary preprint uploaded to establish priority. Experiments are ongoing; a complete version will be released later.

This preprint is an **empirical characterization and prototype study**. It does not claim a production-quality end-to-end PD serving runtime with cold-tier fetching.

## Contact

Email: yangda1223@outlook.com

GitHub: https://github.com/YangSteve1223

## Code and Data

The experimental artifacts, raw JSON results (38 files, 37 MD5-verified), and experiment scripts are available in the project repository:

**https://github.com/YangSteve1223/kvcache-lab**

The repository includes:
- 13 experiment groups covering perplexity, latency, bandwidth, retrieval (NIAH), robustness, hyperparameter sensitivity, and cross-domain evaluation
- 4 model families: Qwen2.5-7B, Qwen2.5-14B, Mistral-7B, Gemma-2-9B
- Table-to-JSON mapping, reproducibility information, and changelog

## Files

```
source/
├── main.tex          # Main LaTeX source (compile with pdflatex)
├── main.bbl          # Pre-built bibliography (for arXiv/OSF submission)
├── the_final.bib     # BibTeX reference database
```

## Compilation

```bash
cd source/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using the provided compile script:
```bash
bash compile.sh
```

## License

- **Code**: MIT License
- **Data (JSON)**: CC BY 4.0
- **Paper (LaTeX/PDF)**: CC BY 4.0

## Citation

If you reference this preprint, please cite:

> Pengju Yang. "Semantic Working Sets for KV Transfer in Prefill--Decode Disaggregated LLM Serving." Preliminary preprint, May 2026.

## AI Tool Acknowledgment

This work used AI-assisted tools (LLM-based) for code debugging, data analysis, and manuscript drafting. All AI-generated content was reviewed and substantially revised by the author. The research ideas, experimental design, and final conclusions are the author's own.
