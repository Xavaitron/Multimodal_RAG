# Experiment results

All rows below use the complete query and document sets for their dataset.

| dataset | method | model | ndcg@5 | mrr | recall@5 |
| --- | --- | --- | --- | --- | --- |
| mteb/VidoreArxivQARetrieval | global_visual | vidore/colSmol-256M | 0.1829 | 0.1869 | 0.2600 |
| mteb/VidoreArxivQARetrieval | maxsim | vidore/colSmol-256M | 0.7244 | 0.7232 | 0.7740 |
| mteb/VidoreArxivQARetrieval | global_visual | vidore/colSmol-500M | 0.2527 | 0.2538 | 0.3340 |
| mteb/VidoreArxivQARetrieval | maxsim | vidore/colSmol-500M | 0.7466 | 0.7359 | 0.8120 |
| mteb/VidoreDocVQARetrieval | global_visual | vidore/colSmol-256M | 0.1658 | 0.1681 | 0.2336 |
| mteb/VidoreDocVQARetrieval | maxsim | vidore/colSmol-256M | 0.5557 | 0.5453 | 0.6398 |
| mteb/VidoreDocVQARetrieval | global_visual | vidore/colSmol-500M | 0.1914 | 0.1953 | 0.2583 |
| mteb/VidoreDocVQARetrieval | maxsim | vidore/colSmol-500M | 0.5776 | 0.5762 | 0.6427 |
| mteb/VidoreInfoVQARetrieval | global_visual | vidore/colSmol-256M | 0.4535 | 0.4449 | 0.5638 |
| mteb/VidoreInfoVQARetrieval | maxsim | vidore/colSmol-256M | 0.8363 | 0.8296 | 0.8806 |
| mteb/VidoreInfoVQARetrieval | global_visual | vidore/colSmol-500M | 0.5545 | 0.5383 | 0.6670 |
| mteb/VidoreInfoVQARetrieval | maxsim | vidore/colSmol-500M | 0.8686 | 0.8622 | 0.9049 |
| mteb/VidoreTatdqaRetrieval | global_visual | vidore/colSmol-256M | 0.2126 | 0.2184 | 0.2947 |
| mteb/VidoreTatdqaRetrieval | maxsim | vidore/colSmol-256M | 0.7565 | 0.7358 | 0.8570 |
| mteb/VidoreTatdqaRetrieval | global_visual | vidore/colSmol-500M | 0.2838 | 0.2795 | 0.3919 |
| mteb/VidoreTatdqaRetrieval | maxsim | vidore/colSmol-500M | 0.7648 | 0.7394 | 0.8737 |
| mteb/VidoreArxivQARetrieval | bm25 | Okapi BM25 | 0.1851 | 0.1917 | 0.2200 |
| mteb/VidoreArxivQARetrieval | dense_text | BAAI/bge-small-en-v1.5 | 0.3226 | 0.3198 | 0.3680 |
| mteb/VidoreDocVQARetrieval | bm25 | Okapi BM25 | 0.3477 | 0.3516 | 0.3860 |
| mteb/VidoreDocVQARetrieval | dense_text | BAAI/bge-small-en-v1.5 | 0.2712 | 0.2784 | 0.3137 |
| mteb/VidoreInfoVQARetrieval | bm25 | Okapi BM25 | 0.6092 | 0.5960 | 0.6822 |
| mteb/VidoreInfoVQARetrieval | dense_text | BAAI/bge-small-en-v1.5 | 0.6849 | 0.6741 | 0.7615 |
| mteb/VidoreTatdqaRetrieval | bm25 | Okapi BM25 | 0.5744 | 0.5514 | 0.7035 |
| mteb/VidoreTatdqaRetrieval | dense_text | BAAI/bge-small-en-v1.5 | 0.3136 | 0.3143 | 0.3919 |
