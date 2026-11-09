# SpendSmart backend model walkthrough

This note is for the classroom demonstration. The model demo is reproducible and uses synthetic examples only. Live ML endpoints use the transactions saved in the browser and submitted to the local Python server.

## 60-second presentation

1. Open **Smart insights** and point out the live anomaly review. It is fit from the current browser’s expense records.
2. Select **Run model demo**. This calls `GET /api/ml/demo`; the synthetic demo does not send or use personal finance data.
3. Explain that the category model learns from 32 labeled merchant descriptions and evaluates on 8 held-out descriptions. Its current deterministic result is **8 correct of 8 (100%)**. The example `coffee cafe` is classified as **Food**.
4. Explain that the anomaly model builds 48 randomized isolation trees from 49 synthetic expense examples. One ₹12,500 expense is deliberately injected; the page shows whether it appears among the top three review results.
5. Finish with the limitations: the test set is tiny and hand-curated, so the 100% score only proves that this controlled example runs end to end. It is not evidence that the model is ready for real banking or fraud decisions. The anomaly check has one known outlier and is not a full precision/recall evaluation.

## Category classifier

- **Algorithm:** Multinomial Naive Bayes, implemented in Python in `server.py` without an ML package.
- **Input features:** Lowercased word and number tokens from the merchant/description.
- **Labels:** User-selected expense categories such as Food, Transport, Shopping, and Education.
- **Training:** Count documents and tokens per category, estimate category priors and smoothed token likelihoods, then score each candidate category for a new description.
- **Live endpoint:** `POST /api/ml/category` accepts the browser’s labeled transactions and a new description. It requires at least 8 labeled expenses and at least 2 categories with 2 examples each. The UI shows the score and asks the user before applying the suggestion.
- **Evaluation:** The classroom demo reserves 2 examples in each of 4 categories (8 total) from fitting, predicts those descriptions, and reports exact-match accuracy plus correct/total counts by category.

A small test set makes the score sensitive to individual examples. A 67% Naive Bayes score is a relative model score, not a calibrated probability or guarantee.

## Anomaly review

- **Algorithm:** A compact Isolation Forest implementation with 48 randomized trees.
- **Input features:** `log(1 + amount)` and `log(1 + amount / category median)`.
- **Scoring:** Short average tree path lengths indicate points that are easier to isolate. The endpoint returns a ranking score and a small set of high-scoring expenses for review.
- **Live endpoint:** `POST /api/ml/anomalies` fits on the user’s expense history; it waits for at least 12 expenses.
- **Classroom check:** The demo supplies 48 ordinary synthetic transactions and one deliberately large injected expense. It reports whether that specific record appears in the top three.

An unusual amount is not evidence of fraud. There is only one injected anomaly in the demo, so its detection is a controlled sanity check, not a general accuracy metric.

## API map

| Route | Purpose |
| --- | --- |
| `GET /api/ml/demo` | Reproducible synthetic training/evaluation for presentation |
| `POST /api/ml/category` | Fit a category model on browser labels and classify a new description |
| `POST /api/ml/anomalies` | Fit an anomaly model to browser expense amounts |
| `POST /api/doctor` | Send consented transaction context through the local server to OpenAI; separate from the ML models |

The server binds to `127.0.0.1`. ML is recalculated on request from the supplied data; this prototype does not persist model weights or train on external financial datasets.
