# SpendSmart AI classroom prototype

A responsive personal-finance design thinking prototype. The sample transactions are synthetic; records you add are saved in this browser using local storage.

## Run it

From this folder, start a local web server:

```powershell
py -m http.server 8000
```

Then open <http://localhost:8000>. Stop the server with `Ctrl+C`.

No package installation, account, API key, or external service is required. Internet is only used for the optional Google Fonts; the app itself runs locally.

## What works

- Dashboard totals and spending charts derived from saved records
- Add, edit, delete, search, and filter transactions
- CSV export
- Total and category budget editing
- Savings goals and contributions
- Rule-based spending observations and a deterministic Money Doctor that answers category, income, budget, affordability, daily pace, and month-comparison questions
- Money Multiverse purchase comparison and a transparent Future Me estimate
- Local data reset and deletion

The Money Doctor uses simple rules and exact local calculations. It is a static, rule-based assistant rather than a generative AI model; it does not send financial data to an AI provider. The projection is a straight-line estimate, not a trained model or financial advice. There is no login or multi-user database in this classroom prototype.
