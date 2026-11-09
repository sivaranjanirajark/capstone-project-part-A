# Instamart Customer Intelligence Copilot

## Overview

This repository contains an end-to-end data and AI pipeline for a quick-commerce company (Instamart). It starts from deliberately dirty raw exports and ends with a support-ticket triage pipeline, a policy question-answering assistant and a refund agent.

| Part | What it does | Main scripts |
| --- | --- | --- |
| A | Audit, clean and store the data, then answer business questions in SQL and pandas | `audit_data.py`, `clean_data.py`, `validate_data.py`, `run_queries.py`, `pandas_analysis.py` |
| B | RFM customer segments with K-means and PCA, plus a lapse-prediction model | `segments.py`, `build_lapse_dataset.py`, `train_lapse_model.py` |
| C | Support-ticket triage with input guardrails, PII redaction, retries, output validation and a token budget | `triage.py` |
| D | A retrieval-augmented policy assistant and a resumable refund agent | `policy_rag.py`, `refund_agent.py` |

All data is synthetic and is produced by `generate_data.py`. The language model is the supplied offline stub `mock_llm.py` (unmodified), so no API key is needed. All money is in INR.

## Reproduce

Run these commands from the repository root, in order (Windows PowerShell shown; on macOS/Linux activate with `source .venv/bin/activate`).

1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python audit_data.py`
6. `python clean_data.py`
7. `python validate_data.py`
8. `python run_queries.py`
9. `python pandas_analysis.py`
10. `python segments.py`
11. `python build_lapse_dataset.py`
12. `python train_lapse_model.py`
13. `python triage.py`
14. `python policy_rag.py`
15. `python refund_agent.py`

Step 7 must print `ALL CHECKPOINTS PASSED`. Steps 6 to 15 create `data/clean/`, `data/instamart.db`, `outputs/` and `state/`.

## Part A - Data Foundation

### Audit of the raw data

| Table | Shape (rows x columns) | Missing values |
| --- | --- | --- |
| orders.csv | 13,084 x 9 | order_amount 53 (0.41%), delivery_minutes 2,541 (19.42%) |
| customers.csv | 1,200 x 5 | age 61 (5.08%), gender 30 (2.50%) |
| products.csv | 40 x 4 | none |

Specific audit figures:

| Check | Value |
| --- | --- |
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id values | 12,309 |
| order_amount entries that cannot be converted to a number | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age (customers) | 61 |
| Age outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules and how much each one changed

Customers:

| # | Rule | Changed |
| --- | --- | --- |
| 1 | Strip spaces and apply title case to city, leaving exactly six cities | 476 values |
| 2 | Map every gender spelling to male, female or unknown (blank becomes unknown) | 950 values rewritten, 30 of them set to unknown |
| 3 | Treat ages outside 10-100 as missing, then fill all missing ages with the median age of the customer's city (groupby + transform) | 6 set to missing, 67 ages filled (61 blank + 6 invalid) |
| 4 | Convert signup_date to datetime | 1,200 values |

Orders, applied in this order:

| # | Rule | Changed |
| --- | --- | --- |
| 1 | Convert order_ts to datetime | 13,084 values |
| 2 | Remove exact duplicate rows | 119 rows removed |
| 3 | Keep only the latest-order_ts record for each order_id (drops the earlier pending records) | 656 rows removed |
| 4 | Convert order_amount to numeric and fill missing amounts with round(price_inr x quantity x (1 - discount_pct / 100), 2) | 112 amounts filled |
| 5 | Set negative delivery_minutes to missing | 105 values (after duplicates were removed) |
| 6 | For completed orders, fill missing delivery_minutes with the median of the order's product category among completed orders | 774 values filled (669 blank + 105 negative) |
| 7 | For cancelled orders, leave delivery_minutes missing, because a cancelled order was never delivered and imputing a time would invent data | 1,199 left missing |

Result: 12,309 rows, one per order_id; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing order_amount; 0 missing delivery_minutes among completed orders; completed revenue INR 2,998,668.80.

Cleaned customers: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; gender female 610, male 560, unknown 30; 0 missing ages and every age between 18 and 60.

### Storage and validation

The clean tables are saved to `data/clean/` as CSV and to `data/instamart.db` (SQLite tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `validate_data.py` asserts the reference checkpoints above and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`)

| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING (completed orders above INR 500, customer total above INR 8,000) | 30 customers |
| Q2 three-table join, revenue by city and category | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics at INR 233,909.45 |
| Q3 scalar subquery, products priced above the catalogue average | 11 products |
| Q4 LEFT JOIN, customers with no completed order | 25 customers |
| Q5 CTE, customers whose total spend is above the average | 550 customers, average INR 2,552.06 |
| Q6 ranking window functions per city | top customers: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK() different from ROW_NUMBER() |
| Q7 running total of daily revenue | 183 daily rows, last cumulative value INR 2,998,668.80 |

Why RANK() and ROW_NUMBER() differ in 8 rows: when two customers in the same city have exactly the same total spend, RANK() gives them the same rank and skips the next number, while ROW_NUMBER() always gives each row a unique number.

### pandas cross-check and exploratory analysis

The pivot table (`pd.pivot_table(..., margins=True)`) has a grand total of INR 2,998,668.80, equal to the SQL total from Q2.

| City | Sum (INR) | Mean (INR) | Count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount statistics: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, upper fence 645.30 with 897 outliers above it, lower fence -283.50 with none below.

Correlations over completed orders: order_amount and discount_pct -0.095 (largest in absolute value), order_amount and quantity +0.087, all other pairs below 0.02 in absolute value.

**What is this data about?** About six months (183 days) of orders from 1,200 customers in six Indian cities, across the 40-product catalogue. **Distributions:** order values are strongly right-skewed (mean 269.91 versus median 135.00, skewness 4.61). **Outliers:** 897 orders lie above the upper fence of INR 645.30; these are large baskets rather than errors, and none lie below the lower fence. **Relationships:** the strongest correlation is order_amount with discount_pct at about -0.095, and order_amount with quantity is +0.087. Correlations this small are weak, and correlation of this size does not establish causation.

## Part B - Customer Segments and Lapse Prediction

### RFM table

Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers. Recency: median 19, minimum 0, maximum 180 days. Frequency: mean 9.46, maximum 33. Monetary: mean INR 2,552.06, maximum INR 14,766.55.

### K-means (StandardScaler, n_init=10, random_state=42)

| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

I chose **K = 4**. Inertia drops sharply up to K = 4 and flattens afterwards (the elbow), K = 4 has the highest silhouette score, and four segments is a number the retention team can run campaigns for. The inertia of K = 8 is always lower than that of K = 2 because more centres always fit the points at least as closely, so a lower inertia alone would always push us to pick a larger K; the elbow, the silhouette score and business usefulness are needed to choose.

### Segment profiles (K = 4)

Cluster numbers are only labels and carry no ordering.

| Segment | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Suggested name and action |
| --- | --- | --- | --- | --- | --- |
| A | 436 | 28.9 | 3.8 | 1,057.62 | Occasional shoppers: send small-basket offers and category suggestions |
| B | 138 | 24.1 | 6.8 | 7,129.70 | **Premium** big spenders (highest mean monetary value): priority service and early access |
| C | 419 | 8.8 | 19.4 | 3,410.02 | Loyal regulars: loyalty rewards and subscription offers |
| D | 182 | 109.0 | 2.2 | 685.99 | Dormant customers: win-back discount |

The names in `outputs/customer_segments.csv` (`segment_name` column) are the ones used by the scripts; keep this table consistent with that file.

PCA on the scaled RFM matrix: explained variance ratios 0.634 and 0.218, total 0.852.

### Lapse problem

| Item | Value |
| --- | --- |
| Cutoff | 2025-09-01 |
| Cohort (at least one completed order before the cutoff) | 1,156 customers |
| Lapsed (no completed order 2025-09-01 to 2025-09-30) | 443 (38.32%) |
| Train / test | 924 / 232 (`test_size=0.2`, stratified, `random_state=42`) |
| Baseline accuracy (always predict not lapsed) | 61.6% (lapsed-class F1 = 0) |

Features: recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category, city, age, gender and tenure in days.

**Leakage statements.** No encoder, scaler or resampler saw test rows: they are fitted inside a scikit-learn `Pipeline` on training rows only, so cross-validation refits them on each fold. No order dated on or after 2025-09-01 entered the features. The segment labels from the segmentation step were not used as features. Proof: the scaler's mean for frequency (8.1147, learned from the 924 training rows) differs from the full-cohort mean (8.0285), because the full-cohort mean includes the 232 test rows.

### Models

| Model | Best parameters | Best CV F1 |
| --- | --- | --- |
| Logistic regression | C = 100 | 0.6506 |
| Random forest (100 trees) | max_depth = 8, min_samples_leaf = 10 | 0.6811 |

Test-set results (lapsed class is positive):

| Model | Confusion matrix [[TN FP] [FN TP]] | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic regression | [[111 32] [31 58]] | 0.6444 | 0.6517 | 0.6480 |
| Random forest | [[113 30] [31 58]] | 0.6591 | 0.6517 | 0.6554 |

**Model choice.** I selected the random forest. It has the higher F1 and precision at the same recall. Missing a customer who is about to lapse costs lost revenue, while wrongly targeting a customer who would have stayed costs only a small discount, so recall matters more than precision here. Both models catch about 65% of lapsing customers, and the random forest wastes fewer offers.

### Class-balance experiment

The lapsed rows in the training set were randomly duplicated (with replacement, `random_state=42`) until both classes had 570 rows. Test data was not touched.

| Logistic regression | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Original training set | 0.6444 | 0.6517 | 0.6480 |
| Balanced training set | 0.5891 | 0.8539 | 0.6972 |

Recall moved most (0.65 to 0.85). Balancing makes the model flag more customers as lapsing, so the retention team catches more real lapsers but also contacts more customers who would have stayed. This is a good trade when a missed lapser costs more than a wasted offer.

### Interpretation, persistence and retention list

Top five random-forest features: frequency (0.2796), monetary (0.2712), recency (0.1151), avg_delivery (0.0613), cancel_rate (0.0523).

Logistic-regression coefficients on standardised features: frequency -1.6982 (customers who order often are much less likely to lapse), monetary +0.2846, recency +0.0396 (almost no effect once frequency is in the model). A positive sign raises the lapse risk. Monetary and frequency are strongly related, so once frequency is in the model the monetary sign should not be read as "higher spenders lapse more".

The chosen model is saved with `joblib` to `outputs/lapse_model.joblib`; reloading it reproduces the test predictions exactly (checked in the script).

Retention list (`outputs/retention_list.csv`): 40 Premium customers are in the test set; the 10 with the highest lapse probability are listed, and 1 of those 10 actually lapsed. **The Premium flag comes from whole-window RFM, which includes September, the same window that defines the lapse label, so this list is illustrative and not a leakage-free evaluation.**

## Part C - Support Ticket Triage

Pipeline per ticket: skip or flag unusable input, redact personal data, call the model with retries, validate the reply, apply the token budget, then route.

### Guardrails and redaction

164 tickets: 4 empty (skipped) and 4 too long (estimated 734 tokens, limit 400, flagged); neither kind is sent to the model.

Redaction runs in this order with three regular expressions (email, 10-digit Indian mobile number starting 6-9, `ORD-` followed by six digits):

| Pattern | Substitutions |
| --- | --- |
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| Total | 278 |

After redaction 0 matches remain, and the script asserts that no prompt contains personal data.

### Retries

Calls are wrapped with up to 3 retries and exponential backoff with jitter. A full run had 19 transient errors; each recovered on its first retry, for example `[retry] ticket=T0002 retry=0 wait=0.0530s`, and none failed permanently.

### Output validation

| Status | Count |
| --- | --- |
| classified_json (valid JSON with an allowed category) | 135 |
| classified_keyword (not JSON, but exactly one allowed category word) | 12 |
| flagged_invalid_output (neither) | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

Outputs checked by rules: the 147 classified tickets (JSON schema or single-keyword check). Outputs sent to humans: the 9 invalid outputs plus the 4 empty and 4 long tickets, 17 in total (`human_review`).

### Zero-shot versus few-shot

| Prompt | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
| --- | --- | --- | --- | --- |
| Zero-shot | 147 | 85.7% (126 correct) | 8,776 | 59.70 |
| Few-shot | 147 | 100.0% (147 correct) | 20,164 | 137.17 |

Zero-shot confusion table (rows are true categories, columns are predicted):

| True \ Predicted | billing | delivery | fraud | technical |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 2.3 times as many tokens per ticket (137.17 versus 59.70) but removes all 21 mistakes, which were fraud and technical tickets sent to billing. A wrongly routed fraud ticket is expensive, so the extra tokens are worth paying for on categories that are costly to get wrong; for high-volume, low-risk categories the cheaper zero-shot prompt may be enough.

### Token budget and routing

A zero-shot budget demonstration with a budget of 6,000 tokens used 6,030 tokens and marked 49 tickets `budget_exhausted` (`outputs/triage_budget_demo.csv`). The final run uses the few-shot prompt with a budget of 100,000 tokens (`outputs/triage_results.csv`, 164 rows).

| Route | Tickets | Priority |
| --- | --- | --- |
| support_queue | 102 | NORMAL |
| security_team (fraud) | 34 | HIGH |
| human_review | 17 | NORMAL |
| premium_desk (Premium customers, from `customer_segments.csv`) | 11 | HIGH |

## Part D - Policy Assistant and Refund Agent

### Policy assistant (retrieval-augmented generation)

Policy documents are split into 40-word chunks with 10 words of overlap, giving 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). Chunks are vectorised with TF-IDF; a question is vectorised with the same fitted vectorizer and the top 3 chunks are found by cosine similarity. The question and the retrieved chunks, each labelled with its source file, go to the model with an instruction to answer only from the context. For the query "refund policy late delivery" all three top chunks come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk; IDF (inverse document frequency) lowers the weight of words that appear in many chunks, so rare, informative words count more. A limitation is that TF-IDF only matches words: two phrasings with the same meaning but different words get a low similarity.

| Setting | Chunks | Expected document in top 3 | Expected fact in answer | Unanswerable gets fallback |
| --- | --- | --- | --- | --- |
| Overlap 10 | 15 | 12/12 | 12/12 | 2/2 |
| Zero overlap | 13 | 12/12 | 10/12 | 2/2 |

With zero overlap two questions fail, for example "What delivery time makes a completed order eligible for a full refund?" The relevant sentence is cut at a chunk boundary, so the right document is retrieved but the chunk no longer contains the key phrase. Overlap repeats a few words across boundaries and prevents this.

### Refund agent

The agent has five tools: `get_order` (reads `data/instamart.db`), `search_policy` (reuses the retrieval code above), `calculate_refund` (100% refund when delivery took more than 45 minutes, otherwise 0), `issue_refund` and `escalate_to_human` (both append to `outputs/refund_ledger.jsonl`). It runs a ReAct loop: the model chooses a tool, the tool runs, the observation is added to the messages, and this repeats until the model gives a final answer. A refund above INR 1,000 is escalated to a human.

Seven scenarios (`outputs/agent_runs.jsonl`):

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 1,042 |
| S2 | REFUND | 135.00 | 5 | 1,042 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,029 |
| S4 | DENIED | 0.00 | 4 | 721 |
| S5 | DENIED | 0.00 | 3 | 424 |
| S6 | NOT_FOUND | 0.00 | 2 | 130 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,174 |

Token counts can differ slightly between runs; statuses, refunds and call counts are fixed.

### Safe stopping

| Demonstration | Setting | Status |
| --- | --- | --- |
| Token limit | S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT (441 tokens used, the requested tool was not executed) |
| Time limit | S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT (0 model calls) |

S7 stops with FALLBACK_MAX_ITERATIONS after 6 model calls. Every fallback returns a handoff message that lists the tools completed so far.

### Resumable runs

After every tool observation the agent writes a checkpoint to `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number and tools executed). In the demonstration the run for S1 was crashed on purpose after the second tool execution and then resumed from the checkpoint. It finished with status REFUND (INR 128.25) and the tools executed across the crash and the resume were, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. No tool ran twice, so the refund was issued once.

### ReAct compared with plan-and-execute

ReAct decides the next step after seeing each observation, so one agent handles every case without a fixed script. Scenario S6 (order not found) stopped after 2 model calls and S5 (denied) after 3, because the loop could react to the first observation, while S1 needed 5. A plan-and-execute agent would write the whole plan first, for example look up, search policy, calculate, issue, and would run all steps even for S6, wasting calls on an order that does not exist unless it re-plans. Plan-and-execute uses fewer model calls on predictable tasks (one planning call, then tool calls without the model) and is easier to audit, because the plan can be reviewed before any money moves. Its weakness is rigidity: when an observation changes the situation, the plan is wrong until it is replanned. ReAct is more flexible but costs a model call for every step, its prompt grows with every observation, and it can loop, which is why scenario S7 needs the iteration, token and time limits. For this refund task, where the correct path depends on what the lookup returns, ReAct fits better, with hard limits and checkpoints as safeguards.

## Limitations and Next Steps

- **Retention list is not a clean evaluation.** The Premium flag uses whole-window RFM that includes September, the same period that defines the lapse label, so the retention list is illustrative only. A leakage-free version would compute segments as of the cutoff date.
- **Synthetic data and a stub model.** All data is generated and the language model is a deterministic stub, so accuracy figures (for example 100% few-shot) describe this setup and will not carry over to real tickets or a real model.
- **Small lapse dataset.** The test set has 232 customers, so F1 differences of 0.01 are within noise. Repeated cross-validation would give a steadier comparison.
- **Over-sampling by duplication.** Random duplication can make a model memorise repeated rows; techniques such as class weights or SMOTE could be compared.
- **TF-IDF retrieval** matches words, not meaning. Sentence embeddings would handle paraphrases better.
- **Token counts** use the stub's estimate, not a real tokenizer.
- **Next steps:** add automated tests for the cleaning rules and the agent tools, schedule the pipeline, monitor model drift, and add a human-approval step for refunds near the INR 1,000 limit.