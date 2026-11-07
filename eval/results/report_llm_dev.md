# تقرير تقييم مُرشِد — llm-dev (sample docs)

| المقياس | القيمة |
|---|---|
| run | llm-dev (sample docs) |
| n_questions | 12 |
| hit@5 | 1.0 |
| precision@5 | 0.333 |
| mrr | 0.889 |
| refusal_recall (should refuse -> refused) | 1.0 |
| false_refusal_rate | 0.0 |
| citation_rate | 1.0 |
| faithfulness | 0.972 |
| correctness | 0.944 |
| passed | 12/12 |
| llm | answer=gemini-3.5-flash-lite, judge=gemini-3.1-flash-lite |
| answers_by_mode | {'llm': 11, 'rule': 1} |
| all_llm | True |
| avg_in_tokens_per_llm_answer | 1475.636 |
| avg_out_tokens_per_llm_answer | 149.273 |
| avg_latency_ms | 289.25 |
| judge_scores_llm/proxy | 18/0 |

## النتائج حسب نوع السؤال

| النوع | العدد | نجح |
|---|---|---|
| dialect | 2 | 2 |
| direct | 3 | 3 |
| follow_up | 1 | 1 |
| multi_doc | 2 | 2 |
| not_in_docs | 2 | 2 |
| out_of_scope | 1 | 1 |
| table | 1 | 1 |

## الأسئلة التي فشلت (حلّلوا السبب وحسّنوا)

| # | السؤال | المشكلة | أعلى مقطع | dense | coverage |
|---|---|---|---|---|---|

## معايرة عتبة الرفض

قارنوا قيم dense/coverage لأعلى مقطع في الأسئلة التي يجب رفضها مقابل الأسئلة القابلة للإجابة، واضبطوا MIN_DENSE_SCORE و MIN_TERM_COVERAGE في ملف .env بحيث تفصل بينهما.

| # | يجب الرفض؟ | dense | coverage |
|---|---|---|---|
| d2 | لا | 0.888 | 0.4 |
| d3 | لا | 0.918 | 0.75 |
| d4 | لا | 0.913 | 0.9 |
| f1 | لا | 0.913 | 0.89 |
| g1 | لا | 0.891 | 0.36 |
| g2 | لا | 0.898 | 0.78 |
| t1 | لا | 0.89 | 0.71 |
| m1 | لا | 0.922 | 0.67 |
| m2 | لا | 0.883 | 0.58 |
| n1 | نعم | 0.865 | 0.33 |
| n2 | نعم | 0.918 | 0.86 |
| o1 | نعم | 0 | 0 |