# Baseline Research Plan Before New MAS Design

Generated on 2026-06-16.

## Direction Change

Do not treat the existing MAS v2 design as the next implementation target. The next research step should be:

1. Freeze and migrate the dataset.
2. Reproduce simple, external, empirical-study-style baselines on this dataset.
3. Identify concrete failure modes of those baselines.
4. Design a new MAS only as a response to those observed failures.

This closes the research logic: baseline first, failure analysis second, MAS contribution third.

## Dataset Tasks

The dataset naturally creates two tasks:

| Task | Input | Gold Target | Main Metrics |
| --- | --- | --- | --- |
| Stage 2 filtering | `Dataset/stage1.csv` | membership in `Dataset/stage2.csv` | precision, recall, F1, false positive categories, false negative categories |
| Stage 3 labeling | `Dataset/stage2.csv` | labels in `Dataset/stage3.csv` | exact match, per-label macro/micro F1, hierarchical or semantic agreement if taxonomy is normalized |

The first task asks whether a candidate record should be accepted as bug-relevant. The second asks how to assign empirical-study labels such as `symptom`, `root_cause`, `bug_type`, `component`, and `fix_type`.

## Recommended Baselines

### B0: Majority And Heuristic Baselines

Purpose: establish a low-cost lower bound.

Implementation:

- Stage 2: accept all candidates, reject all candidates, and simple keyword/label heuristics.
- Stage 3: predict the most frequent label per field from the training split.

Why it matters: if an LLM or MAS does not beat these baselines, the task setup or prompt is broken.

### B1: Classical Text Classification Baseline

Purpose: match older empirical software-engineering bug classification practice.

Implementation:

- Input text: `title + body + comments`.
- Features: TF-IDF word and character n-grams.
- Models: logistic regression, linear SVM, and random forest where class balance allows it.
- Splits: paper-level leave-one-paper-out and grouped URL split.

Literature anchor: traditional bug-report classification studies commonly evaluate machine-learning classifiers such as Naive Bayes, SVM, decision tree, and random forest for issue or bug report classification. One relevant older anchor is "Automated classification of software issue reports using machine learning techniques: an empirical study" ([ResearchGate record](https://www.researchgate.net/publication/318657940_Automated_classification_of_software_issue_reports_using_machine_learning_techniques_an_empirical_study)).

### B2: Single-LLM Zero-Shot Baseline

Purpose: reproduce the simplest LLM automation setting.

Implementation:

- One prompt per record.
- No multi-agent debate.
- No external retrieval.
- Output strict JSON.
- Run separately for Stage 2 filtering and Stage 3 labeling.

Expected failure modes to inspect:

- hallucinated labels not in taxonomy,
- overconfident labels when `body/comments` are thin,
- confusion between symptom and root cause,
- failure to preserve empirical-study-specific label granularity.

### B3: Single-LLM Few-Shot / In-Context Baseline

Purpose: test whether examples from the same empirical workflow fix the zero-shot errors.

Implementation:

- Select few-shot examples only from training papers or training groups.
- Keep test paper unseen for leave-one-paper-out.
- Compare random examples vs taxonomy-balanced examples.

This baseline is important because if a carefully controlled few-shot prompt is enough, the MAS contribution must target efficiency, auditability, or consistency rather than raw accuracy.

### B4: Self-Consistency / Majority Vote Baseline

Purpose: isolate the value of repeated independent LLM judgments before adding roles or communication.

Implementation:

- Run the same single-agent prompt `k=3` or `k=5` times.
- Aggregate by majority vote for closed labels.
- Mark ties or invalid JSON as uncertain.

This is the minimum fair LLM baseline for any later MAS design.

### B5: Retrieval-Augmented Single-LLM Baseline

Purpose: test whether failures come from missing evidence rather than missing agent structure.

Implementation:

- Add retrieved or pre-extracted evidence fields: linked issue, PR, commit title, labels, and comments where available.
- Keep the model single-agent.
- Compare with B2/B3 under equal model and cost settings.

This baseline is especially relevant for Stage 2 filtering because bug relevance often depends on developer confirmation, linked fixes, and issue discussion.

## Papers And Sources Consulted

Sources used during this handoff:

- Local AutoEmpirical paper attachment: `Attachments/Yu чн?- 2025 - AutoEmpirical LLM-based automated research for empirical software fault analysis.pdf`.
- Public AutoEmpirical PDF search result: [Towards Automatic Software Faults Analysis / AutoEmpirical PDF](https://ttfish.cc/content/Papers/ICSE26-AutoEmpirical.pdf).
- ACM result for LLM-based fine-grained bug categorization: [Exploring Fine-Grained Bug Report Categorization with Large Language Models](https://dl.acm.org/doi/10.1145/3736408).
- Recent arXiv result on invalid bug report root-cause subclassification and LLM/RAG/web-search comparisons: [Automated Root-Cause Subclassification and No-Code Fix Generation of Invalid Bug Reports](https://arxiv.org/html/2605.17561v2).
- Survey context for LLM-based multi-agent systems in software engineering: [LLM-based multi-agent systems for software engineering](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=11489&context=sis_research).
- OpenAlex API searches were run for broad scholarly discovery, but broad queries returned noisy results. The practical baseline list above therefore prioritizes baselines that directly map to this dataset's two tasks.

## Evaluation Protocol

Use at least two split strategies:

- Leave-one-paper-out: tests cross-study transfer.
- Grouped issue URL split: prevents leakage from duplicated GitHub issues appearing in both train and test.

Report:

- macro-F1 and micro-F1 for each label field,
- per-paper performance,
- invalid-output rate,
- abstention or uncertain rate,
- cost per 100 records,
- latency per 100 records,
- failure mode counts from sampled errors.

## How This Leads To A New MAS

Only propose MAS components after baseline errors show a need:

- If B2/B3 hallucinate labels, add taxonomy validation or constrained decoding.
- If B2/B3 confuse symptom and root cause, add separate symptom/root-cause reviewers.
- If B5 improves Stage 2 heavily, add an evidence retrieval or evidence synthesis step.
- If self-consistency improves accuracy but costs too much, design a confidence gate that only escalates uncertain records.
- If paper-level generalization is poor, add paper-aware calibration or taxonomy adaptation.

The contribution should be stated as: "Existing empirical-study baselines fail on X, Y, Z when applied to this cross-paper dataset; our MAS introduces A, B, C to address those specific failures."
