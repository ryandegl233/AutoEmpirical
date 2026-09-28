"""Bounded class-association mining and CART from approved source-only examples.

Training targets are source-grounded method interpretations, never gold annotations.
Recommendations nominate candidates; they neither prove nor exclude current causes.
"""
from collections import Counter
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
import re

from .explainable_tools import _issue_key, _sha

MIN_RECORDS = 4
MIN_SUPPORT = 2
MIN_CONFIDENCE = 0.6
STOP_WORDS = frozenset('the and for that this with from have has was were are but not can could would should will issue comment error when then into your you our its causes caused because'.split())


def features(view):
    return set(re.findall(r'\b[a-z][a-z0-9_]{2,39}\b', ' '.join(item.content for item in view.items).lower())) - STOP_WORDS


def mining_manifest():
    return dict(version='source-only-apriori-cart-v1', implementation_sha256=_sha(Path(__file__).read_bytes()),
                sklearn_version=version('scikit-learn'), min_records=MIN_RECORDS, min_support_count=MIN_SUPPORT,
                min_confidence=MIN_CONFIDENCE, max_antecedent_size=2, max_features=64,
                tree_max_depth=3, tree_min_samples_leaf=2, random_state=0,
                targets='source_supported_method_interpretations_not_gold')


def recommend(examples, view):
    uris = {_issue_key(item.source_uri) for item in view.items}
    training = []
    for source, query in examples:
        if source.record_id == view.record_id or any(_issue_key(i.source_uri) in uris for i in source.items):
            raise ValueError('mining library overlaps the evaluation case')
        # An unresolved candidate mention is NOT a training target.
        labels = {c.label for c in query.candidates if c.dimension == 'root_cause'
                  and c.assessment == 'supported' and c.label != 'Unknown'}
        if len(labels) == 1:
            training.append((source.record_id, features(source), next(iter(labels))))
    counts = Counter(label for _, _, label in training)
    output = dict(status='insufficient_source_supported_causes', source_records=len(examples),
        training_records=len(training), class_counts=dict(counts), candidate_range=[], association_rules=[], tree_path=[],
        semantics='candidate_hints_only_not_causal_evidence_or_a_closed_label_set',
        target_provenance='approved_source_grounded_method_interpretations_not_gold')
    if len(training) < MIN_RECORDS or len(counts) < 2 or min(counts.values()) < MIN_SUPPORT:
        return output
    freq = Counter(token for _, tokens, _ in training for token in tokens)
    vocabulary = sorted((t for t, count in freq.items() if count >= MIN_SUPPORT), key=lambda t: (-freq[t], t))[:64]
    current = features(view)
    if not vocabulary or not current.intersection(vocabulary):
        return {**output, 'status': 'no_feature_coverage'}
    # ponytail: Apriori restricted to one/two-item antecedents; larger itemsets only if coverage warrants them.
    antecedents = [(t,) for t in vocabulary]
    antecedents += [pair for pair in combinations(vocabulary, 2)
                    if sum(set(pair) <= tokens for _, tokens, _ in training) >= MIN_SUPPORT]
    rules = []
    n = len(training)
    for antecedent in antecedents:
        if not set(antecedent) <= current:
            continue
        matches = [(rid, label) for rid, tokens, label in training if set(antecedent) <= tokens]
        for label, count in sorted(Counter(label for _, label in matches).items()):
            confidence = count / len(matches)
            lift = confidence / (counts[label] / n)
            if count >= MIN_SUPPORT and confidence >= MIN_CONFIDENCE and lift > 1.0:
                rules.append(dict(antecedent=list(antecedent), candidate=label, support_count=count,
                    antecedent_count=len(matches), consequent_count=counts[label], total_records=n,
                    support=count/n, confidence=confidence, lift=lift,
                    source_record_ids=[rid for rid, candidate in matches if candidate == label]))
    rules.sort(key=lambda r: (-r['lift'], -r['support_count'], len(r['antecedent']), r['candidate'], r['antecedent']))
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0)
    tree.fit([[int(token in tokens) for token in vocabulary] for _, tokens, _ in training], [label for _, _, label in training])
    vector = [[int(token in current) for token in vocabulary]]
    path = []
    for node in tree.decision_path(vector).indices:
        index = tree.tree_.feature[node]
        if index >= 0:
            path.append(dict(feature=vocabulary[index], present=bool(vector[0][index]),
                             threshold=float(tree.tree_.threshold[node]), training_count=int(tree.tree_.n_node_samples[node])))
    leaf = int(tree.apply(vector)[0]); leaf_count = int(tree.tree_.n_node_samples[leaf])
    nominations = [dict(candidate=str(label), leaf_fraction=float(fraction), leaf_count=leaf_count,
                        calibration='training_leaf_fraction_not_accuracy')
                   for label, fraction in zip(tree.classes_, tree.predict_proba(vector)[0])
                   if fraction * leaf_count >= MIN_SUPPORT]
    return {**output, 'status': 'available', 'association_rules': rules[:12], 'tree_path': path,
            'tree_nominations': nominations,
            'candidate_range': sorted({r['candidate'] for r in rules[:12]} | {r['candidate'] for r in nominations})}
