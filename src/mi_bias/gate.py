"""Prompt-level bias-sensitivity classifier with recall-prioritized thresholding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptSensitivityGate:
    vectorizer: object
    classifier: object
    threshold: float
    validation_metrics: dict[str, float]

    def predict_sensitive(self, prompts: list[str]) -> list[bool]:
        probabilities = self.classifier.predict_proba(self.vectorizer.transform(prompts))[:, 1]
        return [bool(score >= self.threshold) for score in probabilities]


def fit_prompt_gate(
    prompts: list[str], labels, *, groups=None, seed: int = 42, target_recall: float = 0.95
) -> PromptSensitivityGate:
    """Fit a TF-IDF/logistic gate and choose its threshold on held-out data.

    Labels must distinguish bias-relevant contexts from neutral contexts. Merely
    mentioning a protected group should not automatically imply a positive label.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    y = np.asarray(labels, dtype=int)
    if len(prompts) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Gate training needs aligned prompts with both 0/1 labels")
    indices = np.arange(len(y))
    if groups is None:
        train, validation = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=y
        )
    else:
        group_ids = np.asarray(groups)
        if len(group_ids) != len(y):
            raise ValueError("groups must contain one group id per prompt")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train, validation = next(splitter.split(indices, y, group_ids))
        if len(np.unique(y[train])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError("Group split must contain both sensitivity labels in each partition")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    x_train = vectorizer.fit_transform([prompts[i] for i in train])
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    classifier.fit(x_train, y[train])
    probabilities = classifier.predict_proba(
        vectorizer.transform([prompts[i] for i in validation])
    )[:, 1]

    candidates = sorted(set([0.0, 1.0, *probabilities.tolist()]))
    scored = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        recall = recall_score(y[validation], predicted, zero_division=0)
        precision = precision_score(y[validation], predicted, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y[validation], predicted, labels=[0, 1]).ravel()
        scored.append((threshold, recall, precision, fp, fn, tp, tn))
    feasible = [item for item in scored if item[1] >= target_recall]
    # If the requested recall cannot be reached, retain the highest-recall
    # threshold and make the shortfall visible in validation_metrics.
    selected = min(feasible, key=lambda item: (item[3], -item[2], -item[0])) if feasible else max(
        scored, key=lambda item: (item[1], item[2], item[0])
    )
    threshold, recall, precision, fp, fn, tp, tn = selected
    return PromptSensitivityGate(
        vectorizer, classifier, float(threshold), {
            "target_recall": float(target_recall),
            "recall": float(recall), "precision": float(precision),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
            "validation_examples": float(len(validation)),
        }
    )
