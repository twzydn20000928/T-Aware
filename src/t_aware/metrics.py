"""Binary metrics with explicit F1 averaging and undefined denominators."""


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Expected equal, nonempty label/prediction sequences")
    if any(x not in (0, 1) for x in list(labels) + list(predictions)):
        raise ValueError("Labels and predictions must be binary")
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    divide = lambda a, b: a / b if b else None
    f1_ai = divide(2 * tp, 2 * tp + fp + fn)
    f1_human = divide(2 * tn, 2 * tn + fp + fn)
    ai_recall = divide(tp, tp + fn)
    human_recall = divide(tn, tn + fp)
    both = tp + fn > 0 and tn + fp > 0
    return {"n": len(labels), "human_n": tn + fp, "ai_n": tp + fn,
            "accuracy": (tp + tn) / len(labels), "ai_recall": ai_recall,
            "human_recall": human_recall, "false_positive_rate": divide(fp, tn + fp),
            "ai_precision": divide(tp, tp + fp), "f1_ai": f1_ai,
            "f1_macro": (f1_ai + f1_human) / 2 if both else None,
            "balanced_accuracy": (ai_recall + human_recall) / 2 if both else None,
            "confusion_matrix": [[tn, fp], [fn, tp]]}
