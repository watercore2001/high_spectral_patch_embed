from torchmetrics import (MetricCollection, Accuracy, Precision,
                          Recall, F1Score, JaccardIndex, CohenKappa, ConfusionMatrix)


def generate_classification_metric(num_classes: int, ignore_index: int = None) -> tuple:
    metric_classes = [Accuracy, JaccardIndex, Precision, Recall, F1Score]

    global_metric_dict = {}
    classes_metric_dict = {}

    # add common metrics
    for metric in metric_classes:
        key = f"{metric.__name__.lower()}_micro"
        global_metric_dict[key] = metric(task="multiclass", average="micro", num_classes=num_classes,
                                         ignore_index=ignore_index)
        key = f"{metric.__name__.lower()}_macro"
        global_metric_dict[key] = metric(task="multiclass", average="macro", num_classes=num_classes,
                                         ignore_index=ignore_index)
        key = f"{metric.__name__.lower()}_class"
        classes_metric_dict[key] = metric(task="multiclass", average="none", num_classes=num_classes,
                                          ignore_index=ignore_index)

    # add top-5 accuracy
    global_metric_dict[f"{Accuracy.__name__.lower()}_top5_micro"] = Accuracy(task="multiclass", average="micro",
                                                                             top_k=5, num_classes=num_classes,
                                                                             ignore_index=ignore_index)
    global_metric_dict[f"{Accuracy.__name__.lower()}_top5_macro"] = Accuracy(task="multiclass", average="macro",
                                                                             top_k=5, num_classes=num_classes,
                                                                             ignore_index=ignore_index)

    # add kappa
    global_metric_dict[f"{CohenKappa.__name__.lower()}"] = CohenKappa(task="multiclass", num_classes=num_classes,
                                                                      ignore_index=ignore_index)
    confusion_matrix = ConfusionMatrix(task="multiclass", normalize="true",
                                       num_classes=num_classes, ignore_index=ignore_index)

    return MetricCollection(global_metric_dict), MetricCollection(classes_metric_dict), confusion_matrix


def separate_classes_metric(classes_metric_value: dict[str, list[float]]) -> dict[str, float]:
    classes_metric_dict = {}
    for metric_name, classes_values in classes_metric_value.items():
        for class_id, class_value in enumerate(classes_values):
            classes_metric_dict[f"{metric_name}_{class_id}"] = class_value

    return classes_metric_dict
