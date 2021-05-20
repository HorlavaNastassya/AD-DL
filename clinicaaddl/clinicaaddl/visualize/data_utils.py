
def get_results(predictions_dir, MS_list, fold):
    possible_selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]
    import pandas as pd
    import os

    results = {}
    selection_metrics = []
    cnn_classification_dir = os.path.join(predictions_dir, 'fold-%i' % fold, 'cnn_classification')
    for f in os.scandir(cnn_classification_dir):
        if os.path.basename(os.path.normpath(f.path)) in possible_selection_metrics:
            selection_metrics.append(os.path.basename(os.path.normpath(f.path)))

    for selection_metric in selection_metrics:
        results[selection_metric] = {}
        modes = ['train', 'validation']
        for ms_el in MS_list:
            modes.append('test_' + ms_el)

        for mode in modes:
            test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                               '%s_image_level_metrics.tsv' % (mode))
            test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
            results[selection_metric][mode] = test_diagnosis_df[["sensitivity", 'specificity', 'balanced_accuracy', 'accuracy']]

    return results
