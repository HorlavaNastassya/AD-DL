
def plot_generic(
        predictions_path,
        prefix,
        selection_metrics=None,
        verbose=0,
        function=None,
        **kwards
):
    output_dir = os.path.join(output_dir_general, network)
    output_dir = pathlib.Path(output_dir)
    modelPatter = "subject_model*"
    folders = [f for f in output_dir.glob(modelPatter)]
    for f in folders[:]:
        currentDirectory = pathlib.Path(f)
        currentPattern = "fold-*"
        path_params = os.path.join(currentDirectory, "commandline_train.json")

        with open(path_params, "r") as f:
            params = json.load(f)

        # loop depending the number of folds found in the model folder
        for fold_dir in currentDirectory.glob(currentPattern):
            fold = int(str(fold_dir).split("-")[-1])
            results = {}

            selection_metrics = []
            performance_dir = os.path.join(currentDirectory, 'fold-%i' % fold, 'cnn_classification')
            for f in os.scandir(performance_dir):
                if os.path.basename(os.path.normpath(f.path)) in possible_selection_metrics:
                    selection_metrics.append(os.path.basename(os.path.normpath(f.path)))

            for selection_metric in selection_metrics:
                results[selection_metric] = {}
                history = pd.read_csv(os.path.join(currentDirectory, 'fold-%i' % fold, 'training.tsv'),
                                      sep='\t')
                modes = ['train', 'validation']
                for ms_el in MS_list:
                    modes.append('test_' + ms_el)

                for mode in modes:
                    test_diagnosis_path = os.path.join(performance_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))
                    test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                    results[selection_metric][mode] = test_diagnosis_df[
                        ["sensitivity", 'specificity', 'balanced_accuracy', 'accuracy']]

            folder_type = 'barplots_with_loss'

            preprocessing = params["preprocessing"]

            augmentations_presense = 'augmentations' if params['data_augmentation'] else "no_augmentations"
            network_type = params['model']

            loss = params['loss']

            params['training MS'] = MS

            name = os.path.basename(os.path.normpath(currentDirectory))
            folder_type = 'barplots_with_loss'
            path = os.path.join(results_folder, folder_type, preprocessing, network_type,
                                augmentations_presense, loss)
            os.makedirs(path, exist_ok=True)
            file_name = 'barplots_' + name + '.png'
            plot_loss_with_results(params, results, history, os.path.join(path, file_name))
            for ms_el in MS_list:
                for mode in results.keys():
                    if results[mode]["test_" + ms_el]["balanced_accuracy"][0] > \
                            reported_best_accuracies[network][ms_el]["max_value"]:
                        reported_best_accuracies[network_type][ms_el]["max_value"] = \
                            results[mode]["test_" + ms_el]["balanced_accuracy"][0]
                        reported_best_accuracies[network_type][ms_el]["mode"] = mode
                        reported_best_accuracies[network_type][ms_el]["results"] = results
                        reported_best_accuracies[network_type][ms_el]["params"] = params
                        reported_best_accuracies[network_type][ms_el]["history"] = history
                        reported_best_accuracies[network_type][ms_el]["model_name"] = name



