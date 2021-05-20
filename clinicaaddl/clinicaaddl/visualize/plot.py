

def plot_generic(
        predictions_path,
        output_path,
        magnet_strength,
        MS_list=[],
        function=None,
        **kwargs
):
    import pathlib
    import os
    import json
    from .plot_utils import Plots

    currentDirectory = pathlib.Path(predictions_path)
    currentPattern = "fold-*"
    path_params = os.path.join(currentDirectory, "commandline_train.json")

    with open(path_params, "r") as f:
        params = json.load(f)

    params['training MS'] = magnet_strength
    model_name = os.path.basename(os.path.normpath(currentDirectory))

    # loop depending the number of folds found in the model folder
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        #plot results depending on
        getattr(Plots, function)(predictions_path=predictions_path, output_path=output_path,
              MS_list=MS_list, model_params=params, fold=fold, model_name=model_name, **kwargs)





