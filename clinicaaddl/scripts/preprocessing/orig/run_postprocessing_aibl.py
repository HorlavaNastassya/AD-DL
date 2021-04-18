####################

from T1_postprocessing import postprocessing_t1w

## run the pipeline
 # for lustre
caps_directory = '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL'
working_dir = '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/working_dir/AIBL'
tsv = '/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_qc/AIBL_after_qc.tsv'
patch_size = 50
stride_size = 50

# for original
#wf = postprocessing_t1w(caps_directory, tsv, patch_size, stride_size, working_directory=working_dir, extract_method='slice')
#wf.run(plugin='MultiProc', plugin_args={'n_procs': 28})

# for rgb tf learning
#wf = postprocessing_t1w(caps_directory, tsv, patch_size, stride_size, working_directory=working_dir, extract_method='slice', slice_mode='rgb')
#wf.run(plugin='MultiProc', plugin_args={'n_procs': 28})

## for patch
wf = postprocessing_t1w(caps_directory, tsv, patch_size, stride_size, working_directory=working_dir, extract_method='patch')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 28})
