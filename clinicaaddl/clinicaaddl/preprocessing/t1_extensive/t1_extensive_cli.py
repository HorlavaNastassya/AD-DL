# coding: utf8

import clinicaml.engine as ce


class T1ExtensiveCli(ce.CmdParser):
    def define_name(self):
        """Define the sub-command name to run this pipeline."""
        self._name = "t1-extensive"

    def define_description(self):
        """Define a description of this pipeline."""
        self._description = (
            "Skull stripping of GM maps segmented by SPM in MNI space\n"
            "https://clinicadl.readthedocs.io/en/latest/Run/T1_Extensive/"
        )

    def define_options(self):
        """Define the sub-command arguments."""
        from clinicaml.engine.cmdparser import PIPELINE_CATEGORIES

        # Clinica compulsory arguments (e.g. BIDS, CAPS, group_label)
        clinica_comp = self._args.add_argument_group(
            PIPELINE_CATEGORIES["CLINICA_COMPULSORY"]
        )
        clinica_comp.add_argument(
            "caps_directory",
            help="Path to the CAPS directory containing the outputs of t1-volume.",
        )
        # Clinica standard arguments (e.g. --n_procs)
        self.add_clinica_standard_arguments()

    def run_command(self, args):
        """Run the pipeline with defined args."""
        from networkx import Graph
        from .t1_extensive_pipeline import T1Extensive
        from clinica.utils.ux import print_end_pipeline, print_crash_files_and_exit

        pipeline = T1Extensive(
            caps_directory=self.absolute_path(args.caps_directory),
            tsv_file=self.absolute_path(args.subjects_sessions_tsv),
            base_dir=self.absolute_path(args.working_directory),
            parameters={},
            name=self.name,
        )

        if args.n_procs:
            exec_pipeline = pipeline.run(
                plugin="MultiProc", plugin_args={"n_procs": args.n_procs}
            )
        else:
            exec_pipeline = pipeline.run()

        if isinstance(exec_pipeline, Graph):
            print_end_pipeline(
                self.name, pipeline.base_dir, pipeline.base_dir_was_specified
            )
        else:
            print_crash_files_and_exit(args.logname, pipeline.base_dir)
