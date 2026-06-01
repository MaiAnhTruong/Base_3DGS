#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                    if value is True:
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                    if value is True:
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 8
        self.full_eval_path = ""
        self.full_eval_images = "images"
        self.full_eval_sparse = "sparse/0"
        self.eval_hold = 8
        self.eval_overlap_shift = "backward"
        self.eval_boundary_forward_fallback = True
        self.eval_strict_backward_shift = False
        self.split_report_enable = True
        self.split_only = False
        self.sparse_train_images = ""
        self.sparse_train_indices = ""
        self.sparse_train_count = 0
        self.full_test_source_path = ""
        self.full_test_images = ""
        self.dpcr_eval_source_path = ""
        self.dpcr_eval_images = ""
        self.dpcr_eval_split_mode = "llffhold"
        self.dpcr_eval_llffhold = 8
        self.dpcr_train_view_list = ""
        self.dpcr_eval_test_view_list = ""
        self.dpcr_write_split_manifest = True
        self.dpcr_eval_require_disjoint = True
        self.dpcr_eval_frame_mode = "strict"
        self.dpcr_eval_alignment_min_common = 4
        self.dpcr_eval_frame_check_tol = 1e-3
        self.split_train_views = "off"
        self.split_hold = 8
        self.split_output_root = ""
        self.split_name = ""
        self.split_copy_mode = "copy"
        self.split_force = False
        self.split_validate_only = False
        self.split_train_sample_mode = "paper_even"
        self.split_strict_no_overlap = True
        self.split_init_policy = "sparsegs_triangulate"
        self.split_colmap_exe = "colmap"
        self.split_colmap_matcher = "exhaustive"
        self.split_require_all_train_registered = True
        self.split_min_train_points = 100
        self.split_min_triangulated_points = 100
        self.split_strict_sparsegs = True
        self.external_test_source_path = ""
        self.auto_split_report_path = ""
        self.auto_split_validation_report_path = ""
        self.source_path_original = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if getattr(g, "model_path", ""):
            g.model_path = os.path.abspath(g.model_path)
        if getattr(g, "full_eval_path", ""):
            g.full_eval_path = os.path.abspath(g.full_eval_path)
        if g.full_test_source_path:
            g.full_test_source_path = os.path.abspath(g.full_test_source_path)
        if getattr(g, "dpcr_eval_source_path", ""):
            g.dpcr_eval_source_path = os.path.abspath(g.dpcr_eval_source_path)
        if getattr(g, "dpcr_train_view_list", ""):
            g.dpcr_train_view_list = os.path.abspath(g.dpcr_train_view_list)
        if getattr(g, "dpcr_eval_test_view_list", ""):
            g.dpcr_eval_test_view_list = os.path.abspath(g.dpcr_eval_test_view_list)
        if getattr(g, "split_output_root", ""):
            g.split_output_root = os.path.abspath(g.split_output_root)
        if getattr(g, "external_test_source_path", ""):
            g.external_test_source_path = os.path.abspath(g.external_test_source_path)
        if getattr(g, "auto_split_report_path", ""):
            g.auto_split_report_path = os.path.abspath(g.auto_split_report_path)
        if getattr(g, "auto_split_validation_report_path", ""):
            g.auto_split_validation_report_path = os.path.abspath(g.auto_split_validation_report_path)
        if not getattr(g, "source_path_original", ""):
            g.source_path_original = g.source_path
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
