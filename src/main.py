import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from pynvml import *

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("alpha")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def open_yaml_file(path, config_name = None):
    with open(path, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


def _get_val_by_argument(params, argument):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == argument:
            return _v.split("=")[1]

def _get_gpu_id(params):
    return _get_val_by_argument(params, "--gpu_id")

def _get_map_config(params):
    map_name = _get_val_by_argument(params, "--map_name")
    map_train_traintime = open_yaml_file(os.path.join(os.path.dirname(__file__), "config", "map_traintime.yaml"), "map_traintime_config")
    map_train_attribute = open_yaml_file(os.path.join(os.path.dirname(__file__), "config", "map_attributes.yaml"), "map_attributes_config")

    map_config = {
        "map_name": map_name,
        "t_max": map_train_traintime[map_name],
        "ally_num": map_train_attribute[map_name][0],
        "enemy_num": map_train_attribute[map_name][1],
        "token_dim": map_train_attribute[map_name][2]
    }

    return map_config

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        return open_yaml_file(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), config_name)


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    map_config = _get_map_config(params)

    env_config["env_args"]["map_name"] = map_config["map_name"]
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, map_config)
    config_dict["agent"] = _get_val_by_argument(params, "--agent")
    config_dict["use_cuda"] = th.cuda.is_available()
    config_dict["gpu_id"] = _get_gpu_id(params)

    if th.cuda.is_available():
        nvmlInit()

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "experiment{}".format(config_dict["experiment"]), "sacred", config_dict["map_name"], config_dict["mixer"])
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # ex.run_commandline(params)
    ex.run()
