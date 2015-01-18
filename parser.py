import os
import json
import importlib

from components import path_to_file
from models import MODELS
from data_interfaces import compute_json_hash, compute_file_hash


IMAGENET_EXTRACTION = 'gitbrains/imagenet'


def imagenet_parser(repo_path=IMAGENET_EXTRACTION, cache_stages=True):
    with open(path_to_file(repo_path, 'pipeline.json'), 'r+') as config_file:
        import ipdb; ipdb.set_trace()
        config = json.loads(config_file.read(), encoding='utf-8')
        config['repo_path'] = repo_path

        init_env(repo_path)

        models = []
        for stage in config['stages']:
            model_cls = importlib.import_module(MODELS[stage['model_type']])
            model = model_cls(stage, config)
            models.append(model)

        is_cached = detect_changes(models, config['stages'])

        for model, stage in zip(models, config['stages'], is_cached):
            if cache_stages and is_cached:
                continue
            else:
                methods = getattr(model, stage['action'], [])

                # check if output files exist on disk, and hashes match
                model.chain(methods)
                config = store_hashes(model, config)

        config_file.seek(0)
        config_file.write(json.dumps(config, indent=4, ensure_ascii=False))
        # auto-commit ?


def init_env(repo_path):
    cachedir = os.path.join(repo_path, 'cache')
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)


def store_hashes(model, config):
    """
    Following the execution of a model, rewrite the hashes for the files.

    Raise an exception if a hash was specified for an output, the model
    was re-run, and the new output has a different hash.
    """
    for output in model.outputs:
        filehash = compute_file_hash(output.filename)
        if output.sha1 and filehash != output.sha1:
            raise Exception(
                "Generated contents of {0} does not match the sha1 hash"
                " specified in pipeline.json.  The pipeline may generate"
                " different results".format(output.filename)
            )
        else:
            config['data'][output.name]['sha1'] = filehash

    for data_type in model.inputs + model.files:
        filehash = compute_file_hash(output.filename)
        config['data'][output.name]['sha1'] = filehash

    return config


def detect_changes(models, stage_configs):
    """
    Check preceeding stages have not changed AND files match hashes, including parameter files
    """
    is_cached = [False for model in models]
    for index, (model, stage_config) in enumerate(zip(models, stage_configs)):
        # has stage config changed?
        stage_hash = model.get('sha1', None)
        if not stage_hash:
            break
        else:
            current_stage_hash = compute_json_hash(stage_config)
            if stage_hash != current_stage_hash:
                break

        # are the input hashes present, and do they match
        for input_data in model.inputs:
            if not input_data.matches_hash():
                break

        # are the model files hashes present, and do they match
        for model_file in model.files:
            if not model_file.matches_hash():
                break

        # are the output hashes present, and do they match
        for output_data in model.outputs:
            if not output_data.matches_hash():
                break

        is_cached[index] = True

    return is_cached