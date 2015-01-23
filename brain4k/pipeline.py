import os
import json
import logging

from data import path_to_file
from transforms import TRANSFORMS

IMAGENET_EXTRACTION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'brain4k/imagenet'
)


def imagenet_parser(repo_path=IMAGENET_EXTRACTION, cache_stages=True):
    with open(path_to_file(repo_path, 'pipeline.json'), 'r+') as config_file:
        config = json.loads(config_file.read(), encoding='utf-8')
        config['repo_path'] = repo_path

        init_env(repo_path)

        transforms = []
        for stage in config['stages']:
            module_name, class_name = TRANSFORMS[config['transforms'][stage['transform']]['transform_type']].rsplit('.',1)
            module = __import__(module_name, fromlist=[class_name])
            transform_cls = getattr(module, class_name)
            transform = transform_cls(stage, config)
            transforms.append(transform)

        cached_stages = detect_changes(transforms, config['stages'])

        for stage_index, (transform, stage, stage_is_cached) in enumerate(zip(transforms, config['stages'], cached_stages)):
            if cache_stages and stage_is_cached:
                logging.info("Skipping stage {0} (cached)".format(stage_index + 1))
                continue
            else:
                logging.info("Starting stage {0}".format(stage_index + 1))
                methods = stage.get('actions', [])

                actions = transform.chain(methods)
                config['stages'][stage_index]['sha1'] = transform.compute_hash()

        del config['repo_path']

        if False in cached_stages:
            config_file.seek(0)
            config_file.write(json.dumps(config, indent=4, ensure_ascii=False))
            # auto-commit ?


def init_env(repo_path):
    cachedir = os.path.join(repo_path, 'cache')
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)


def detect_changes(transforms, stage_configs):
    """
    Check preceeding stages have not changed AND files match hashes, including parameter files
    """
    cached_stages = [False for transform in transforms]
    for index, (transform, stage_config) in enumerate(zip(transforms, stage_configs)):

        stage_hash = stage_config.get('sha1', None)
        if not stage_hash:
            break
        else:
            current_stage_hash = transform.compute_hash()
            if stage_hash != current_stage_hash:
                break

        cached_stages[index] = True

    return cached_stages