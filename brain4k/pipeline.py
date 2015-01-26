import os
import json
import logging

from data import path_to_file, Data
from transforms import TRANSFORMS


def execute_pipeline(repo_path, cache_stages=True):
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
        metrics_updated = False

        for stage_index, (transform, stage, stage_is_cached) in enumerate(zip(transforms, config['stages'], cached_stages)):
            if cache_stages and stage_is_cached:
                logging.info("Skipping stage {0} (cached)".format(stage_index + 1))
                continue
            else:
                logging.info("Starting stage {0}".format(stage_index + 1))
                methods = stage.get('actions', [])

                actions = transform.chain(methods)
                config['stages'][stage_index]['sha1'] = transform.compute_hash()

                # does this stage output a metric?
                if set(config['metrics']) & set(config['stages'][stage_index]['outputs']):
                    metrics_updated = True

                del config['repo_path']

                config_file.seek(0)
                config_file.write(json.dumps(config, indent=4, ensure_ascii=False))

                config['repo_path'] = repo_path

        if metrics_updated:
            render_metrics(config)


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
            if not varying_files_exist(transform, stage_config):
                break

        cached_stages[index] = True

    return cached_stages


def varying_files_exist(transform, stage_config):
    """
    Do the files we are not checking the hash for actually exist on the
    file system?
    """
    for varying_file in stage_config.get('accept_variance_in', []):
        for data in [transform.inputs, transform.outputs]:
            for datum in data:
                if datum.name == varying_file:
                    if not os.path.exists(datum.filename):
                        return False

    return True


def render_metrics(config):
    """
    Concatenate the markdown files that make up the metrics.
    Output it as the README.md
    """
    input_files = []
    output_file = path_to_file(config['repo_path'], 'README.md')

    header_file = path_to_file(
        config['repo_path'],
        os.path.join('metrics', 'HEADER.md')
    )
    if os.path.exists(header_file):
        input_files.append(header_file)

    for metric_name in config['metrics']:
        datum = Data(metric_name, config, config['data'][metric_name])
        input_files.append(datum.filename)

    with open(output_file, 'w') as outfile:
        for fname in input_files:
            with open(fname) as infile:
                outfile.write(infile.read())
