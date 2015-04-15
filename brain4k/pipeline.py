import os
import json
import logging
from itertools import chain

from data import path_to_file, Data
from transforms import TRANSFORMS
from graph import render_pipeline, pipeline_md_for_name


def execute_pipeline(repo_path, pipeline_name, pipeline_args=[], cache_stages=True, force_render_metrics=False):
    with open(path_to_file(repo_path, 'pipeline.json'), 'r+') as config_file:
        config = json.loads(config_file.read(), encoding='utf-8')
        config['repo_path'] = repo_path

        init_env(repo_path)

        transforms = []
        named_stages = config['pipelines'].get(pipeline_name, None)

        if not named_stages:
            if len(config['pipelines']) == 1:
                named_stages = [config['pipelines'].keys()[0]]
            else:
                raise ValueError(
                    "Pipeline.json does not contain a stage named '{0}'"
                    .format(pipeline_name)
                )
        else:
            pipeline_is_ephemeral = named_stages.get('ephemeral', False)
            named_stages = named_stages['stages']

        for stage in named_stages:
            module_name, class_name = TRANSFORMS[config['transforms'][stage['transform']]['transform_type']].rsplit('.',1)
            module = __import__(module_name, fromlist=[class_name])
            transform_cls = getattr(module, class_name)
            transform = transform_cls(stage, config, pipeline_is_ephemeral)
            transforms.append(transform)

        if pipeline_is_ephemeral:
            cached_stages = [False for s in xrange(len(transforms))]
        else:
            cached_stages = detect_changes(transforms, named_stages)
        metrics_updated = False

        for stage_index, (transform, stage, stage_is_cached) in enumerate(zip(transforms, named_stages, cached_stages)):
            if cache_stages and stage_is_cached:
                logging.info("Skipping stage {0} (cached)".format(stage_index + 1))
                continue
            else:
                logging.info("Starting stage {0}".format(stage_index + 1))
                methods = stage.get('actions', [])

                # if the transform is expecting any in-memory arguments passed
                # to it, make sure they are passed
                input_arguments = [t for t in transform.inputs if t.data_type == 'argument']
                if pipeline_args and input_arguments:
                    if len(pipeline_args) != len(input_arguments):
                        raise ValueError(
                            "Argument mismatch: {0} passed arguments {1}"
                            ", but accepts {2}"
                            .format(transform.transform_name, pipeline_args, input_arguments)
                        )
                    for argument, input_arg in zip(pipeline_args, input_arguments):
                        input_arg.value = argument

                actions = transform.chain(methods)

                # if it outputs anything, store these as pipeline_args in case
                # the next stage is expecting them
                output_arguments = [t for t in transform.outputs if t.data_type == 'argument']
                if output_arguments:
                    # this check is just to ensure output arguments are
                    # explicitly documented in pipeline.json
                    pipeline_args = list(chain.from_iterable([a for a in actions if a]))

                if not pipeline_is_ephemeral:
                    named_stages[stage_index]['sha1'] = transform.compute_hash()

                # does this stage output a metric?
                if set(config.get('metrics', [])) & set(named_stages[stage_index]['outputs']) \
                    or not os.path.exists(path_to_file(config['repo_path'], 'README.md')):
                    metrics_updated = True

                del config['repo_path']

                config_file.seek(0)
                config_file.write(
                    json.dumps(
                        config,
                        sort_keys=True,
                        indent=4,
                        ensure_ascii=False
                    )
                )

                config['repo_path'] = repo_path

        if False in cached_stages or force_render_metrics:
            # a better way would be store the hash of the pipeline.json
            # and check if it has changed
            # emphemeral pipelines will be re-rendered every time :\
            # TODO: make render_metrics into a hidden stage
            render_metrics(config, transforms, pipeline_name)


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
            if not transform.blob_files_exist():
                break
            current_stage_hash = transform.compute_hash()
            if stage_hash != current_stage_hash:
                break

        cached_stages[index] = True

    return cached_stages


def render_metrics(config, transforms, pipeline_name):
    """
    Concatenate the markdown files that make up the metrics.
    Output it as the README.md
    """
    input_files = []
    output_file = path_to_file(config['repo_path'], 'README.md')

    for named_pipeline in config['pipelines']:
        # render the named pipeline specified by the user
        # if the other pipelines have already been rendered, include
        # them in the output
        if named_pipeline == pipeline_name:
            pipeline_graph = render_pipeline(config, transforms, pipeline_name)
            input_files.append(pipeline_graph)
        else:
            pipeline_md = pipeline_md_for_name(config, named_pipeline)
            if os.path.exists(pipeline_md.filename):
                input_files.append(pipeline_md.filename)

    header_file = path_to_file(config['repo_path'], 'HEADER.md')
    if os.path.exists(header_file):
        input_files.append(header_file)

    for metric_name in config.get('metrics', []):
        datum = Data(metric_name, config, config['data'][metric_name])
        input_files.append(datum.filename)

    with open(output_file, 'w') as outfile:
        for i, fname in enumerate(input_files):
            with open(fname) as infile:
                if i == 2:
                    outfile.write("\n\n# Pipeline performance metrics\n")
                outfile.write("\n\n{0}\n".format(infile.read()))

