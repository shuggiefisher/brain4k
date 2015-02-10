from data import Data
from settings import template_env


def render_pipeline(config, transforms):
    pipeline_figure = Data('pipeline_graph', config, {'local_filename': 'pipeline.dot', 'data_type': 'figure'})
    pipeline_md = Data('pipeline_figure', config, {'local_filename': 'pipeline_graph.md', 'data_type': 'markdown'})

    pipeline_template = template_env.get_template('templates/pipeline.dot')
    pipeline_dot = pipeline_template.render(transforms=transforms)
    pipeline_figure.io.save(pipeline_dot)

    pipeline_md.io.write(
        'templates/pipeline_figure',
        {'dot_graph': pipeline_dot}
    )

    return pipeline_md.filename
