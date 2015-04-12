import re
import os
import json
import urllib2
import logging

from data import Data
from settings import template_env


def render_pipeline(config, transforms, pipeline_name):
    pipeline_figure = Data(
        'pipeline_graph',
        config,
        {
            'local_filename': '{0}_pipeline.dot'.format(pipeline_name),
            'data_type': 'figure'
        }
    )
    pipeline_md = pipeline_md_for_name(config, pipeline_name)

    pipeline_template = template_env.get_template('templates/pipeline.dot')
    pipeline_dot = pipeline_template.render(transforms=transforms)
    pipeline_figure.io.save(pipeline_dot)

    uglified_pipeline = re.sub('[\n\t\s]+', ' ', pipeline_dot)
    gravizo_base_url = "http://g.gravizo.com/g?"
    pipeline_image_url = gravizo_base_url + urllib2.quote(uglified_pipeline)
    if len(pipeline_image_url) <= 500:
        # if url is short no need to shorten it
        short_url = pipeline_image_url
    elif len(pipeline_image_url) >= 3340:
        # if url is very long, will need to use graphviz locally to render
        short_url = render_dot_locally(uglified_pipeline, config, pipeline_name)
    else:
        # use google to shorten the url
        try:
            short_url = shorten_url(pipeline_image_url)
        except Exception as e:
            logging.error("Unable to shorten url: {0}".format(e))
            short_url = pipeline_image_url

    pipeline_md.io.write(
        'templates/{0}_pipeline_figure.md'.format(pipeline_name),
        {'short_url': short_url, 'pipeline_name': pipeline_name}
    )

    return pipeline_md.filename


def pipeline_md_for_name(config, pipeline_name):
    pipeline_md = Data(
        'pipeline_figure',
        config,
        {
            'local_filename': '{0}_pipeline_graph.md'.format(pipeline_name),
            'data_type': 'markdown'
        }
    )
    return pipeline_md


def shorten_url(long_url):
    url = "https://www.googleapis.com/urlshortener/v1/url"
    req = urllib2.Request(
        url,
        json.dumps({'longUrl': long_url}),
        {'Content-Type': 'application/json'}
    )
    f = urllib2.urlopen(req)
    response = f.read()
    data = json.loads(response)
    f.close()
    short_url = data.get('id', None)

    if not short_url:
        raise Exception(
            "URL shortener did not return result for url: {0}"
            .format(long_url)
        )

    return short_url


def render_dot_locally(dot_string, config, pipeline_name):
    try:
        import pygraphviz as pgv
    except ImportError as e:
        raise ImportError(
            "pygraphviz required to render complex pipelines: {0}".format(e)
        )
    else:
        pipeline_graph = pgv.AGraph(dot_string)
        rendered_pipeline = Data(
            'rendered_pipeline',
            config,
            {
                'local_filename': '{0}_pipeline.png'.format(pipeline_name),
                'data_type': 'figure'
            }
        )

        relative_path = os.path.relpath(
            rendered_pipeline.filename,
            config['repo_path']
        )

        pipeline_graph.draw(rendered_pipeline.filename, prog='dot')

        return relative_path
