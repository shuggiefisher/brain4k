import re
import json
import urllib2
import logging

from data import Data
from settings import template_env


def render_pipeline(config, transforms):
    pipeline_figure = Data(
        'pipeline_graph',
        config,
        {'local_filename': 'pipeline.dot', 'data_type': 'figure'}
    )
    pipeline_md = Data(
        'pipeline_figure',
        config,
        {'local_filename': 'pipeline_graph.md', 'data_type': 'markdown'}
    )

    pipeline_template = template_env.get_template('templates/pipeline.dot')
    pipeline_dot = pipeline_template.render(transforms=transforms)
    pipeline_figure.io.save(pipeline_dot)

    uglified_pipeline = re.sub('[\n\t\s]+', ' ', pipeline_dot)
    gravizo_base_url = "http://g.gravizo.com/g?"
    pipeline_image_url = gravizo_base_url + urllib2.quote(uglified_pipeline)
    if len(pipeline_image_url) > 500:
        try:
            short_url = shorten_url(pipeline_image_url)
        except Exception as e:
            logging.error("Unable to shorten url: {0}".format(e))
            short_url = pipeline_image_url
    else:
        short_url = pipeline_image_url

    pipeline_md.io.write(
        'templates/pipeline_figure.md',
        {'short_url': short_url}
    )

    return pipeline_md.filename


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
