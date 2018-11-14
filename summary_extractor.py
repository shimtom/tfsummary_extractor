import argparse
import os
import re
from enum import Enum
from io import BytesIO
from typing import Any, List, NamedTuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


class SummaryType(Enum):
    SCALAR = 0
    IMAGE = 1
    HISTOGRAM = 2
    AUDIO = 3


class Value(NamedTuple):
    step: int
    value: Any


class Summary(NamedTuple):
    tag: str
    value: Value
    type: SummaryType


class Event(NamedTuple):
    step: int
    summaries: List[Summary] = []


def load_audio(audio_msg):
    raise NotImplementedError('extract audio not implemented')


def load_histo(histo_msg):
    raise NotImplementedError('extract histo not implemented')


def load_image(image_msg):
    if image_msg.ByteSize() == 0:
        raise ValueError('empty image.')
    image = Image.open(BytesIO(image_msg.encoded_image_string))
    return image


def load_event(values_msg, tag_patterns):
    summaries = []

    for v in values_msg:
        if sum([re.match(p, v.tag) is not None for p in tag_patterns]) == 0:
            continue

        if v.image.ByteSize() != 0:
            # summaries.append(Summary(type=SummaryType.IMAGE, tag=v.tag, value=load_image(v.image)))
            s = Summary(type=SummaryType.IMAGE, tag=v.tag, value=load_image(v.image))
            summaries.append(s)
        elif v.histo.ByteSize() != 0:
            summaries.append(Summary(type=SummaryType.HISTOGRAM, tag=v.tag, value=load_histo(v.histo)))
        elif v.audio.ByteSize() != 0:
            summaries.append(Summary(type=SummaryType.AUDIO, tag=v.tag, value=load_audio(v.audio)))
        else:
            summaries.append(Summary(type=SummaryType.SCALAR, tag=v.tag, value=v.simple_value))

    return summaries


def save_scalar(dirpath: str, tag: str, values: List[Value]):
    values.sort(key=lambda v: v.step)
    df = pd.DataFrame(columns=['step', 'value'])

    for v in values:
        df = df.append(pd.Series([int(v.step), v.value], index=df.columns, name=v.step))
    df.to_csv(os.path.join(dirpath, tag.replace('/', '_') + '.csv'), index=False)


def save_image(dirpath: str, tag: str, values: List[Value]):
    values.sort(key=lambda v: v.step)
    digit = str(len(str(len(values))))
    for v in values:
        step = v.step
        img = v.value
        filename = os.path.join(dirpath, tag.replace('/', '_') + ('_%0' + digit + 'd') % step)
        if img.format is not None:
            filename += '.' + img.format
        img.save(filename)


def save_histo(dirpath: str, tag: str, values: List[Value]):
    raise NotImplementedError()


def save_audio(dirpath: str, tag: str, values: List[Value]):
    raise NotImplementedError()


def extract(eventsfile, tag_patterns, step=None):
    iterator = tf.train.summary_iterator(eventsfile)

    events = []
    for e in iterator:
        if step is not None and e.step != step:
            continue
        if e.summary.value:
            events.append(Event(step=e.step, summaries=load_event(e.summary.value, tag_patterns)))

    scalars = dict()
    images = dict()
    histograms = dict()
    audios = dict()

    for e in events:
        step = e.step
        for summary in e.summaries:
            tag = summary.tag
            value = summary.value
            summary_type = summary.type
            if summary_type == SummaryType.SCALAR:
                scalars.setdefault(tag, []).append(Value(step=step, value=value))
            elif summary_type == SummaryType.IMAGE:
                images.setdefault(tag, []).append(Value(step=step, value=value))
            elif summary_type == SummaryType.HISTOGRAM:
                histograms.setdefault(tag, []).append(Value(step=step, value=value))
            elif summary_type == SummaryType.AUDIO:
                audios.setdefault(tag, []).append(Value(step=step, value=value))
            else:
                raise ValueError('Unknown summary type', summary_type)

    return scalars, images, histograms, audios


def main():
    parser = argparse.ArgumentParser(description='extract data from tensorboard events file.')
    parser.add_argument('dirpath', type=str, help='extracted data will be saved in this directory.')
    parser.add_argument('eventsfile', type=str, help='eventfile')
    parser.add_argument(
        'tags',
        type=str,
        nargs='*',
        default=['*'],
        help='extract data tag. you can use regular expression. `tensorboard --inspect --logdir=[logdir]` tells you available tags.',
    )
    parser.add_argument(
        '-s',
        '--step',
        type=int,
        default=None,
        help='extracted step. if `step` is None, all data will be extracted. Default: None',
    )
    args = parser.parse_args()
    dirpath = args.dirpath
    eventsfile = args.eventsfile
    tag_patterns = args.tags
    step = args.step

    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    scalars, images, histograms, audios = extract(eventsfile, tag_patterns, step)

    print('scalars', '' if scalars else '-')
    for k, v in scalars.items():
        print('    ' + k)
        save_scalar(dirpath, k, v)

    print('images', '' if images else '-')
    for k, v in images.items():
        save_image(dirpath, k, v)

    print('histograms', '' if histograms else '-')
    for k, v in histograms.items():
        save_histo(dirpath, k, v)

    print('audios', '' if audios else '-')
    for k, v in audios.items():
        save_audio(dirpath, k, v)


if __name__ == "__main__":
    main()
