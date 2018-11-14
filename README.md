# tf summary extractor

tensorflowのsummaryからデータを抽出する．

## Usage

```
usage: summary_extractor.py [-h] [-s STEP]
                            dirpath eventsfile [tags [tags ...]]

extract data from tensorboard events file.

positional arguments:
  dirpath               extracted data will be saved in this directory.
  eventsfile            eventfile
  tags                  extract data tag. you can use regular expression.
                        `tensorboard --inspect --logdir=[logdir]` tells you
                        available tags.

optional arguments:
  -h, --help            show this help message and exit
  -s STEP, --step STEP  extracted step. if `step` is None, all data will be
                        extracted. Default: None
```
