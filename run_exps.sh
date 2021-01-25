#!/bin/sh

python mnist.py --pseudo-init --pickle-stats
python mnist.py --pseudo-init --pickle-stats
python mnist.py --pseudo-init --pickle-stats

python mnist.py --pickle-stats --qbits 5
python mnist.py --pickle-stats --qbits 5
python mnist.py --pickle-stats --qbits 5

python eval.py