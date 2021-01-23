#!/bin/sh

python mnist.py --pseudo-init --pickle_stats
python mnist.py --pseudo-init --pickle_stats
python mnist.py --pseudo-init --pickle_stats
python mnist.py --pseudo-init --pickle_stats
python mnist.py --pseudo-init --pickle_stats

python mnist.py --pickle_stats
python mnist.py --pickle_stats
python mnist.py --pickle_stats
python mnist.py --pickle_stats
python mnist.py --pickle_stats