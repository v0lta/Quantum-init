#!/bin/sh

python mnist.py --pseudo-init --pickle-stats
python mnist.py --pseudo-init --pickle-stats
python mnist.py --pseudo-init --pickle-stats
# python mnist.py --pseudo-init --pickle-stats
# python mnist.py --pseudo-init --pickle-stats

python mnist.py --pickle-stats
python mnist.py --pickle-stats
# python mnist.py --pickle-stats
# python mnist.py --pickle-stats
# python mnist.py --pickle-stats