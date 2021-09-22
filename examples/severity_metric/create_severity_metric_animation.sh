#!/bin/sh
python -m HotGauge.thermal.metrics graph_metric_range
ffmpeg -y -i severity_metric_%03d.png -q:v 3 severity_metric.mp4
