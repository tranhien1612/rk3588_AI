#!/bin/bash
# Cố định xung CPU/NPU/DDR để FPS ổn định (cần root): sudo tools/perf_mode.sh
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $p/scaling_governor; done
for d in /sys/class/devfreq/*npu*; do [ -e $d/governor ] && echo performance > $d/governor; done
for d in /sys/class/devfreq/*dmc*; do [ -e $d/governor ] && echo performance > $d/governor; done
echo done
