#!/bin/bash
# Check CPU state to diagnose performance variance

echo "=== CPU Frequency Information ==="
if [ -f /proc/cpuinfo ]; then
    echo "Current CPU frequencies:"
    grep "MHz" /proc/cpuinfo | head -8
    echo ""
fi

if command -v lscpu &> /dev/null; then
    echo "CPU Architecture:"
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|MHz"
    echo ""
fi

echo "=== CPU Governor (frequency scaling) ==="
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    echo "CPU0 governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
    echo "CPU0 current freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 'N/A') kHz"
    echo "CPU0 max freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 'N/A') kHz"
else
    echo "Frequency scaling info not available"
fi
echo ""

echo "=== CPU Temperature (if available) ==="
if command -v sensors &> /dev/null; then
    sensors | grep -E "Core|Package|CPU"
else
    echo "lm-sensors not installed (install with: sudo apt install lm-sensors)"
fi
echo ""

echo "=== CPU Load ==="
uptime
echo ""

echo "=== Top CPU-consuming processes ==="
ps aux --sort=-%cpu | head -6
echo ""

echo "=== Memory Usage ==="
free -h
echo ""

echo "Tip: If CPU governor is 'powersave', run:"
echo "  sudo cpupower frequency-set --governor performance"
echo "to prevent CPU throttling."
