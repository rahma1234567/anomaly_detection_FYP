import os
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
log_file = f"logs/anomalies_{today}.log"

print(f"Log file size: {os.path.getsize(log_file)} bytes")

with open(log_file) as f:
    lines = f.readlines()
print(f"\nTotal log entries: {len(lines)}")
print("\nLast 3 entries:")
for line in lines[-3:]:
    print(f"  {line.strip()}")