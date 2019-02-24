import random

nega_sr = []

# between 0-9, avoid 4

can_srs = random.choices(range(0,9), k=10)
for can_sr in can_srs:
    if can_sr >= 4:
        can_sr += 1
    nega_sr.append(can_sr)
print(nega_sr)