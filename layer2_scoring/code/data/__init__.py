"""Reading Layer 2's data and turning it into batches.

dataset.py    joins the three jsonl files into one row per scored pair
torch_data.py encodes those rows into padded tensor batches and owns the split
"""
