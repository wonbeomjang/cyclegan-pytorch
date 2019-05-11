# code.py
# Link:https://code.i-harness.com/ko/q/6e87e6
import os

def download_dataset(from_style, to_style):
    print(f"[!] download {to_style}2{from_style}")
    os.system(os.path.join(f'download_dataset.sh {to_style}2{from_style}'))
