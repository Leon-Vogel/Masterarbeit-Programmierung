from _dir_init import *
from tensorboard import program
import os
import path_definitions
from pathlib import Path


def open_newest(log_path):
    subdirs = sorted(Path(log_path).iterdir(), key=os.path.getmtime, reverse=True)
    subdirs = list(filter(lambda x: "monitor.csv" not in x.name, subdirs))
    #subdirs.pop(0)
    newest = os.path.join(log_path, subdirs[0].name)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', newest])
    url = tb.launch()
    print(f"\n\nTensorflow listening on: {url}")

def load_all_logs():
    subdirs = list(map(lambda x: x[0],os.walk(log_path)))
    subdirs.pop(0)
    i = 0
    for d in subdirs:
        print(f"[{i}] {d.split('/')[-1]}")
        i = i+1
    print(f"[x] REFRESH")

    n = str(input('\n\n\nLog numer to open: '))
    if n.lower() == "x":
        load_all_logs()
        
    logdir = subdirs[int(n)]

    print(f'\n\n\nOpen tensorboard log: {logdir}\n==============================')
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])

    url = tb.launch()
    print(f"\n\nTensorflow listening on: {url}")

    n = str(input('\n\n\n[x] to reload or anything else to stop'))
    if n.lower() == "x":
        load_all_logs()

if __name__ == "__main__":
    #open_newest()
    load_all_logs()