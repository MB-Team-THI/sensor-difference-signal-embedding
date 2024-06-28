import os
import pathlib

path = pathlib.Path(__file__).parent.absolute()

files = []
last_folder  = os.path.basename(path)
all_list = [last_folder]
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and last_folder+str("_") in i:
        all_list.append(i[:-3])
__all__ = all_list

