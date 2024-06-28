import os
import pathlib

path = pathlib.Path(__file__).parent.absolute()

files = []
last_folder  = os.path.basename(path)
all_list = [last_folder]
#Import the .py files with meet condition last_folder_name +str("_")
#Example: last_folder_name = dataloader
#Include dataloader_0, dataloader_1,...
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and last_folder+str("_") in i:
        all_list.append(i[:-3])
__all__ = all_list


