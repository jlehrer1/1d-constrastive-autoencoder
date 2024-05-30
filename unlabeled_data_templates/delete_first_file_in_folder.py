import os

#Delete the first file in the folder
for folder in os.listdir('ephys'):
    path = os.path.join('ephys', folder)
    path = os.path.join(path,'derived/kilosort2')
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
        print(f"Deleted {file}")
        break #Delete only the first file in the folder