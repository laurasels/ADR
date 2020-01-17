import os,sys
from pathlib import Path

path = Path('/home/killaarsl/Documents/adr/Cameradata/')

output = 'saveimages'

filelist = os.listdir(path)

empty_command = 'vlc "%s" --video-filter=scene --vout=dummy --scene-ratio=1 --scene-path="temp" vlc://quit'

for file in filelist:
    if file.endswith('MP4'):
        print(file)
        print(empty_command %(file))
        command = empty_command %(file)
        os.system(command)

        for outputfile in os.listdir(str(path / 'temp')):
            os.rename(str(path / 'temp' / outputfile),str(path / output / str(file.replace('.MP4','')+'_'+outputfile)))


