import shutil
import glob

def untar_files(model, year, month):
    directory = '/home/lgaudet/model-data/downloads/'
    #dirpath = f'{directory}{model}/{year}/{month}/'
    dirpath = directory
    print(f'{dirpath}{model.lower()}*.tar')
    files = glob.glob(f'{dirpath}{model.lower()}*.tar')
    
    if files:
        for fname in files:
            if fname.endswith('tar'):
                shutil.unpack_archive(fname, f'{dirpath}{model.lower()}')
    else:
        print('no tar files exist')
            
for month in range(1, 4):
    untar_files('NAM', '2022', str(month).zfill(2))