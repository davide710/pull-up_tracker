import os

os.chdir('data/')
i = 1
for im in os.listdir():
    ext = im.split('.')[-1]
    newname = '000000' + str(i)
    newname = newname[-6:] + '.' + ext
    os.rename(im, newname)
    i += 1