import os

f = 'loocv'

os.system('rm -r ' + f)
os.system('mkdir ' + f)

for i in range(1, 151):
    os.system('mkdir {}/fold_{}'.format(f, str(i)))
