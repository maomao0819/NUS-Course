import os
for file in os.listdir('./'):
    if 'block' in file:
        os.remove(file)