import os
import random
out = open("groups_train_Exposure.txt",'w')
lines=[]
with open("Exposure.txt", 'r') as infile:
     for line in infile:
         lines.append(line)
random.shuffle(lines)
for line in lines:
    out.write(line)

infile.close()
out.close()

