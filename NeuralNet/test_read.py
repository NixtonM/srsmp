import json
import os




with open('Thorlabs_Red.json','r') as fp:
    y = json.load(fp)

with open('pc_01.json','r') as fp:
    x = json.load(fp)
c=0