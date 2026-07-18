import json
import os

path = 'data/soccernet/mvfouls/test/annotations.json'
with open(path) as f:
    data = json.load(f)

print('Top level keys:', list(data.keys()))
print('Number of actions:', data['Number of actions'])
print()

actions = data['Actions']
first_key = list(actions.keys())[0]
print('First action key:', first_key)
print('First action data:')
for k, v in actions[first_key].items():
    print(f'  {k}: {v}')