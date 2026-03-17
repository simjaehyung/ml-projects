import re

with open('c:/Users/jhsim/Erica261/머신러닝/presentation_visualizer.html', 'r', encoding='utf-8') as f:
    text = f.read()

m = re.findall(r'<img [^>]*>', text)
print(f'Found {len(m)} img tags in presentation_visualizer')
for match in m:
    print(match[:100])
