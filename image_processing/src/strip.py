import re
with open('color_spaces_visualization.html', encoding='utf-8') as f:
    html = f.read()

res = re.sub(r'src="data:image/png;base64,[^"]+"', 'src="data:image/png;base64,..."', html)

with open('temp.html', 'w', encoding='utf-8') as f:
    f.write(res)
print("done")
