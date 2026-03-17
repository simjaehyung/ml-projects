import base64

with open('jean7.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

with open('color_spaces_visualization.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace the specific <img> tag src
target_str = 'src="jean7.png"'
replace_str = f'src="data:image/png;base64,{img_b64}"'
html = html.replace(target_str, replace_str)

with open('color_spaces_visualization.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Base64 injection complete.")
