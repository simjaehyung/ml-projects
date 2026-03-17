import re

with open('c:/Users/jhsim/Erica261/머신러닝/presentation_visualizer.html', 'r', encoding='utf-8') as f:
    text = f.read()

m = re.search(r'<img id=\"shared-source\"\s*src=\"(data:image/[a-zA-Z0-9]+;base64,[^\"]+)\"', text)
if m:
    jean12_b64 = m.group(1)
    print('Found base64 for jean12, length:', len(jean12_b64))
    
    # Replace in presentation_lighting.html
    with open('c:/Users/jhsim/Erica261/머신러닝/presentation_lighting.html', 'r', encoding='utf-8') as f2:
        text2 = f2.read()
    
    new_text2 = re.sub(r'<img id=\"jean-source\" src=\"data:image/[^\"]+\"', f'<img id=\"jean-source\" src=\"{jean12_b64}\"', text2)
    
    with open('c:/Users/jhsim/Erica261/머신러닝/presentation_lighting.html', 'w', encoding='utf-8') as f3:
        f3.write(new_text2)
    print('Base64 replaced successfully')
else:
    print('Base64 not found in presentation_visualizer.html')
