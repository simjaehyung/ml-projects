import re

with open('c:/Users/jhsim/Erica261/머신러닝/color_spaces_visualization.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract the block
m = re.search(r'(<div class="plot-box full-width-box" style="margin-bottom: 20px;">\s*<h2>Lighting Robustness: Diagonal vs\. Axial Scaling</h2>.*?</div>\s*</div>\s*</div>)', text, re.DOTALL)
if m:
    block = m.group(1)
    with open('c:/Users/jhsim/Erica261/머신러닝/presentation_lighting.html', 'r', encoding='utf-8') as f2:
        text2 = f2.read()
    
    new_text2 = re.sub(r'<div class="container" id="lighting-robustness-section" style="padding-top:0; display:none;">\s*</div>', f'<div class="container" id="lighting-robustness-section" style="padding-top:0; display:none;">\n{block}\n</div>', text2)
    
    with open('c:/Users/jhsim/Erica261/머신러닝/presentation_lighting.html', 'w', encoding='utf-8') as f3:
        f3.write(new_text2)
    print('Block inserted')
else:
    print('Block not found')
