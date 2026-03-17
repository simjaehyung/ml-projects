import os
import base64
import re

def process():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images = ['실험용사진1.jpeg', '실험용사진2.PNG', '실험용사진3.jpg', '실험용사진4.jpg', '실험용사진5.png']
    
    # We will use the original "분할_1번주제.html" which has the full A/B comparison logic
    html_template_path = os.path.join(base_dir, 'presentations', '분할_1번주제.html')
    
    with open(html_template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    # Process and embed each image
    for img_name in images:
        img_path = os.path.join(base_dir, 'assets', img_name)
        if not os.path.exists(img_path):
            print(f"Skipping {img_name}, not found.")
            continue
            
        with open(img_path, 'rb') as f:
            # We don't know the exact extension, but browsers handle data:image/jpeg fine for PNGs too, 
            # or we can be precise:
            ext = os.path.splitext(img_name)[1].lower()
            mime = 'image/png' if ext == '.png' else 'image/jpeg'
            
            b64_data = f"data:{mime};base64," + base64.b64encode(f.read()).decode('utf-8')
            
        # The JavaScript logic uses <img id="shared-source" src="...">
        # We replace any src inside this specific img tag
        new_html = re.sub(
            r'(<img\s+id="shared-source"\s*?\n?\s*?src=")[^"]*(")',
            r'\g<1>' + b64_data + r'\g<2>',
            html_template
        )
        
        # Save output
        out_name = f"{os.path.splitext(img_name)[0]}_발표.html"
        out_path = os.path.join(base_dir, 'presentations', out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_html)
            
        print(f"Generated {out_name} with comparison for original, Method A, and Method B.")

if __name__ == '__main__':
    process()
