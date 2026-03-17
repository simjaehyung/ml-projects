import base64
import os
import re

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images = {
        'orig': os.path.join(base_dir, 'assets', 'test_image_original.jpg'),
        'cv2': os.path.join(base_dir, 'assets', 'test_image_gray_cv2.jpg'),
        'mean': os.path.join(base_dir, 'assets', 'test_image_gray_mean.jpg')
    }

    b64_data = {}
    for key, path in images.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                b64_data[key] = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')
        else:
            print(f'Missing: {path}')
            return

    html_path = os.path.join(base_dir, 'presentations', '분할_1번주제.html')
    if not os.path.exists(html_path):
        print("HTML file missing")
        return
        
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # The HTML has these IDs for img:
    # id="shared-source" src="..."
    # id="img-result-a" src="..."
    # id="img-result-b" src="..."
    
    html_content = re.sub(
        r'(<img\s+id="shared-source"\s+src=")[^"]*(")',
        r'\g<1>' + b64_data['orig'] + r'\g<2>',
        html_content
    )
    
    html_content = re.sub(
        r'(<img\s+id="img-result-a"\s+src=")[^"]*(")',
        r'\g<1>' + b64_data['cv2'] + r'\g<2>',
        html_content
    )
    
    html_content = re.sub(
        r'(<img\s+id="img-result-b"\s+src=")[^"]*(")',
        r'\g<1>' + b64_data['mean'] + r'\g<2>',
        html_content
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print("Injection successful!")

if __name__ == '__main__':
    main()
