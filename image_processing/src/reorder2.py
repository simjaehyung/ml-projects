import re

with open('color_spaces_visualization.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Extract the base64 string from the current html to reuse it for Method A/B
# Look for id="jean-source" src="data:image/png;base64,..."
b64_match = re.search(r'id="jean-source" src="(data:image/png;base64,[^"]+)"', html)
if b64_match:
    b64_src = b64_match.group(1)
else:
    print("Could not find base64 image")
    exit()

# We need to change the Grayscale Conversion section.
# Remove the <input type="file"> and button, replace with fixed logic on load.
# Let's find the grayscale section string and replace it.

grayscale_section_start = html.find('<div class="container" id="grayscale-section">')
lighting_robustness_start = html.find('<div class="container" id="lighting-robustness-section" style="padding-top:0;">')

if grayscale_section_start == -1 or lighting_robustness_start == -1:
    print("Could not find sections")
    exit()

grayscale_html = html[grayscale_section_start:lighting_robustness_start]

# Modify grayscale HTML to remove upload button and hardcode the image source
new_grayscale_html = grayscale_html.replace(
    'Upload any image to see how it converts to black and white using two different mathematical approaches.',
    'Test how the current image converts to black and white using two different mathematical approaches.'
)

controls_start = new_grayscale_html.find('<div class="controls">')
controls_end = new_grayscale_html.find('</div>', controls_start) + 6
# Remove upload controls
new_grayscale_html = new_grayscale_html[:controls_start] + f"""
            <div style="display:none;">
                <img id="shared-source" src="{b64_src}">
            </div>
""" + new_grayscale_html[controls_end:]

# Now we need to modify the JS for Grayscale Conversion
js_start = html.find('        // --- 1. Grayscale Conversion Logic (2D UI) ---')
js_end = html.find('        // --- 2. 3D Plotly Logic ---')
js_logic = html[js_start:js_end]

new_js_logic = """        // --- 1. Grayscale Conversion Logic (Fixed Image) ---
        const sharedImg = document.getElementById('shared-source');
        const origCanvas = document.getElementById('original-canvas');
        const origCtx = origCanvas.getContext('2d');
        const aCanvas = document.getElementById('method-a-canvas');
        const aCtx = aCanvas.getContext('2d');
        const bCanvas = document.getElementById('method-b-canvas');
        const bCtx = bCanvas.getContext('2d');

        function processMethodAB() {
            // Hide placeholder text
            document.getElementById('original-text').style.display = 'none';
            document.getElementById('method-a-text').style.display = 'none';
            document.getElementById('method-b-text').style.display = 'none';

            // Set canvas sizes
            origCanvas.width = sharedImg.naturalWidth;
            origCanvas.height = sharedImg.naturalHeight;
            aCanvas.width = sharedImg.naturalWidth;
            aCanvas.height = sharedImg.naturalHeight;
            bCanvas.width = sharedImg.naturalWidth;
            bCanvas.height = sharedImg.naturalHeight;

            // Draw original
            origCtx.drawImage(sharedImg, 0, 0);

            // Get image data
            const imageData = origCtx.getImageData(0, 0, sharedImg.naturalWidth, sharedImg.naturalHeight);
            const data = imageData.data;

            // Create data for Method A and Method B
            const aData = new ImageData(sharedImg.naturalWidth, sharedImg.naturalHeight);
            const bData = new ImageData(sharedImg.naturalWidth, sharedImg.naturalHeight);

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];

                const grayA = (0.299 * r) + (0.587 * g) + (0.114 * b);
                aData.data[i] = grayA; aData.data[i + 1] = grayA; aData.data[i + 2] = grayA; aData.data[i + 3] = a;

                const grayB = (r + g + b) / 3;
                bData.data[i] = grayB; bData.data[i + 1] = grayB; bData.data[i + 2] = grayB; bData.data[i + 3] = a;
            }

            aCtx.putImageData(aData, 0, 0);
            bCtx.putImageData(bData, 0, 0);
        }

        if (sharedImg.complete && sharedImg.naturalHeight !== 0) {
            processMethodAB();
        } else {
            sharedImg.onload = processMethodAB;
        }

"""
# Now we need to reorder the sections
# Current order:
# <h2 Basic Color Spaces>
# <div plots-container>
# <h2 Color Transformations>
# <div grayscale-section>
# <div lighting-robustness-section>

# New Order:
# <h2 Color Transformations>
# <div grayscale-section>
# <div lighting-robustness-section> (incl 3D RGB map)
# <h2 Basic Color Spaces>
# <div plots-container>

plots_h2_start = html.find('<h2 style="text-align:center; color: #fff; margin-top: 50px;">1. Basic Color Spaces (3D)</h2>')
plots_cont_end = html.find('<h2 style="text-align:center; color: #fff; margin-top: 50px;">2. Color Transformations & Robustness</h2>')

trans_h2_start = html.find('<h2 style="text-align:center; color: #fff; margin-top: 50px;">2. Color Transformations & Robustness</h2>')
script_start = html.find('<script>')

header_end = html[:plots_h2_start]

# Extract lighting robustness (which also contains the 3D RGB interactive map at the end of it... wait. No!)
# Let's check where the 3D RGB Mapping is.
# In the temp file:
# lines 322-358: plots-container (RGB Mapping, HSV, YCbCr, Lab) -> wait, RGB mapping is inside plots-container!
# The user wants "method a,b 비교를 이미지 변환 아래에다가 해주고 중간에 있는 RGB HSV 를 맨 아래에다가 해줄래"
# - Method A,B comparison goes below "image transformation". What is "image transformation"? 
# - Actually, the user says "이미지 업로드 하는거 그냥 jean7 으로 이미지 고정하고 처음에 있는 메소드 a,b 방식 비교를 이미지 변환 아래에다가 해주고"
#   Wait, "처음에 있는 메소드 a,b 방식 비교" = "Method A,B comparison which is at the top". (Before my previous reordering, it WAS at the top).
#   "이미지 변환 아래에다가 해주고" = "Put it below the image transformation". Does "image transformation" refer to the new Lighting Robustness? Yes, Lighting Robustness scales brightness.
# - "중간에 있는 RGB HSV 를 맨 아래에다가 해줄래?" = "Put the RGB/HSV things that are in the middle to the very bottom?" -> The Basic Color spaces.
# So the order:
# 1. Lighting Robustness ("Image Transformation")
# 2. Method A/B Grayscale Comparison
# 3. 3D RGB Space Color Mapping (the interactive one)
# 4. Basic Color Spaces (RGB, HSV, YCbCr, Lab)

# Let's extract the components explicitly:
plots_container = html[html.find('<div class="container" id="plots-container"'):html.find('<h2 style="text-align:center; color: #fff; margin-top: 50px;">2. Color Transformations & Robustness</h2>')]

# Strip the "3D RGB Space Color Mapping" out of plots_container
rgb_map_start = plots_container.find('<div class="plot-box full-width-box" style="margin-bottom: 20px;">')
hsv_box_start = plots_container.find('<div class="plot-box">\n            <h2>HSV Space</h2>')
rgb_map_html = plots_container[rgb_map_start:hsv_box_start]
basic_plots_html = '<div class="container" id="plots-container" style="display:none;">\n' + plots_container[hsv_box_start:]

lighting_sec_start = html.find('<div class="container" id="lighting-robustness-section"')
lighting_sec_end = html.find('<script>')
lighting_html = html[lighting_sec_start:lighting_sec_end]

final_html = header_end + \
    '<h2 style="text-align:center; color: #fff; margin-top: 50px;">1. Color Transformations (Lighting & Grayscale)</h2>\n' + \
    lighting_html + '\n' + \
    new_grayscale_html + '\n' + \
    '<div class="container">\n' + rgb_map_html + '</div>\n' + \
    '<h2 style="text-align:center; color: #fff; margin-top: 50px;">2. Basic Color Spaces (3D)</h2>\n' + \
    basic_plots_html + '\n' + \
    '<script>\n' + \
    new_js_logic + html[js_end:]

with open('color_spaces_visualization.html', 'w', encoding='utf-8') as f:
    f.write(final_html)

print("success")
