import re

with open('color_spaces_visualization.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. We need to restructure the HTML body logic.
# Let's find the sections using unique markers.

header_end_idx = html.find('<div class="container" id="grayscale-section">')
if header_end_idx == -1:
    print("Could not find grayscale-section")
    exit()

grayscale_sec_start = header_end_idx
grayscale_sec_end = html.find('<div class="container" id="plots-container"')
grayscale_section_html = html[grayscale_sec_start:grayscale_sec_end]

plots_sec_start = grayscale_sec_end
script_tag_idx = html.find('<script>')
plots_section_html = html[plots_sec_start:script_tag_idx]

# Inside plots_section_html, we have:
# a) Lighting Robustness Experiment
# b) 3D RGB Space Color Mapping
# c) HSV Space
# d) YCbCr Space

robust_end_idx = plots_section_html.find('<div class="plot-box full-width-box" style="margin-bottom: 20px;">\n            <h2>3D RGB Space Color Mapping</h2>')
robustness_section_html = plots_section_html[:robust_end_idx].replace('<div class="container" id="plots-container" style="display:none; padding-top:0;">', '')

# Remove closing tag of the container from the end of plots_section_html
# we can find the last '</div>' before script.
last_div_idx = plots_section_html.rfind('</div>')
color_spaces_html = plots_section_html[robust_end_idx:last_div_idx].strip()

# Now we need to append the Lab space to color_spaces_html
lab_html = """
        <div class="plot-box">
            <h2>CIELAB (Lab) Space</h2>
            <div id="lab-plot" class="plot"></div>
            <p style="text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-top: 15px;">
                CIE $L^*a^*b^*$ 공간은 인간의 시각 인지와 유사한 균일 색공간(Perceptually Uniform Space)입니다. 밝기($L^*$)와 두 가지 색상 축($a^*$, $b^*$)으로 구성됩니다.
            </p>
        </div>
"""
color_spaces_html += lab_html


# Now reconstruct the body:
new_body = html[:header_end_idx] + \
    '<h2 style="text-align:center; color: #fff; margin-top: 50px;">1. Basic Color Spaces (3D)</h2>\n' + \
    '<div class="container" id="plots-container" style="display:none;">\n' + \
    color_spaces_html + '\n</div>\n\n' + \
    '<h2 style="text-align:center; color: #fff; margin-top: 50px;">2. Color Transformations & Robustness</h2>\n' + \
    grayscale_section_html + '\n' + \
    '<div class="container" id="lighting-robustness-section" style="padding-top:0;">\n' + \
    robustness_section_html + '\n</div>\n\n' + \
    '<script>' + html[script_tag_idx+8:]


# Now update the JS logic to include Lab
js_rgb_to_lab = """
        // CIELAB conversion
        let lab_x = [], lab_y = [], lab_z = [];
        
        function rgbToXyz(r, g, b) {
            let _r = r / 255.0;
            let _g = g / 255.0;
            let _b = b / 255.0;

            _r = _r > 0.04045 ? Math.pow((_r + 0.055) / 1.055, 2.4) : _r / 12.92;
            _g = _g > 0.04045 ? Math.pow((_g + 0.055) / 1.055, 2.4) : _g / 12.92;
            _b = _b > 0.04045 ? Math.pow((_b + 0.055) / 1.055, 2.4) : _b / 12.92;

            _r *= 100;
            _g *= 100;
            _b *= 100;

            let x = _r * 0.4124564 + _g * 0.3575761 + _b * 0.1804375;
            let y = _r * 0.2126729 + _g * 0.7151522 + _b * 0.0721750;
            let z = _r * 0.0193339 + _g * 0.1191920 + _b * 0.9503041;
            return [x, y, z];
        }

        function xyzToLab(x, y, z) {
            let ref_X = 95.047;
            let ref_Y = 100.000;
            let ref_Z = 108.883;

            x /= ref_X;
            y /= ref_Y;
            z /= ref_Z;

            x = x > 0.008856 ? Math.pow(x, 1/3) : (7.787 * x) + (16 / 116);
            y = y > 0.008856 ? Math.pow(y, 1/3) : (7.787 * y) + (16 / 116);
            z = z > 0.008856 ? Math.pow(z, 1/3) : (7.787 * z) + (16 / 116);

            let l = (116 * y) - 16;
            let a = 500 * (x - y);
            let b = 200 * (y - z);

            return [l, a, b];
        }

        function rgbToLab(r, g, b) {
            let [x, y, z] = rgbToXyz(r, g, b);
            return xyzToLab(x, y, z);
        }
"""

# inject right before `function rgbToHsv(r, g, b) {`
new_body = new_body.replace('function rgbToHsv(r, g, b) {', js_rgb_to_lab + '\n        function rgbToHsv(r, g, b) {')

# Add the calculation in the step loop
# `ycbcr_x.push(cb); ycbcr_y.push(cr); ycbcr_z.push(y);`
ycbcr_idx = new_body.find('ycbcr_x.push(cb); ycbcr_y.push(cr); ycbcr_z.push(y);')
# Inject lab pushing
lab_pushing = """
                        // Lab
                        const [l_val, a_val, b_val] = rgbToLab(r, g, b);
                        lab_x.push(a_val); lab_y.push(b_val); lab_z.push(l_val);
"""
new_body = new_body[:ycbcr_idx] + 'ycbcr_x.push(cb); ycbcr_y.push(cr); ycbcr_z.push(y);\n' + lab_pushing + new_body[ycbcr_idx+len('ycbcr_x.push(cb); ycbcr_y.push(cr); ycbcr_z.push(y);'):]


# Add the layout for Lab and trace plotting
lab_config = """
            // Lab Config
            let layoutLab = JSON.parse(JSON.stringify(layoutBase));
            layoutLab.scene = JSON.parse(JSON.stringify(sceneLayout));
            layoutLab.scene.xaxis.title = 'a* (Green-Red)'; layoutLab.scene.xaxis.range = [-128, 128];
            layoutLab.scene.yaxis.title = 'b* (Blue-Yellow)'; layoutLab.scene.yaxis.range = [-128, 128];
            layoutLab.scene.zaxis.title = 'L* (Lightness)'; layoutLab.scene.zaxis.range = [0, 100];

            const traceLab = { x: lab_x, y: lab_y, z: lab_z, mode: 'markers', marker: commonMarker, type: 'scatter3d', hoverinfo: 'text', text: hoverTexts };
"""
trace_ycbcr_str = "const traceYcbcr = { x: ycbcr_x, y: ycbcr_y, z: ycbcr_z, mode: 'markers', marker: commonMarker, type: 'scatter3d', hoverinfo: 'text', text: hoverTexts };"
new_body = new_body.replace(trace_ycbcr_str, trace_ycbcr_str + '\n' + lab_config)

# Add plotting call
# `Plotly.newPlot('ycbcr-plot', [traceYcbcr], layoutYcbcr, config);`
plotly_call = "Plotly.newPlot('ycbcr-plot', [traceYcbcr], layoutYcbcr, config);"
new_body = new_body.replace(plotly_call, plotly_call + "\n            Plotly.newPlot('lab-plot', [traceLab], layoutLab, config);")


with open('color_spaces_visualization.html', 'w', encoding='utf-8') as f:
    f.write(new_body)

print("HTML Structure completely rewritten")
