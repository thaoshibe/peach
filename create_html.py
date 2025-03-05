import glob
import os

import pandas as pd

# SUBJECT_NAMES = ["tasha", "dog8", "Oasis", "Nozis", "Lindsay",
#                 "Yana", "Ocre", "omi-babi", "sparky", "Diva-4"]

# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Display</title>
    <style>
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
        }}
        th, td {{ 
            border: 1px solid black; 
            padding: 8px; 
            text-align: center; 
        }}
        .image-container {{ 
            display: flex; 
            flex-wrap: wrap; 
            justify-content: center; 
        }}
        .image-container img {{ 
            max-width: 100px; 
            max-height: 100px; 
            margin: 5px; 
        }}
    </style>
</head>
<body>
    <table>
            <tr>
                <td></td>
                <td colspan="4"><span style="font-size: 72px;">Purely Reference</span>
                                <br> <span style="font-size: 30px;">
                                This photo(s) of Max [image1]...[imagek]. Another photo of Max: ???</span>
                                </td>
                <td colspan="4"><span style="font-size:72px"> In-Context Learning </span>
                                <br> <span style="font-size: 30px;">
                                This photo(s) of Max [image1]. This is another photo of Max: [image2]
                                <br> This photo(s) of Max [image3]. This is another photo of Max: [image4]
                                <br> ...
                                <br> This photo(s) of Max [imagek]. This is another photo of Max: ???</span>
                                </th>
            </tr>
            <tr>
                <td>Dog Name</th>
                <td><b> Purely Reference (3) </b></td>
                <td><b> Purely Reference (5) </b></td>
                <td><b> Purely Reference (7) </b></td>
                <td><b> Purely Reference (9) </b></td>
                <td><b> In-Context (1) </b></td>
                <td><b> In-Context (2) </b></td>
                <td><b> In-Context (3) </b></td>
                <td><b> In-Context (4) </b></td>
            </tr>
"""

  # Open the row once
SUBJECT_NAMES = os.listdir('./generated_images/icl_indomain/')
for i, dog_name in enumerate(SUBJECT_NAMES):
    html_content += "<tr style='background-color: lightgray;'><td></td><td colspan=8>"
    true_image_path = os.path.join('./test/', dog_name)
    true_image = sorted(glob.glob(os.path.join(true_image_path, '*.jpg'))[:9])
    
    for index, img in enumerate(true_image):
        html_content += f"Image {index}: <img src='{img}' height='100px' alt='{img}'>"

    html_content += "</td></tr>"  # Close the row once

    # html_content += f"<td>{dog_name}</td>"
    html_content +=f'''
                <tr>
                <td>{dog_name}</td>
                <td><span style="font-size: 30px;"> Purely Reference (3) </span></td>
                <td><span style="font-size: 30px;"> Purely Reference (5) </span></td>
                <td><span style="font-size: 30px;"> Purely Reference (7) </span></td>
                <td><span style="font-size: 30px;"> Purely Reference (9) </span></td>
                <td><span style="font-size: 30px;"> In-Context (1) </span></td>
                <td><span style="font-size: 30px;"> In-Context (2) </span></td>
                <td><span style="font-size: 30px;"> In-Context (3) </span></td>
                <td><span style="font-size: 30px;"> In-Context (4) </span></td>
                </tr>
    '''
    
    # Add purely reference images
    html_content += f"<tr><td></td>"
    purely_reference_path = os.path.join('./generated_images/purely_reference', dog_name)
    for num_of_images in ['3', '5', '7', '9']:
        gen_images = glob.glob(os.path.join(purely_reference_path, num_of_images, '*.png'))
        html_content += "<td>"
        for img in sorted(gen_images):
            html_content += f"<img src='{img}' height='200px' alt='{img}'>"
        html_content += "</div></td>"
        # html_content += f"<img src='{img}' alt='{img}'>"
    # html_content += "</tr>"

    # Add purely reference images
    ict_path = os.path.join('./generated_images/icl_indomain', dog_name)
    for num_of_images in ['1', '2', '3', '4']:
        gen_images = glob.glob(os.path.join(ict_path, num_of_images, '*.png'))
        html_content += "<td>"
        for img in sorted(gen_images):
            html_content += f"<img src='{img}' height='200px' alt='{img}'>"
        html_content += "</div></td>"
        # html_content += f"<img src='{img}' alt='{img}'>"
    html_content += "</tr>"

html_content += """
    </table>
</body>
</html>
"""

# Write HTML file
with open('image_display.html', 'w') as f:
    f.write(html_content)

print("HTML file generated successfully: image_display.html")
