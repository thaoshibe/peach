import glob
import os


def generate_image_gallery(base_directory, max_images_per_cell=100):
    """
    Generate an HTML gallery from image directories
    
    Args:
    base_directory (str): Root directory containing setting folders
    max_images_per_cell (int): Maximum number of images to display in each cell
    
    Returns:
    str: HTML content of the gallery
    """
    # Get all setting directories
    settings = [
        'willinvietnam',
        'mam',
        'ciin',
        'ciin-without-sks',
        'willinvietnam-without-sks', 
        'mam-without-sks',
        'willinvietnam-without-sks-reverse',
        'ciin-without-sks-reverse', 
        'mam-without-sks-reverse',
    ]
    
    # Define token counts (based on second image)
    token_counts = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Create token headers
    token_headers = ''.join(f'<th>{token}</th>' for token in token_counts)
    
    # Create table rows
    table_rows = []
    for setting in settings:
        row_cells = [f'<td>{setting}</td>']
        
        for token in token_counts:
            # Construct path pattern to find images
            search_pattern = os.path.join(base_directory, setting, str(token), '*')
            matching_images = glob.glob(search_pattern)
            
            if matching_images:
                # Limit the number of images
                limited_images = matching_images[:max_images_per_cell]
                
                # Create image container with multiple images
                image_html = '<td><div class="image-container">'
                for image_path in limited_images:
                    relative_path = os.path.relpath(image_path, base_directory)
                    image_html += f'<img src="{relative_path}" alt="{setting} - {token}">'
                
                # Add count of total images if more than max_images_per_cell
                if len(matching_images) > max_images_per_cell:
                    image_html += f'<div style="width:100%;text-align:center;">+{len(matching_images) - max_images_per_cell} more</div>'
                
                image_html += '</div></td>'
                row_cells.append(image_html)
            else:
                row_cells.append('<td>No image</td>')
        
        table_rows.append(f'<tr>{"".join(row_cells)}</tr>')
    
    # Start HTML content with improved styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin-top: 20px; 
            }}
            th, td {{ 
                border: 2px solid #333; 
                padding: 10px; 
                text-align: center; 
            }}
            th {{ 
                background-color: #f2f2f2; 
            }}
            .image-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 5px;
            }}
            .image-container img {{
                max-height: 50px;
                max-width: 50px;
                object-fit: cover;
            }}
        </style>
    </head>
    <body>
        <h1>Image Gallery</h1>
        <table>
            <tr>
                <th>Setting / Tokens</th>
                {token_headers}
            </tr>
            {''.join(table_rows)}
        </table>
    </body>
    </html>
    """
    
    return html_content

def save_gallery_html(html_content, output_path='gallery.html'):
    """
    Save HTML gallery to a file
    
    Args:
    html_content (str): HTML content to save
    output_path (str): Path to save the HTML file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Gallery HTML saved to {output_path}")

# Example usage
base_dir = '/Users/thaoshibe/Documents/result/0326/'  # Current directory, adjust as needed
gallery_html = generate_image_gallery(base_dir, max_images_per_cell=20)
save_gallery_html(gallery_html, output_path=os.path.join(base_dir, 'gallery.html'))
