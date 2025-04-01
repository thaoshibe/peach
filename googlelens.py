import asyncio
import logging  # For setting log level example

import numpy as np
from chrome_lens_py import LensAPI
from PIL import Image


async def run_lens_tasks():
    # Initialize API (consider proxy, cookies, logging level here)
    # Example: Enable debug logging and set a proxy
    api = LensAPI(
        config={'proxy': 'socks5://127.0.0.1:9050',
        'debug_out': 'debug_response.json',
        'cookies': 'google_cookies.txt'},
        logging_level=logging.DEBUG
    )

    image_path = './break-a-scene/examples/bear/img.jpg' # Your image path
    # image_url = 'https://www.google.com/images/branding/googlelogo/1x/googlelogo_light_color_272x92dp.png' # Example URL

    try:
        # --- Test 1: Get all data from local file ---
        print("\n--- Testing get_all_data (local file) ---")
        result_all_file = await api.get_all_data(image_path, coordinate_format='pixels')
        print(result_all_file)

        # --- Test 2: Get full text from URL ---
        # print("\n--- Testing get_full_text (URL) ---")
        # # Corresponds to full_text_default in CLI
        # result_text_url = await api.get_full_text(image_url)
        # print(result_text_url)

        # --- Test 3: Get coordinates from PIL Image ---
        print("\n--- Testing get_text_with_coordinates (PIL Image) ---")
        try:
            pil_image = Image.open(image_path)
            result_coords_pil = await api.get_text_with_coordinates(pil_image, coordinate_format='percent')
            print(result_coords_pil)
            pil_image.close()
        except FileNotFoundError:
            print(f"PIL Test skipped: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error processing PIL image: {e}")

        # --- Test 4: Get smart stitched text from NumPy array ---
        print("\n--- Testing get_stitched_text_smart (NumPy array) ---")
        # Corresponds to full_text_new_method in CLI
        try:
            np_image = np.array(Image.open(image_path)) # Load image into numpy array
            result_smart_np = await api.get_stitched_text_smart(np_image)
            print(result_smart_np)
        except FileNotFoundError:
            print(f"NumPy Test skipped: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error processing NumPy array: {e}")

        # --- Test 5: Get sequential stitched text from local file ---
        print("\n--- Testing get_stitched_text_sequential (local file) ---")
        # Corresponds to full_text_old_method in CLI
        result_seq_file = await api.get_stitched_text_sequential(image_path)
        print(result_seq_file)

    except Exception as e:
        print(f"\n--- An error occurred during testing: {e} ---")
        logging.exception("Error details:") # Log traceback if logging is enabled
    finally:
        # --- IMPORTANT: Close the session when done ---
        print("\n--- Closing API session ---")
        await api.close_session()

if __name__ == "__main__":
    # Basic logging setup for the test script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    asyncio.run(run_lens_tasks())
