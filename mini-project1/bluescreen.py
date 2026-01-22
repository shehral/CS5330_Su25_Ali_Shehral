"""
File: bluescreen.py
--------------------
This program shows an example of "greenscreening" (actually
"bluescreening" in this case).  This is where we replace the
pixels of a certain color intensity in a particular channel
(here, we use blue) with the pixels from another image.
"""


import os
from simpleimage import SimpleImage

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INTENSITY_THRESHOLD = 1.2  # Lower = more pixels detected as "blue"


def bluescreen(main_filename, back_filename):
    """
    Implements the notion of "bluescreening".  That is,
    the image in the main_filename has its "sufficiently blue"
    pixels replaced with pixel from the corresponding x,y
    location in the image in the file back_filename.
    Returns the resulting "bluescreened" image.
    """
    image = SimpleImage(main_filename)
    back = SimpleImage(back_filename)

    for pixel in image:
        # Calculate average of RGB values
        average = (pixel.red + pixel.green + pixel.blue) / 3
        # Check if pixel is "sufficiently blue"
        # (blue channel is significantly higher than average)
        if pixel.blue > average * INTENSITY_THRESHOLD:
            # Get corresponding pixel from background image
            # Use modulo to handle different image sizes
            back_x = pixel.x % back.width
            back_y = pixel.y % back.height
            back_pixel = back.get_pixel(back_x, back_y)
            # Replace main image pixel with background pixel
            pixel.red = back_pixel.red
            pixel.green = back_pixel.green
            pixel.blue = back_pixel.blue
    return image


def main():
    """
    Run your desired image manipulation functions here.
    You should store the return value (image) and then
    call .show() to visualize the output of your program.
    """
    # Build paths to images
    musk_path = os.path.join(SCRIPT_DIR, 'images', 'musk.jpg')
    flower_path = os.path.join(SCRIPT_DIR, 'images', 'flower.png')

    # Show the original musk image
    original = SimpleImage(musk_path)
    original.show()

    # Demonstrate bluescreen effect
    # Blue background behind Musk will be replaced with flower.png
    result = bluescreen(musk_path, flower_path)
    result.show()


if __name__ == '__main__':
    main()
