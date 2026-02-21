from PIL import Image
import sys

if len(sys.argv) != 3:
    print("Usage: python jpg_to_png_alpha.py input.jpg output.png")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Open the JPEG image
img = Image.open(input_path).convert("RGBA")

# Set alpha channel to fully opaque
r, g, b, a = img.split()
alpha = Image.new("L", img.size, 255)  # 255 = fully opaque
good_img = Image.merge("RGBA", (r, g, b, alpha))

good_img.save(output_path)
print(f"Saved PNG with alpha channel to {output_path}")
