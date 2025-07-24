import pdfplumber
import fitz
import os
import io
import argparse
import re
from PIL import Image

# Default paths and parameters
DEFAULT_INPUT_FILE = '/media/chirurgie/hdd01/Soft/GitHub/Paper2Code/examples/segar/segar_paper.pdf'
DEFAULT_OUTPUT_DIR = '/media/chirurgie/hdd01/Soft/GitHub/Paper2Code/examples/segar/Images'
DEFAULT_IMG_FORMAT = 'jpg'
DEFAULT_IMG_QUALITY = 80

# --- аргументы ---
parser = argparse.ArgumentParser(description="PDF Image Extractor: извлекает и сохраняет картинки из PDF, печатает текст каждой страницы.")
parser.add_argument("input_file", nargs='?', default=DEFAULT_INPUT_FILE, help="PDF файл для разбора")
parser.add_argument("output_dir", nargs='?', default=DEFAULT_OUTPUT_DIR, help="куда сохранять картинки")
parser.add_argument("img_format", nargs='?', default=DEFAULT_IMG_FORMAT, help="формат (jpg/png)")
parser.add_argument("img_quality", nargs='?', type=int, default=DEFAULT_IMG_QUALITY, help="качество (0-100)")
args = parser.parse_args()

PDF_PATH = args.input_file
OUTPUT_IMAGES_DIR = args.output_dir
fmt = args.img_format.lower()
if fmt in ('jpg', 'jpeg'):
    IMAGE_FORMAT = 'JPEG'
elif fmt == 'png':
    IMAGE_FORMAT = 'PNG'
else:
    IMAGE_FORMAT = args.img_format.upper()
IMAGE_QUALITY = args.img_quality

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def remove_letters(text):
    return ''.join([c for c in text if c and not c.isalpha()])

def sanitize_filename(s, max_len=50):
    """Sanitize string to safe filename."""
    # replace non-alphanumeric chars with underscore
    s = re.sub(r'[^0-9A-Za-z]+', '_', s)
    s = s.strip('_')
    return s[:max_len]

def resize_image(image):
    max_size = int(max(image.width, image.height) * 0.7)
    if image.width > max_size or image.height > max_size:
        if image.width > image.height:
            aspect_ratio = image.height / image.width
            new_width = max_size
            new_height = int(max_size * aspect_ratio)
        else:
            aspect_ratio = image.width / image.height
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
        image = image.resize((new_width, new_height), Image.BICUBIC)
    return image

def save_images_from_page(document, page_number, product_reference):
    saved_images = []
    pagina = document.load_page(page_number)
    imagens = pagina.get_images(full=True)
    for img_index, img in enumerate(imagens):
        xref = img[0]
        base_image = document.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        if image.width < 500 or image.height < 500:
            continue
        image = resize_image(image)
        safe_ref = sanitize_filename(product_reference)
        image_filename = os.path.join(
            OUTPUT_IMAGES_DIR, f"{safe_ref}_{page_number+1}_{img_index}.{IMAGE_FORMAT.lower()}")
        image.save(image_filename, IMAGE_FORMAT, quality=IMAGE_QUALITY)
        saved_images.append(image_filename)
    return saved_images

document = fitz.open(PDF_PATH)
with pdfplumber.open(PDF_PATH) as pdf:
    pages = pdf.pages
    for index, pagina in enumerate(pages):
        texto = pagina.extract_text() or f"page_{index+1}"
        print(f"Text on Page {index+1}:")
        print(texto.strip() if texto else "<NO TEXT>")
        print("-"*50)
        text_for_name = remove_letters(texto) or f"page_{index+1}"
        imagens = save_images_from_page(document, index, text_for_name)
        for img in imagens:
            print(f"Image Saved: {img}")
        print("="*50)
document.close()
