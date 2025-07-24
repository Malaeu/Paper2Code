import os
from pdf2image import convert_from_path

PDF_PATH = '/media/chirurgie/hdd01/Soft/GitHub/Paper2Code/examples/segar/segar_paper.pdf'
IMAGES_DIR = '/media/chirurgie/hdd01/Soft/GitHub/Paper2Code/examples/segar/Images'

os.makedirs(IMAGES_DIR, exist_ok=True)

# 1. Конвертируем все страницы pdf в PNG
pages = convert_from_path(PDF_PATH, dpi=300)
for i, page in enumerate(pages):
    img_path = os.path.join(IMAGES_DIR, f'page_{i+1}.png')
    page.save(img_path, 'PNG')
    print(f'[OK] Saved page {i+1} as {img_path}')

print('\nВсе страницы теперь лежат в папке Images и готовы к подаче на Marker, OmniParser или LLM!')

