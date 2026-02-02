import os
import easyocr

model_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")

# instantiate reader using local model directory and disable automatic download
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=model_dir, download_enabled=False)

# example: read from a crop (numpy array) or file path
results = reader.readtext(r"C:\Users\tsvaevq\Downloads\namePlate.jpg")  # or pass numpy array

# results format: [ (bbox, text, confidence), ... ]
if results:
    best = max(results, key=lambda r: r[2])   # take highest-confidence text
    nameplate_text = best[1]
    conf = best[2]
    print("Nameplate OCR:", nameplate_text, conf)
else:
    print("No OCR text found")
