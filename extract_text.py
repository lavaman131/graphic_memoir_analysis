from typing import List, Dict
import re
from google.cloud import vision_v1
import io
from pathlib import Path
import json
from tqdm import tqdm
import math


DATA_PATH = Path("/data/wr153")
SAVE_PATH = DATA_PATH.joinpath("text")
LIMIT = 16

if not SAVE_PATH.exists():
    SAVE_PATH.mkdir()


def process_text(raw_text: str) -> str:
    text = raw_text
    text = re.sub("-\n|\n|\*", " ", text)
    text = text.replace("MARTANE", "MARJANE")
    text = re.sub(r"\b\w*[a-z]+\w*\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


def detect_text(data_path: Path, save_path: Path) -> None:
    """Detects text in the file."""

    client = vision_v1.ImageAnnotatorClient()

    # Create a list to hold the image content
    image_list = []
    image_names = []

    # Load the images into memory and append them to the list
    for im in data_path.iterdir():
        if im.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            with io.open(im, "rb") as image_file:
                content = image_file.read()
                # Create a list of `Image` objects
                image = vision_v1.types.Image(content=content)
                image_list.append(image)
                image_names.append(im.stem)
        
    all_content = {}
    for i in range(math.ceil(len(image_list) / LIMIT)):
        batch_image_list = image_list[i*LIMIT:(i+1)*LIMIT]
        # Create a `BatchAnnotateImagesRequest` object
        batch_request = vision_v1.types.BatchAnnotateImagesRequest(
            requests=[
                vision_v1.types.AnnotateImageRequest(
                    image=image,
                    image_context={"language_hints": ["en"]},
                    features=[{"type_": vision_v1.Feature.Type.TEXT_DETECTION}],
                )
                for image in batch_image_list
            ]
        )

        # Call the `batch_annotate_images()` method to perform OCR on the images
        response = client.batch_annotate_images(request=batch_request)

        # Extract the text from the response object
        content = {}
        for im_names, image_response in zip(image_names[i*LIMIT:(i+1)*LIMIT], response.responses):
            raw_text = image_response.full_text_annotation.text
            content[im_names] = process_text(raw_text)
        all_content.update(content)
    try:
        with open(save_path.joinpath(f"{data_path.name}.json"), "w") as fp:
            json.dump(all_content, fp, indent=4)
            # print("Success")
    except:
        print("Failed")
    


for i in tqdm(range(1, 39)):
    detect_text(DATA_PATH.joinpath("images", f"chapter_{i}"), SAVE_PATH)
