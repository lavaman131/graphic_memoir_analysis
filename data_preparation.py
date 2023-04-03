from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

DATA_PATH = Path("/data/wr153")

if not DATA_PATH.joinpath("images").exists():
    DATA_PATH.joinpath("images").mkdir()

all_pages = convert_from_path(DATA_PATH / "the_complete_persepolis.pdf")

indices = [
    6,
    14,
    23,
    32,
    40,
    48,
    56,
    64,
    73,
    84,
    93,
    101,
    109,
    119,
    128,
    136,
    145,
    155,
    164,
    176,
    186,
    196,
    204,
    214,
    224,
    234,
    251,
    262,
    276,
    289,
    299,
    309,
    319,
    327,
    335,
    349,
    358,
    367,
    381,
]

chapters = []
for i in range(0, len(indices) - 1):
    chapter = {}
    for p in range(indices[i], indices[i + 1]):
        chapter[str(p + 1)] = all_pages[p]
    chapters.append(chapter)


for idx, chapter in tqdm(enumerate(chapters), total=len(chapters)):
    save_dir = DATA_PATH.joinpath("images", f"chapter_{str(idx + 1)}")
    save_dir.mkdir(exist_ok=True)
    for page_num, page in chapter.items():
        page.save(save_dir / f"{page_num}.jpg", "JPEG")
