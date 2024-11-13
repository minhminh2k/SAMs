import os
import wget
import zipfile

# Tạo thư mục coco và thư mục images bên trong
os.makedirs("data/", exist_ok=True)
os.makedirs("data/images", exist_ok=True)


# Danh sách URL tải dữ liệu hình ảnh
image_urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    # "http://images.cocodataset.org/zips/unlabeled2017.zip"
]

for url in image_urls:
    filename = url.split("/")[-1]
    filepath = os.path.join("data/images", filename)
    
    print(f"Downloading {filename}...")
    wget.download(url, filepath)
    print(f"\nDownloaded {filename}")

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall("data/images")
    os.remove(filepath)
    print(f"Extracted and removed {filename}")

# Annotations
annotation_urls = [
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/image_info_test2017.zip",
    # "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
]

for url in annotation_urls:
    filename = url.split("/")[-1]
    filepath = os.path.join("data", filename)
    
    print(f"Downloading {filename}...")
    wget.download(url, filepath)
    print(f"\nDownloaded {filename}")

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall("data")
    os.remove(filepath)
    print(f"Extracted and removed {filename}")

print("COCO dataset downloaded and organized successfully.")