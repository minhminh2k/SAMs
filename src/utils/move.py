import os
import shutil

source_folder = "out/training"

destination_folder = "images/0411"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(destination_folder, exist_ok=True)

# Duyệt qua tất cả các file trong thư mục nguồn
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg"):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        shutil.move(source_file, destination_file)
        print(f"Đã chuyển: {filename}")

print("Di chuyển hoàn tất!")
