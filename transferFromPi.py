import os
import shutil
from time import strftime, gmtime

local_dir = os.path.expanduser('~/Desktop/presloh_camera/photos/')
organized_dir = os.path.expanduser('~/Desktop/presloh_camera/organized_photos/')

# Ensure the directories exist
os.makedirs(local_dir, exist_ok=True)
os.makedirs(organized_dir, exist_ok=True)

def organize_images():
    while True:
        for image in os.listdir(local_dir):
            if image.endswith('.jpg'):
                timestamp = int(image.split('_')[1].split('.')[0])
                date_str = strftime('%Y-%m-%d', gmtime(timestamp / 1000.0))
                date_dir = os.path.join(organized_dir, date_str)
                os.makedirs(date_dir, exist_ok=True)
                shutil.move(os.path.join(local_dir, image), os.path.join(date_dir, image))
        time.sleep(60)

if __name__ == '__main__':
    organize_thread = Thread(target=organize_images)
    organize_thread.start()
