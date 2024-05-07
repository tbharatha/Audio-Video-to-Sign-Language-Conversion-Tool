import json
import os
import requests

# Function to download a file from a URL
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Load WLASL dataset JSON file
json_path = '/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/archive/WLASL_v0.3.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Folder where videos will be saved
save_dir = '/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/mother_sign_videos'
os.makedirs(save_dir, exist_ok=True)

# Iterate through each entry and download the "mother" sign videos
for entry in data:
    if entry['gloss'] == 'mother':
        for instance in entry['instances']:
            video_url = instance['url']
            video_id = instance['video_id']
            file_ext = os.path.splitext(video_url)[1]
            local_filename = os.path.join(save_dir, f'{video_id}{file_ext}')
            print(f'Downloading video {video_id} from {video_url}')
            try:
                download_file(video_url, local_filename)
                print(f'Successfully downloaded to {local_filename}')
            except requests.exceptions.RequestException as e:
                print(f'Failed to download {video_url}: {e}')

print('Finished downloading all videos for the sign "mother".')
