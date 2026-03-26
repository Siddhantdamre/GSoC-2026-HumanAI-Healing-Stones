import requests
import os

def download_fragments():
    url = "https://cernbox.cern.ch/s/hQO24HxuKi6VeQo/download"
    output = "data/fragments.zip"
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print("Downloading 3D fragments from CERNbox...")
    response = requests.get(url, stream=True)
    with open(output, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete. Please extract fragments into the /data folder.")

if __name__ == "__main__":
    download_fragments()