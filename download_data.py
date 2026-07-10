from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="data/soccernet")

for split in ["train", "valid", "test", "challenge"]:
    print(f"Downloading {split}...")
    try:
        downloader.downloadDataTask(
            task="mvfouls",
            split=[split],
            password="s0cc3rn3t"
        )
        print(f"{split} done.")
    except Exception as e:
        print(f"{split} failed: {e}")