Scripts for creating and curating the cosplay image dataset.

## Crawler Scripts

Crawlers for downloading images from different sources:

- **Wikimedia Commons** (`wikimedia_commons/`) — BFS category crawl, metadata collection, and image download from Wikimedia Commons. 
- **Reddit** (`reddit/`) — scrapes cosplay posts by character from subreddits.
- **ACP** (`acp/`) — browser-side JS scripts to extract image URLs from the ACP cosplay gallery.
- **Endorphin** (`endorphin_crawler.py`, `faster_endorphin_crawler.py`) — crawls the Endorphin cosplay platform.
- **Fantasy Basel** (`fantasy_basel_crawler.py`) — scrapes cosplay photos from the Fantasy Basel convention website.
- **Google / DuckDuckGo** (`download_google_test_images.py`) — downloads images per character label using image search queries.

## Labeling Workflow

A Streamlit app (`streamlit_app/app.py`) for manually reviewing and labeling crawled images:

1. Point the app at a JSONL metadata file produced by a crawler.
2. Each image is shown alongside its metadata. Assign a `label` (character name) and `franchise`.
3. Mark the record as **validated** — it gets moved to `validated.jsonl` and the reviewer advances automatically.
4. Special cases: use **Move to MULTIPLE** for images showing more than one character, **Move to NONE** for images with no recognizable cosplay character. Both move the image file into a subfolder and write the record to a sidecar JSONL.
5. Records can also be deleted outright (removes the image file and the JSONL entry).

`move_special_labels.py` is a CLI utility to batch-move already-labeled MULTIPLE/NONE records into their subfolders and update the paths in the JSONL.


## Cleanup Scripts 

Various Scripts that help cleanup the images, find duplicates and split the dataset into train/val/test sets. 

