import os
import sys
import time
import itertools
import urllib.request
import arxiv

# Keywords for paper generation
group1 = ["UAV", "drone", "hexacopter"]
group2 = ["trajectory optimization", "path planning", "spraying payload"]
group3 = ["reinforcement learning", "MAVLink", "flight control"]

permutations = list(itertools.product(group1, group2, group3))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

downloaded_titles = set()
MAX_DOWNLOADS = 15
current_count = 0

client = arxiv.Client()

def print_status_box(query, current_file, downloaded, total):
    """Draws a live multi-line status box using standard terminal ANSI sequences."""
    percent = int((downloaded / total) * 100)
    bar_length = 30
    filled = int(bar_length * downloaded // total)
    bar = '=' * filled + '-' * (bar_length - filled)
    
    # \033[K clears to the end of the line
    lines = [
        "\033[H\033[J",  # Move cursor to top-left & clear screen
        "============================================================",
        "             UAV DATASET DOWNLOADER STATUS                  ",
        "============================================================",
        f" [1] Target Downloads : {downloaded}/{total} PDFs",
        f" [2] Progress         : [{bar}] {percent}%",
        f" [3] Current Query    : {query[:45]}",
        f" [4] Active Download  : {current_file[:45]}",
        "============================================================",
        " Status: DOWNLOADING (Press Ctrl+C to abort)...",
        ""
    ]
    
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()

# Clear screen once on boot
sys.stdout.write("\033[2J")

for combo in permutations:
    if current_count >= MAX_DOWNLOADS:
        break

    query_str = f"{combo[0]} {combo[1]} {combo[2]}"
    print_status_box(query_str, "Searching arXiv database...", current_count, MAX_DOWNLOADS)

    search = arxiv.Search(
        query=query_str,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    try:
        results = list(client.results(search))
        for result in results:
            if current_count >= MAX_DOWNLOADS:
                break

            clean_title = "".join(c for c in result.title if c.isalnum() or c in " _-").strip()
            if clean_title in downloaded_titles:
                continue

            downloaded_titles.add(clean_title)
            filename = f"paper_{current_count + 1}.pdf"
            filepath = os.path.join(DATA_DIR, filename)

            print_status_box(query_str, f"{filename} ({result.title[:25]}...)", current_count, MAX_DOWNLOADS)
            
            # Download file
            try:
                req = urllib.request.Request(result.pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(filepath, 'wb') as out_file:
                    out_file.write(resp.read())
            except Exception:
                urllib.request.urlretrieve(result.pdf_url, filepath)

            current_count += 1
            print_status_box(query_str, f"Saved {filename}", current_count, MAX_DOWNLOADS)
            time.sleep(1)

    except Exception as e:
        print_status_box(query_str, f"Error: {e}", current_count, MAX_DOWNLOADS)
        time.sleep(1)

# Final Status View
print_status_box("ALL TASKS COMPLETED", "None", current_count, MAX_DOWNLOADS)
print(f"\n Finished! {current_count} PDFs are ready in '{DATA_DIR}/'.")
print(" Ingestion was NOT auto-triggered. Run 'python3 src/ingestion.py' when ready.\n")
