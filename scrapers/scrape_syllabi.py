import os
import sys
import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import timedelta
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from markdownify import markdownify as md
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from sites import load_or_exit # type: ignore
_cfg = load_or_exit()

# ==========================================
# SCRIPT SETTINGS
# ==========================================
TEST_LIMIT = False
MAX_SYLLABUSES = 5

BASE_FOLDER = str(_cfg.data_root)
DOMAIN = "https://sylabusy.agh.edu.pl"
START_URL = "https://sylabusy.agh.edu.pl/pl/1/2/22/0/0"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

# ==========================================
# SESSION AND RETRY CONFIGURATION
# ==========================================
session = requests.Session()
session.headers.update(HEADERS)

retry_strategy = Retry(
    total=3,
    connect=3,
    read=3,
    status=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ==========================================
# STATISTICS
# ==========================================
stat_downloaded = 0
stat_skipped = 0
stat_errors = 0
stat_size_bytes = 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_name(text):
    """Secures file names against strict Windows system rules."""
    name = re.sub(r'[\\/*?:"<>|]', "_", text.strip())
    name = name.rstrip(" .")
    
    MAX_LENGTH = 120
    if len(name) > MAX_LENGTH:
        name = name[:MAX_LENGTH].strip() + "..."

    reserved = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "LPT1", "LPT2", "LPT3"}
    if name.upper() in reserved:
        name += "_file"
        
    return name if name else "unknown_name"

def fetch_soup(url):
    """Fetches a page and returns a BeautifulSoup object, handling exceptions."""
    try:
        res = session.get(url, timeout=10)
        res.raise_for_status()
        return BeautifulSoup(res.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Connection error with {url}: {e}")
        return None

# ==========================================
# MAIN CRAWLER PROCESS
# ==========================================
print("Starting Syllabus Crawler\n")
start_time = time.time()

soup_main = fetch_soup(START_URL)
if not soup_main:
    print("Failed to fetch the main page. Stopping the script.")
    sys.exit(1)

academic_years = {}
year_header = soup_main.find(string=lambda t: t and "Rozpoczęcie studiów" in t)
if year_header:
    filter_box = year_header.find_parent('div', class_='syl-filter-box')
    for li in filter_box.find_all('li', class_='list-group-item'):
        if 'active' in li.get('class', []):
            name = clean_name(li.text)
            academic_years[name] = START_URL
        else:
            a_tag = li.find('a')
            if a_tag:
                name = clean_name(a_tag.text)
                academic_years[name] = urljoin(DOMAIN, a_tag.get('href'))

is_finished = False

for year, year_url in academic_years.items():
    if is_finished: break
    print(f"Processing year: {year}")
    
    soup_year = fetch_soup(year_url)
    if not soup_year: continue
    
    faculty_tags = soup_year.find_all('div', class_='syl-content-box-header')
    
    for faculty_tag in faculty_tags:
        if is_finished: break
        faculty_name = clean_name(faculty_tag.text)
        a_faculty = faculty_tag.find_parent('a')
        if not a_faculty: continue
        
        faculty_url = urljoin(DOMAIN, a_faculty.get('href'))
        print(f"  Faculty: {faculty_name}")
        
        soup_faculty = fetch_soup(faculty_url)
        if not soup_faculty: continue
        
        field_headings = soup_faculty.find_all('div', class_='student-view-content-list-heading')
        
        for heading in field_headings:
            if is_finished: break
            field_name = clean_name(heading.text)
            variant_list = heading.find_next_sibling('ul', class_='student-view-content-list')
            
            if not variant_list: continue
            
            for a_mode in variant_list.find_all('a'):
                if is_finished: break
                study_type = clean_name(a_mode.text.replace(', ', '_'))
                field_url = urljoin(DOMAIN, a_mode.get('href'))
                
                save_path = os.path.join(BASE_FOLDER, "Sylabusy_Markdown", year, faculty_name, field_name, study_type)
                os.makedirs(save_path, exist_ok=True)
                
                soup_subjects = fetch_soup(field_url)
                if not soup_subjects: continue
                
                buttons = soup_subjects.find_all('button', attrs={'data-syllabus-id': True})
                
                subjects_dict = {}
                for btn in buttons:
                    syl_id = btn.get('data-syllabus-id')
                    subject_name = clean_name(btn.text)
                    if subject_name and syl_id not in subjects_dict:
                        subjects_dict[syl_id] = subject_name
                
                for syl_id, subj_name in subjects_dict.items():
                    if is_finished: break
                    
                    file_path = os.path.join(save_path, f"{subj_name}.md")
                    
                    if os.path.exists(file_path):
                        stat_skipped += 1
                        continue
                        
                    api_url = urljoin(DOMAIN, f"/pl/document/{syl_id}.html?partialType=modal&forceEpd=false")
                    soup_api = fetch_soup(api_url)
                    
                    if soup_api:
                        raw_html = str(soup_api)
                        clean_markdown = md(raw_html, heading_style="ATX", strip=['script', 'style'])
                        clean_markdown = re.sub(r'\n{3,}', '\n\n', clean_markdown)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(clean_markdown.strip())
                        
                        stat_downloaded += 1
                        stat_size_bytes += len(clean_markdown.encode('utf-8'))
                        print(f"    Saved MD: {subj_name}")
                        
                        if TEST_LIMIT and stat_downloaded >= MAX_SYLLABUSES:
                            print("\nTest limit reached. Stopping...")
                            is_finished = True
                    else:
                        stat_errors += 1
                    
                    time.sleep(0.2)

# ==========================================
# FINAL REPORT
# ==========================================
end_time = time.time()
duration = str(timedelta(seconds=int(end_time - start_time)))
size_mb = stat_size_bytes / (1024 * 1024)
total_processed = stat_downloaded + stat_skipped + stat_errors

avg_time = (end_time - start_time) / max(stat_downloaded, 1)

report = f"""==================================================
DATA EXTRACTION REPORT
==================================================
Generation date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Total execution time: {duration} (HH:MM:SS)

Total processed files: {total_processed}
  - Downloaded and saved: {stat_downloaded}
  - Skipped (already existed): {stat_skipped}
  - Network/download errors: {stat_errors}

Downloaded data size (text only): {size_mb:.2f} MB
Average full cycle time per 1 file: {avg_time:.2f} s
==================================================
"""

print(report)

with open("syllabuses_statistics.txt", "w", encoding="utf-8") as f:
    f.write(report)