import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import re
import xml.etree.ElementTree as ET

BETA_TEST = True
conference_json_file = "all_papers.json"
output_dir = "papers"

class ArxivPDFDownloader:
    def __init__(self, output_dir: str = "papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        self.arxiv_base = "http://export.arxiv.org/api/query"

    
    def sanitize_filename(self, title: str, max_length: int = 150) -> str:
        
        sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
        sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
        
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rsplit('_', 1)[0]
        return sanitized
    
    def search_arxiv(self, title: str) -> Optional[Dict]:
        clean_title = re.sub(r'[^\w\s]', ' ', title)
        clean_title = ' '.join(clean_title.split())
        
        params = {
            'search_query': f'ti:"{clean_title}"',
            'max_results': 3,
            'sortBy': 'relevance'
        }
        
        response = requests.get(self.arxiv_base, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        
        if not entries:
            return None
        
        entry = entries[0]
        arxiv_title = entry.find('atom:title', namespace).text.strip()
        
        
        def normalize(t):
            t = t.lower()
            t = re.sub(r'[^\w\s]', '', t)
            return ' '.join(t.split())
        
        t1 = normalize(title)
        t2 = normalize(arxiv_title)
        
        if not (t1 in t2 or t2 in t1):
            words1 = set(t1.split())
            words2 = set(t2.split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            jaccard = intersection / union
            if jaccard < 0.8:
                return None
        
        
        categories = []
        for category_elem in entry.findall('atom:category', namespace):
            term = category_elem.get('term')
            if term:
                categories.append(term)
        
        pdf_link = None
        for link in entry.findall('atom:link', namespace):
            if link.get('title') == 'pdf':
                pdf_link = link.get('href')
                break
        
        if not pdf_link:
            arxiv_id = entry.find('atom:id', namespace).text.split('/abs/')[-1]
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return {
            'title': arxiv_title,
            'pdf_url': pdf_link,
            'arxiv_id': pdf_link.split('/')[-1].replace('.pdf', ''),
            'categories': categories
        }
    
    def process_paper(self, paper: Dict):
        title = paper['title']
        arxiv_result = self.search_arxiv(title)
        
        if not arxiv_result:
            print(f"Not on arXiv: {title}")
            return
        
        paper['arxiv_url'] = arxiv_result['pdf_url']
        
        arxiv_id = arxiv_result['arxiv_id']
        sanitized_title = self.sanitize_filename(paper['title'])
        output_path = self.output_dir / f"{sanitized_title}.pdf"
        
        if output_path.exists():
            return
        
        
        response = requests.get(arxiv_result['pdf_url'], headers=self.headers, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
    
    def download_papers(self, papers: List[Dict], delay: float = 3.0):
        
        for i, paper in enumerate(papers, 1):
            print(f"[{i}/{len(papers)}] {paper['title'][:60]}")
            try:
                self.process_paper(paper)
            except Exception as e:
                print(f"Error processing paper: {e}")
            
            if i < len(papers):
                time.sleep(delay)

            if BETA_TEST and i >= 50:
                break
        
        return papers

def main():
    

    with open(conference_json_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    downloader = ArxivPDFDownloader(output_dir=output_dir)
    papers = downloader.download_papers(papers, delay=5.0)

    with open(conference_json_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()

