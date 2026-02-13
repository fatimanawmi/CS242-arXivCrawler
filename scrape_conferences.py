import requests
import json
import time
from typing import List, Dict
import re

def sanitize_filename(title: str, max_length: int = 150) -> str:
    
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit('_', 1)[0]
    return sanitized


class OpenReviewScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    
    def scrape(self, venue: str, year: int) -> List[Dict]:
        venue_id = f'{venue}.cc/{year}/Conference'
        papers = []
        offset = 0
        limit = 1000
        
        while offset < 15000:
            response = requests.get(
                "https://api2.openreview.net/notes",
                params={'invitation': f'{venue_id}/-/Submission', 'details': 'replyCount,presentation', 'offset': offset, 'limit': limit},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                break
            
            notes = response.json().get('notes', [])
            if not notes:
                break
            
            for note in notes:
                content = note.get('content', {})
                title = self._extract(content, 'title')
                if not title:
                    continue
                
                authors = self._extract(content, 'authors')
                if isinstance(authors, list):
                    authors = ', '.join(str(a) for a in authors)
                
                papers.append({
                    'title': title,
                    'authors': authors or '',
                    'abstract': self._extract(content, 'abstract') or '',
                    'conference': venue,
                    'year': year,
                    'url': f"https://openreview.net/forum?id={note.get('id', '')}",
                    'id': note.get('id', ''),
                    "sanitized_title": sanitize_filename(title)
                })
            
            offset += len(notes)
            if len(notes) < limit:
                break
            time.sleep(0.5)
        
        return papers
    
    def _extract(self, content: dict, field: str):
        value = content.get(field, '')
        return value.get('value', '') if isinstance(value, dict) else value
    
    def save(self, papers: List[Dict], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    scraper = OpenReviewScraper()
    venues = ['ICLR', 'ICML', 'NeurIPS', 'COLM']
    years = [2024, 2025]
    
    all_papers = []
    for venue in venues:
        for year in years:
            print(f"Scraping {venue} {year}")
            papers = scraper.scrape(venue, year)
            all_papers.extend(papers)
            time.sleep(2)
    
    scraper.save(all_papers, 'all_papers.json')
