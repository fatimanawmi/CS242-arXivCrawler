import re
import json
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from download_papers import conference_json_file

input_folder = "paper_markdown_cleaned"
output_folder = "papers_json"

os.makedirs(output_folder, exist_ok=True)

conference_json_file = "all_papers.json"
with open(conference_json_file, 'r', encoding='utf-8') as f:
    all_papers = json.load(f)

def parse_markdown_to_json(text):

    lines = text.split('\n')
    
    
    while lines and not lines[0].strip():
        lines.pop(0)
    
    

    to_remove = ['Published as a conference paper at ICLR', 'Conference on Neural Information Processing Systems']
    
    if lines and any(phrase in lines[0] for phrase in to_remove):
        lines.pop(0)
    
    
    while lines and not lines[0].strip():
        lines.pop(0)
    
    
    title = ""
    title_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            title = stripped.lstrip('#').strip()
            title_idx = i
            break
    
    
    authors = ""
    authors_idx = -1
    for i in range(title_idx + 1, len(lines)):
        if lines[i].strip():
            authors = lines[i].strip()
            
            if authors.endswith(' Abstract'):
                authors = authors[:-9].strip()
            elif authors.endswith(' ABSTRACT'):
                authors = authors[:-9].strip()
            authors_idx = i
            break
    
    
    
    
    
    section_pattern = re.compile(r'^(\d+)\s+([A-Z][A-Z\s\-:]+[A-Z])$')
    bold_section_pattern = re.compile(r'^\*\*(\d+)\.\s+(.+?)\*\*$')
    plain_section_pattern = re.compile(r'^(\d+)\.\s+([A-Z][A-Za-z\s\-\(\)]+)$')
    
    roman_section_pattern = re.compile(r'^([IVX]+)\.\s+([A-Z][A-Z\s\-]+)$')
    
    markdown_heading_pattern = re.compile(r'^###\s+(\d+)\s+(.+)$')
    abstract_pattern = re.compile(r'^ABSTRACT$')
    bold_abstract_pattern = re.compile(r'^\*\*Abstract\*\*$')
    
    plain_abstract_pattern = re.compile(r'^Abstract$')
    
    abstract_with_dash_pattern = re.compile(r'^Abstract\s*[—\-–]')
    references_pattern = re.compile(r'^REFERENCES$')
    appendix_pattern = re.compile(r'^(APPENDIX|APPENDICES)$')
    appendix_heading_pattern = re.compile(r'^##\s+Appendix', re.IGNORECASE)
    
    sections_data = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        
        if abstract_pattern.match(stripped):
            sections_data.append({
                'type': 'special',
                'name': 'ABSTRACT',
                'number': None,
                'line_idx': i
            })
        
        elif bold_abstract_pattern.match(stripped):
            sections_data.append({
                'type': 'special',
                'name': 'ABSTRACT',
                'number': None,
                'line_idx': i
            })
        
        elif plain_abstract_pattern.match(stripped):
            sections_data.append({
                'type': 'special',
                'name': 'ABSTRACT',
                'number': None,
                'line_idx': i
            })
        
        elif abstract_with_dash_pattern.match(stripped):
            
            sections_data.append({
                'type': 'special',
                'name': 'ABSTRACT',
                'number': None,
                'line_idx': i,  
                'inline_start': True  
            })
        
        elif references_pattern.match(stripped):
            sections_data.append({
                'type': 'special',
                'name': 'REFERENCES',
                'number': None,
                'line_idx': i
            })
        
        elif appendix_pattern.match(stripped):
            sections_data.append({
                'type': 'special',
                'name': stripped,  
                'number': None,
                'line_idx': i
            })
        
        elif appendix_heading_pattern.match(stripped):
            sections_data.append({
                'type': 'special',
                'name': 'APPENDIX',
                'number': None,
                'line_idx': i
            })
        
        elif section_pattern.match(stripped):
            match = section_pattern.match(stripped)
            sections_data.append({
                'type': 'section',
                'name': match.group(2).strip(),
                'number': match.group(1),
                'line_idx': i
            })
        
        elif bold_section_pattern.match(stripped):
            match = bold_section_pattern.match(stripped)
            sections_data.append({
                'type': 'section',
                'name': match.group(2).strip().upper(),
                'number': match.group(1),
                'line_idx': i
            })
        
        elif plain_section_pattern.match(stripped):
            match = plain_section_pattern.match(stripped)
            section_name = match.group(2).strip()
            
            if len(section_name) <= 40:  
                sections_data.append({
                    'type': 'section',
                    'name': section_name.upper(),
                    'number': match.group(1),
                    'line_idx': i
                })
        
        elif roman_section_pattern.match(stripped):
            match = roman_section_pattern.match(stripped)
            section_name = match.group(2).strip()
            sections_data.append({
                'type': 'section',
                'name': section_name,
                'number': match.group(1),
                'line_idx': i
            })
        
        elif markdown_heading_pattern.match(stripped):
            match = markdown_heading_pattern.match(stripped)
            section_name = match.group(2).strip()
            
            if len(section_name) <= 40:
                sections_data.append({
                    'type': 'section',
                    'name': section_name.upper(),
                    'number': match.group(1),
                    'line_idx': i
                })
    
    
    result = {
        'title': title,
        'authors': authors,
        'abstract': '',
        'sections': {}
    }
    
    
    has_abstract_section = any(s['name'] == 'ABSTRACT' for s in sections_data)
    
    
    if not has_abstract_section and authors_idx >= 0:
        if sections_data:
            
            first_section_idx = sections_data[0]['line_idx']
            abstract_lines = []
            for j in range(authors_idx + 1, first_section_idx):
                if lines[j].strip():  
                    
                    for k in range(j, first_section_idx):
                        abstract_lines.append(lines[k])
                    break
            result['abstract'] = '\n'.join(abstract_lines).strip()
        else:
            
            
            abstract_lines = []
            chars_count = 0
            max_abstract_chars = 3000  
            for j in range(authors_idx + 1, len(lines)):
                line = lines[j].strip()
                if not line and abstract_lines:  
                    
                    if chars_count > 200:  
                        break
                if line:
                    abstract_lines.append(line)
                    chars_count += len(line)
                    if chars_count > max_abstract_chars:
                        break
            result['abstract'] = ' '.join(abstract_lines).strip()
    
    for i, section in enumerate(sections_data):
        
        if section.get('inline_start'):
            
            line_text = lines[section['line_idx']].strip()
            abstract_match = re.match(r'^Abstract\s*[—\-–]\s*(.+)', line_text)
            end_idx = sections_data[i + 1]['line_idx'] if i + 1 < len(sections_data) else len(lines)
            
            if abstract_match:
                
                first_line_content = abstract_match.group(1)
                rest_lines = []
                for j in range(section['line_idx'] + 1, end_idx):
                    rest_lines.append(lines[j])
                content = first_line_content + '\n' + '\n'.join(rest_lines)
                content = content.strip()
            else:
                content_lines = []
                for j in range(section['line_idx'] + 1, end_idx):
                    content_lines.append(lines[j])
                content = '\n'.join(content_lines).strip()
        else:
            start_idx = section['line_idx'] + 1
            end_idx = sections_data[i + 1]['line_idx'] if i + 1 < len(sections_data) else len(lines)
            
            content_lines = []
            for j in range(start_idx, end_idx):
                content_lines.append(lines[j])
            
            content = '\n'.join(content_lines).strip()
        
        if section['name'] == 'ABSTRACT':
            result['abstract'] = content
        elif section['type'] == 'special':
            
            result['sections'][section['name']] = content
        else:
            
            key = f"{section['number']} {section['name']}" if section['number'] else section['name']
            result['sections'][key] = content
    
    return result

def find_metadata(file_name):
    for paper in all_papers:
        if 'sanitized_title' not in paper:
            continue

        if paper['sanitized_title'] in file_name:
            return paper
    return None

def process_markdown_file(args):
    input_path, output_path = args
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parsed_data = parse_markdown_to_json(content)
    metadata = find_metadata(os.path.basename(input_path))
    parsed_data['metadata'] = metadata
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    return os.path.basename(input_path)


def main():

    input_path = Path(input_folder)
    md_files = list(input_path.glob("*.md"))
    
    print(f"Found {len(md_files)} markdown files to convert")
    
    process_args = [
        (str(md_file), os.path.join(output_folder, md_file.stem + '.json'))
        for md_file in md_files
    ]
    
    num_workers = cpu_count()
    print(f"Processing with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_markdown_file, process_args),
            total=len(process_args),
            desc="Converting to JSON"
        ))
    
    print(f"\nCompleted! Converted {len(md_files)} files")
    print(f"JSON files saved to: {output_folder}/")


if __name__ == "__main__":
    main()
