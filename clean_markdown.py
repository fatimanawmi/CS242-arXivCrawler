import re
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_folder = "paper_markdown"
output_folder = "paper_markdown_cleaned"

os.makedirs(output_folder, exist_ok=True)


def clean_markdown(text):

    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)  
    text = re.sub(r'\$[^\$\n]+\$', '', text)  
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)  
    text = re.sub(r'\\\(.*?\\\)', '', text)  
    text = re.sub(r'[_\s]*\[[^\]]+\][_\s]*', ' ', text)
    text = re.sub(r'\(\s*\[\s*\[.*?\]\s*\]\s*\)', '', text)
    
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        
        if re.match(r'^\s*\|?[\s\-:|]+\|', line):
            continue
            
        if '|' in line and '<br>' in line:
            continue
            
        if line.count('|') >= 3:
            continue
            
        if re.search(r'Col\d+', line) and '|' in line:
            continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\[\d+\]', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'\1', text)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  
    text = re.sub(r'(?<!\*)\*(?!\*)([^\*\n]+)\*(?!\*)', r'\1', text)  
    
    
    unicode_replacements = {
        
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'λ': 'lambda', 'μ': 'mu', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi',
        'ψ': 'psi', 'ω': 'omega', 'Δ': 'Delta', 'Σ': 'Sigma', 'Ω': 'Omega',
        'θ': 'theta', 'ρ': 'rho', 'π': 'pi', 'κ': 'kappa', 'η': 'eta',
        '×': 'x', '·': '*', '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
        '∈': 'in', '∀': 'forall', '∃': 'exists', '∇': 'nabla',
        '→': '->', '←': '<-', '⇒': '=>', '⇔': '<=>', '↑': 'up', '↓': 'down',
        '∼': '~', '∥': '||', '∫': 'integral', '∑': 'sum', '∏': 'product',
        '∞': 'infinity', '√': 'sqrt', '∂': 'partial',
    }
    
    for unicode_char, replacement in unicode_replacements.items():
        text = text.replace(unicode_char, replacement)
    
    
    text = re.sub(r'\n\s*\d+\s*\n\s*Published as a conference paper at ICLR 2025\s*\n', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*\n\s*Published as a conference paper at ICLR 2025\s*\n', '', text)
    text = re.sub(r'^(Figure|Table)\s+\d+:?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)  
    text = re.sub(r' {2,}', ' ', text)  
    
    
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if line.strip() and not re.match(r'^[\s\-_|]+$', line.strip())]
    text = '\n'.join(cleaned_lines)
    
    return text.strip()
def process_markdown_file(args):
    """Process a single markdown file"""
    input_path, output_path = args
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = clean_markdown(content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return os.path.basename(input_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None
def main():    
    
    os.makedirs(output_folder, exist_ok=True)

    input_path = Path(input_folder)
    md_files = list(input_path.glob("*.md"))
    print(f"Found {len(md_files)} markdown files to clean")
    
    process_args = [
        (str(md_file), os.path.join(output_folder, md_file.name))
        for md_file in md_files
    ]
    
    num_workers = cpu_count()
    print(f"Processing with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_markdown_file, process_args),
            total=len(process_args),
            desc="Cleaning markdown files"
        ))
    
    successful = sum(1 for r in results if r is not None)
    print(f"\nCompleted! Successfully cleaned {successful}/{len(md_files)} files")
    print(f"Cleaned files saved to: {output_folder}/")
if __name__ == "__main__":
    main()
