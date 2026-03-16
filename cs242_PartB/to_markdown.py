import pymupdf4llm
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

folder = "papers"
output_folder = "paper_markdown"

os.makedirs(output_folder, exist_ok=True)

def process_pdf(file):
    try:
        input_path = os.path.join(folder, file)
        output_path = os.path.join(output_folder, file.replace(".pdf", ".md"))
        
        md_text = pymupdf4llm.to_markdown(input_path)
        with open(output_path, "w") as f:
            f.write(md_text)
        
        return file
    except Exception:
        return None

if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    
    num_workers = cpu_count()
    
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_pdf, pdf_files), total=len(pdf_files)))