from PyPDF2 import PdfReader
import os
import re

def clean_text(text):
    # Loại bỏ các ký tự đặc biệt không cần thiết
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\n', '', text)
    return text.strip()

def format_section(text):
    # Phát hiện và định dạng các tiêu đề
    if re.match(r'^\d+\.\s+[A-Z]', text):
        return f"## {text}\n"
    elif re.match(r'^\d+\.\d+\.\s+[A-Z]', text):
        return f"### {text}\n"
    return f"{text}\n"

def pdf_to_markdown(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            print(f"Không tìm thấy file: {pdf_path}")
            return None

        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"Số trang trong PDF: {num_pages}")
        
        # Tạo cấu trúc Markdown
        markdown_content = []
        
        # Thêm metadata
        markdown_content.append("---\n")
        markdown_content.append(f"title: {os.path.basename(pdf_path).replace('.pdf', '')}\n")
        markdown_content.append(f"pages: {num_pages}\n")
        markdown_content.append("---\n\n")
        
        # Xử lý từng trang
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            text = clean_text(text)
            
            # Thêm tiêu đề trang
            markdown_content.append(f"# Trang {page_num}\n\n")
            
            # Phân tách và định dạng các đoạn
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    formatted_para = format_section(para)
                    markdown_content.append(formatted_para)
            
            markdown_content.append("\n---\n\n")
        
        # Ghi file Markdown
        output_file = pdf_path.replace('.pdf', '.md')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(markdown_content)
        
        print(f"Đã chuyển đổi thành công sang file: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Lỗi khi chuyển đổi PDF: {str(e)}")
        return None

if __name__ == "__main__":
    pdf_path = "DeepLOF.pdf"
    pdf_to_markdown(pdf_path) 