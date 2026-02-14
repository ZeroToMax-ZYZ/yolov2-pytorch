import os

def merge_code_for_gemini(repo_path, output_file):
    # 想包含的文件后缀
    allowed_extensions = ('.py', '.yaml', '.md', '.txt', '.json', '.csv') 
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(repo_path):
            # 排除隐藏文件夹（比如 .git）和缓存文件夹
            if '.git' in root or '__pycache__' in root or 'runs' in root:
                continue
                
            for file in files:
                if file.endswith(allowed_extensions):
                    filepath = os.path.join(root, file)
                    outfile.write(f"\n\n{'='*50}\n")
                    outfile.write(f"File Path: {filepath}\n")
                    outfile.write(f"{'='*50}\n")
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            outfile.write(f.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")
    print(f"打包完成！文件已保存为: {output_file}")


merge_code_for_gemini('.', 'yolov2_project_code.txt')