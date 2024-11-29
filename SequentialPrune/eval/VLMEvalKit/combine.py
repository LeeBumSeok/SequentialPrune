import os
import re
import csv
# /home/work/workspace_bum/Tokenpruning/FastV/eval/VLMEvalKit/sllava_v1.5_7b_rank_0.75_layer_5_seq_step_5/llava_v1.5_7b/
folder_path = 'llava_v1.5_7b'
csv_filename = 'llava_v1.5_7b/llava_v1.5_7b_AI2D_TEST_acc.csv'

# 현재 디렉토리에서 해당 문자열을 포함하는 폴더/파일 리스트를 가져오기
file_list = [name for name in os.listdir('./') if folder_path in name]

# rank 값을 추출하는 함수 정의
def extract_rank(file_name):
    match = re.search(r'rank_([\d.]+)', file_name)
    return float(match.group(1)) if match else 0

# step 값을 추출하는 함수 정의
def extract_step(file_name):
    match = re.search(r'step_(\d+)', file_name)
    return int(match.group(1)) if match else 0

def extract_layer(file_name):
    match = re.search(r'layer_(\d+)', file_name)
    return int(match.group(1)) if match else 0


# CSV 파일에서 데이터를 읽어오기 위한 함수 정의
def extract_csv_data(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'none':
                return row  # "none", "value" 형태의 데이터를 반환
    return None

# 파일 리스트를 rank와 step 기준으로 정렬
sorted_file_list = sorted(file_list, key=lambda x: (extract_rank(x), extract_step(x), extract_layer(x)))

# 각 폴더에서 csv 파일 열어서 데이터 추출
csv_data = []
for folder in sorted_file_list:
    csv_path = os.path.join(folder, csv_filename)
    print(csv_path)
    if os.path.exists(csv_path):
        data = extract_csv_data(csv_path)[1]
        print(data)
        if data:
            csv_data.append({folder: data})

csv_output_filename = 'extracted_csv_data.csv'
with open(csv_output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Folder', 'Value'])
    for data in csv_data:
        for folder, value in data.items():
            writer.writerow([folder, value])