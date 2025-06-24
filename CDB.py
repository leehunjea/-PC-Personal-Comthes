import os
import json
import concurrent.futures
from tqdm import tqdm

def process_json_file(args):
    json_path, img_dir, style, id_counter = args
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_file = data.get('이미지 정보', {}).get('이미지 파일명', '')
        labels = data.get('데이터셋 정보', {}).get('라벨링', {})
        style_label = ''
        if labels.get('스타일'):
            style_label = labels.get('스타일', [{}])[0].get('스타일', '')
        item = {
            'id': id_counter,
            'style_folder': style,
            'image_file': os.path.join(img_dir, img_file),
            'style_label': style_label,
            'labels': labels
        }
        return item
    except Exception as e:
        print(f"JSON 파일 읽기 오류: {json_path}: {e}")
        return None

def build_clothes_db_parallel(base_dirs, style_folders, max_workers=8):
    clothes_db = []
    id_counter = 0
    tasks = []
    for base_dir in base_dirs:
        json_base_dir = os.path.join(base_dir, 'label')
        img_base_dir = os.path.join(base_dir, 'img')
        for style in style_folders:
            json_dir = os.path.join(json_base_dir, style)
            img_dir = os.path.join(img_base_dir, style)
            if not os.path.isdir(json_dir):
                print(f"폴더 없음: {json_dir}")
                continue
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            for jf in json_files:
                json_path = os.path.join(json_dir, jf)
                tasks.append((json_path, img_dir, style, id_counter))
                id_counter += 1

    # 병렬 처리 시작
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_json_file, tasks), total=len(tasks)):
            if result is not None:
                clothes_db.append(result)
    print(f"총 {len(clothes_db)}개 아이템이 DB에 저장되었습니다.")
    return clothes_db

# 반복할 데이터셋 경로 리스트
base_dirs = [
    r'C:/Users/AI-LHJ/Desktop/K-Fashion/Training',
    r'C:/Users/AI-LHJ/Desktop/K-Fashion/Validation'
]
style_folders = ['기타', '레트로', '로맨틱', '리조트', '매니시','모던','밀리터리','섹시','소피스트케이티드','스트리트','스포티','아방가르드','오리엔탈','웨스턴','젠더리스','컨트리','클래식','키치','톰보이','펑크','페미닌','프레피','히피','힙합']

# max_workers는 CPU 코어 수에 맞게 조정(8~16 권장)
clothes_db = build_clothes_db_parallel(base_dirs, style_folders, max_workers=8)

# 일부 결과 출력
print("DB 샘플:")
for item in clothes_db[:5]:
    print(item)

# DB를 파일로 저장
save_path = 'clothes_db.json'
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(clothes_db, f, ensure_ascii=False, indent=2)
print(f"DB가 {save_path}에 저장되었습니다.")
