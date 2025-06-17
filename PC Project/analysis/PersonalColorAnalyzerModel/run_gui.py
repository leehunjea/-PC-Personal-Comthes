import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from personalcolor_ai.core import PersonalColorAI

def run_analysis():
    filepath = filedialog.askopenfilename(
        title="이미지 선택", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not filepath:
        return

    analyzer = PersonalColorAI()
    result = analyzer.analyze(filepath, visualize=True)

    output_box.delete(1.0, tk.END)  # 기존 텍스트 지우기

    if "error" in result:
        output_box.insert(tk.END, f"[오류] {result['error']}")
        return

    # 결과 포맷 출력
    output_box.insert(tk.END, f"입력 이미지 처리 후 크기: {result.get('image_shape_processed')}\n\n")
    output_box.insert(tk.END, f"추출된 주요 색상 (BGR):\n")
    for part in ['skin', 'eye', 'hair']:
        color = result['extracted_colors_bgr'].get(part)
        output_box.insert(tk.END, f"  - {part}: {color if color is not None else '추출 실패'}\n")

    output_box.insert(tk.END, "\n색상 속성 분석:\n")
    for key, value in result['color_attributes'].items():
        output_box.insert(tk.END, f"  - {key}: {value}\n")

    output_box.insert(tk.END, f"\n예측된 퍼스널컬러 시즌: {result['personal_color_season']}\n")

# ---------- GUI 구성 ----------
root = tk.Tk()
root.title("퍼스널컬러 진단")
root.geometry("600x600")

btn = tk.Button(root, text="이미지 선택 및 분석 시작", command=run_analysis, font=("맑은 고딕", 12))
btn.pack(pady=10)

output_box = scrolledtext.ScrolledText(root, width=80, height=30, font=("Consolas", 10))
output_box.pack(padx=10, pady=10)

root.mainloop()
