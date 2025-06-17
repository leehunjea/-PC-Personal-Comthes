@echo off
chcp 65001 > nul
title 퍼스널컬러 분석기 실행기

REM 가상환경 활성화
call C:\Users\AI-LHJ\AppData\Local\anaconda3\Scripts\activate.bat LHJ2025

REM 실행 디렉토리로 이동
cd /d "C:\Users\AI-LHJ\Desktop\PC Project\analysis\PersonalColorAnalyzerModel"

REM 이미지 경로 입력 받기
set /p image_path=분석할 이미지 경로를 입력하세요 (예: C:\Users\AI-LHJ\Desktop\TEST3.jpg):

REM 분석 실행
python -m personalcolor_ai.main "%image_path%" --visualize

pause
