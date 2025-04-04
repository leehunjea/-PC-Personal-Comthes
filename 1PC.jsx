import React, { useState, useRef, useEffect } from 'react';

const PersonalColorAnalyzer = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [faceSelection, setFaceSelection] = useState(null);
  const [selectionMode, setSelectionMode] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [colorSet, setColorSet] = useState('warm');
  const [selectedColor, setSelectedColor] = useState(null);
  const [simulationImageUrl, setSimulationImageUrl] = useState(null);
  const canvasRef = useRef(null);
  const selectionCanvasRef = useRef(null);
  const simulationCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // 16가지 퍼스널컬러 정의
  const personalColors = {
    // 봄 웜톤
    'SPRING_BRIGHT': { name: '봄 브라이트', characteristics: '선명하고 밝은 색감, 노란 기운의 따뜻한 피부톤' },
    'SPRING_LIGHT': { name: '봄 라이트', characteristics: '부드럽고 밝은 파스텔톤, 밝은 피부톤' },
    'SPRING_WARM': { name: '봄 웜', characteristics: '따뜻하고 선명한 색감, 황금빛 피부톤' },
    'SPRING_TRUE': { name: '봄 트루', characteristics: '맑고 생기있는 색감, 복숭아빛 피부톤' },
    
    // 여름 쿨톤
    'SUMMER_LIGHT': { name: '여름 라이트', characteristics: '부드럽고 연한 파스텔, 푸른빛의 밝은 피부톤' },
    'SUMMER_TRUE': { name: '여름 트루', characteristics: '부드럽고 차분한 중채도, 푸른빛의 피부톤' },
    'SUMMER_SOFT': { name: '여름 소프트', characteristics: '부드럽고 흐린 색감, 회색빛이 도는 피부톤' },
    'SUMMER_COOL': { name: '여름 쿨', characteristics: '시원하고 맑은 색감, 푸른빛이 도는 피부톤' },
    
    // 가을 웜톤
    'AUTUMN_DEEP': { name: '가을 딥', characteristics: '깊고 진한 색감, 황금빛이 도는 피부톤' },
    'AUTUMN_TRUE': { name: '가을 트루', characteristics: '따뜻하고 풍부한 색감, 황토색 피부톤' },
    'AUTUMN_SOFT': { name: '가을 소프트', characteristics: '부드럽고 흐린 색감, 베이지 피부톤' },
    'AUTUMN_WARM': { name: '가을 웜', characteristics: '따뜻하고 깊은 색감, 황금빛 피부톤' },
    
    // 겨울 쿨톤
    'WINTER_DEEP': { name: '겨울 딥', characteristics: '진하고 강렬한 색감, 짙은 피부톤' },
    'WINTER_TRUE': { name: '겨울 트루', characteristics: '선명하고 강한 색감, 푸른빛이 도는 피부톤' },
    'WINTER_BRIGHT': { name: '겨울 브라이트', characteristics: '선명하고 밝은 색감, 선명한 피부톤' },
    'WINTER_COOL': { name: '겨울 쿨', characteristics: '차갑고 강한 색감, 푸른빛이 도는 피부톤' }
  };

  // 컬러 드레이핑용 색상 팔레트
  const drapingColors = {
    warm: [
      { name: '웜톤 오렌지', hex: '#FF6B35', season: 'SPRING' },
      { name: '웜톤 코랄', hex: '#FF8C69', season: 'SPRING' },
      { name: '웜톤 골드', hex: '#D4AF37', season: 'AUTUMN' },
      { name: '웜톤 올리브', hex: '#808000', season: 'AUTUMN' },
    ],
    cool: [
      { name: '쿨톤 블루', hex: '#4682B4', season: 'WINTER' },
      { name: '쿨톤 퍼플', hex: '#800080', season: 'WINTER' },
      { name: '쿨톤 라벤더', hex: '#B57EDC', season: 'SUMMER' },
      { name: '쿨톤 핑크', hex: '#DB7093', season: 'SUMMER' },
    ],
    neutral: [
      { name: '레드', hex: '#FF0000', season: '' },
      { name: '그린', hex: '#008000', season: '' },
      { name: '블랙', hex: '#000000', season: '' },
      { name: '화이트', hex: '#FFFFFF', season: '' },
    ]
  };

  // 계절별 컬러 팔레트
  const seasonPalettes = {
    'SPRING': ['#FF9900', '#FFCC00', '#FF6600', '#CC9933', '#FFFF66'],
    'SUMMER': ['#6699CC', '#9999CC', '#CC99CC', '#CCCCFF', '#99CCCC'],
    'AUTUMN': ['#996633', '#CC6633', '#CC9966', '#999966', '#666633'],
    'WINTER': ['#000000', '#0000CC', '#CC0000', '#FFFFFF', '#006666']
  };

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setImageUrl(URL.createObjectURL(file));
      setResult(null);
      setFaceSelection(null);
      setSelectionMode(false);
      setSimulationImageUrl(null);
      setSelectedColor(null);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  useEffect(() => {
    if (imageUrl && selectionCanvasRef.current) {
      const img = new Image();
      img.src = imageUrl;
      img.onload = () => {
        const canvas = selectionCanvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // 캔버스 크기 설정
        canvas.width = 400;
        canvas.height = 400 * (img.height / img.width);
        
        // 이미지 그리기
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // 얼굴 선택 영역 초기화
        if (faceSelection) {
          drawSelectionBox(ctx, faceSelection);
        }
      };
    }
  }, [imageUrl, faceSelection]);

  // 얼굴 영역 선택 시작
  const startFaceSelection = () => {
    setSelectionMode(true);
  };

  // 마우스 다운 이벤트 핸들러
  const handleMouseDown = (e) => {
    if (!selectionMode) return;
    
    const canvas = selectionCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setStartPoint({ x, y });
    setFaceSelection({ x, y, width: 0, height: 0 });
  };

  // 마우스 이동 이벤트 핸들러
  const handleMouseMove = (e) => {
    if (!selectionMode || !faceSelection) return;
    
    const canvas = selectionCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // 선택 영역 업데이트
    const newSelection = {
      x: startPoint.x,
      y: startPoint.y,
      width: x - startPoint.x,
      height: y - startPoint.y
    };
    
    setFaceSelection(newSelection);
    
    // 캔버스 다시 그리기
    const img = new Image();
    img.src = imageUrl;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawSelectionBox(ctx, newSelection);
  };

  // 마우스 업 이벤트 핸들러
  const handleMouseUp = () => {
    if (!selectionMode) return;
    setSelectionMode(false);
  };

  // 선택 영역 그리기
  const drawSelectionBox = (ctx, selection) => {
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(selection.x, selection.y, selection.width, selection.height);
  };

  // 선택한 얼굴 영역에서 피부톤 추출
  const extractSkinTone = () => {
    if (!faceSelection || !selectionCanvasRef.current) return null;
    
    const canvas = selectionCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 선택 영역의 좌표 정규화
    const selection = normalizeSelection(faceSelection);
    
    // 이미지 데이터 추출
    const imageData = ctx.getImageData(selection.x, selection.y, selection.width, selection.height);
    const data = imageData.data;
    
    // 피부톤 픽셀 수집
    let totalR = 0, totalG = 0, totalB = 0;
    let pixelCount = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // 간단한 피부톤 필터링 (매우 기본적인 접근)
      if (isSkinTone(r, g, b)) {
        totalR += r;
        totalG += g;
        totalB += b;
        pixelCount++;
      }
    }
    
    if (pixelCount === 0) return null;
    
    // 평균 피부톤 계산
    return {
      r: Math.round(totalR / pixelCount),
      g: Math.round(totalG / pixelCount),
      b: Math.round(totalB / pixelCount)
    };
  };

  // 선택 영역 정규화 (음수 너비/높이 처리)
  const normalizeSelection = (selection) => {
    let { x, y, width, height } = selection;
    
    if (width < 0) {
      x += width;
      width = Math.abs(width);
    }
    
    if (height < 0) {
      y += height;
      height = Math.abs(height);
    }
    
    return { x, y, width, height };
  };

  // 피부톤 필터링
  const isSkinTone = (r, g, b) => {
    // RGB를 YCbCr 색 공간으로 변환
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    
    // 넓은 범위의 피부톤 감지 (다양한 인종 포함)
    return (
      y > 80 && 
      cb > 77 && cb < 135 && 
      cr > 130 && cr < 180
    );
  };

  // 색상 세트 변경 핸들러
  const handleColorSetChange = (set) => {
    setColorSet(set);
    setSelectedColor(null);
  };

  // 특정 색상 선택 핸들러
  const handleColorSelect = (color) => {
    setSelectedColor(color);
    simulateColorDraping(color.hex);
  };

  // 컬러 드레이핑 시뮬레이션
  const simulateColorDraping = (colorHex) => {
    if (!faceSelection || !selectionCanvasRef.current) return;
    
    const canvas = simulationCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 원본 이미지 로드
    const img = new Image();
    img.src = imageUrl;
    
    img.onload = () => {
      // 캔버스 크기 설정
      canvas.width = 400;
      canvas.height = 400 * (img.height / img.width);
      
      // 원본 이미지 그리기
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // 선택 영역 정규화
      const selection = normalizeSelection(faceSelection);
      
      // 선택한 색상으로 얼굴 주변에 컬러 드레이핑 효과 추가
      const padding = 20;
      
      // 상단 컬러 블록
      ctx.fillStyle = colorHex;
      ctx.fillRect(
        selection.x - padding, 
        selection.y - padding * 3, 
        selection.width + padding * 2,
        padding * 2
      );
      
      // 양쪽 컬러 블록
      ctx.fillRect(
        selection.x - padding * 3, 
        selection.y - padding, 
        padding * 2,
        selection.height + padding * 2
      );
      
      ctx.fillRect(
        selection.x + selection.width + padding, 
        selection.y - padding, 
        padding * 2,
        selection.height + padding * 2
      );
      
      // 하단 컬러 블록
      ctx.fillRect(
        selection.x - padding, 
        selection.y + selection.height + padding, 
        selection.width + padding * 2,
        padding * 2
      );
      
      // 얼굴 영역 표시
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.strokeRect(selection.x, selection.y, selection.width, selection.height);
      
      // 시뮬레이션 이미지 URL 업데이트
      setSimulationImageUrl(canvas.toDataURL());
    };
  };

  // 얼굴 영역에서 피부톤 분석 및 퍼스널컬러 판정
  const analyzeFace = () => {
    if (!faceSelection) {
      alert("얼굴 영역을 먼저 선택해주세요.");
      return;
    }
    
    setAnalyzing(true);
    
    // 피부톤 추출
    const skinTone = extractSkinTone();
    
    if (!skinTone) {
      alert("피부톤을 추출할 수 없습니다. 다른 영역을 선택해주세요.");
      setAnalyzing(false);
      return;
    }
    
    // HSV 변환
    const { h, s, v } = rgbToHsv(skinTone.r, skinTone.g, skinTone.b);
    
    // 웜/쿨 점수 계산 (0 = 중립, 양수 = 웜, 음수 = 쿨)
    let warmthScore = calculateWarmthScore(skinTone.r, skinTone.g, skinTone.b, h);
    
    // 계절 및 타입 판정
    const { season, type } = determinePersonalColor(warmthScore, s, v);
    
    const personalColor = `${season}_${type}`;
    
    // 결과 저장
    setResult({
      colorCode: personalColor,
      colorInfo: personalColors[personalColor],
      palette: seasonPalettes[season],
      skinTone: skinTone,
      debug: {
        hue: Math.round(h),
        saturation: Math.round(s * 100),
        value: Math.round(v * 100),
        warmthScore: Math.round(warmthScore),
        season: season
      }
    });
    
    setAnalyzing(false);
  };

  // RGB를 HSV로 변환
  const rgbToHsv = (r, g, b) => {
    r /= 255;
    g /= 255;
    b /= 255;
    
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    
    // 색조(Hue) 계산
    let h = 0;
    if (delta !== 0) {
      if (max === r) {
        h = ((g - b) / delta) % 6;
      } else if (max === g) {
        h = (b - r) / delta + 2;
      } else {
        h = (r - g) / delta + 4;
      }
    }
    
    h = h * 60;
    if (h < 0) h += 360;
    
    // 채도(Saturation)와 명도(Value) 계산
    const s = max === 0 ? 0 : delta / max;
    const v = max;
    
    return { h, s, v };
  };

  // 웜/쿨 점수 계산
  const calculateWarmthScore = (r, g, b, hue) => {
    let score = 0;
    
    // 색조(Hue) 기반 점수
    if ((hue >= 0 && hue < 30) || (hue >= 330 && hue <= 360)) {
      // 빨강 영역
      score += 30;
    } else if (hue >= 30 && hue < 90) {
      // 노랑 영역
      score += 50;
    } else if (hue >= 90 && hue < 150) {
      // 초록 영역
      score -= 10;
    } else if (hue >= 150 && hue < 210) {
      // 청록/시안 영역
      score -= 40;
    } else if (hue >= 210 && hue < 270) {
      // 파랑 영역
      score -= 60;
    } else if (hue >= 270 && hue < 330) {
      // 보라/마젠타 영역
      score -= 30;
    }
    
    // RGB 관계 분석
    if (r > g && g > b) {
      // 전형적인 웜톤 패턴
      score += 20;
    }
    
    if (b > r) {
      // 전형적인 쿨톤 패턴
      score -= 30;
    }
    
    // R/B 비율 분석
    const rbRatio = r / (b || 1);
    if (rbRatio < 1.0) {
      score -= 20;
    } else if (rbRatio > 1.5) {
      score += 20;
    }
    
    return score;
  };

  // 퍼스널컬러 판정
  const determinePersonalColor = (warmthScore, saturation, value) => {
    let season, type;
    
    // 웜톤/쿨톤 판정 (임계값 ±10)
    const isWarm = warmthScore > 10;
    const isCool = warmthScore < -10;
    const isNeutral = !isWarm && !isCool;
    
    // 밝기 판정
    const isBright = value > 0.7;
    const isDark = value < 0.5;
    const isMedium = !isBright && !isDark;
    
    // 채도 판정
    const isVivid = saturation > 0.5;
    const isMuted = saturation < 0.3;
    const isMediumSat = !isVivid && !isMuted;
    
    // 계절 판정
    if (isWarm) {
      if (isBright || (isMedium && isVivid)) {
        season = 'SPRING';
      } else {
        season = 'AUTUMN';
      }
    } else if (isCool) {
      if (isBright || (isMedium && !isVivid)) {
        season = 'SUMMER';
      } else {
        season = 'WINTER';
      }
    } else {
      // 중립적인 경우, 밝기와 채도로 판단
      if (isBright) {
        season = value > 0.8 ? 'SPRING' : 'SUMMER';
      } else {
        season = saturation > 0.4 ? 'WINTER' : 'AUTUMN';
      }
    }
    
    // 타입 판정
    if (season === 'SPRING') {
      if (isVivid && isBright) {
        type = 'BRIGHT';
      } else if (isBright && !isVivid) {
        type = 'LIGHT';
      } else if (warmthScore > 40) {
        type = 'WARM';
      } else {
        type = 'TRUE';
      }
    } else if (season === 'SUMMER') {
      if (isBright) {
        type = 'LIGHT';
      } else if (isMuted) {
        type = 'SOFT';
      } else if (warmthScore < -40) {
        type = 'COOL';
      } else {
        type = 'TRUE';
      }
    } else if (season === 'AUTUMN') {
      if (isDark) {
        type = 'DEEP';
      } else if (isMuted) {
        type = 'SOFT';
      } else if (warmthScore > 40) {
        type = 'WARM';
      } else {
        type = 'TRUE';
      }
    } else if (season === 'WINTER') {
      if (isDark) {
        type = 'DEEP';
      } else if (isVivid) {
        type = 'BRIGHT';
      } else if (warmthScore < -40) {
        type = 'COOL';
      } else {
        type = 'TRUE';
      }
    }
    
    return { season, type };
  };

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center">퍼스널컬러 분석 시스템 2.0</h1>
      
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-md p-6 mb-6 flex flex-col items-center">
        <div className="mb-4 w-full max-w-md">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
            ref={fileInputRef}
          />
          <button
            onClick={triggerFileInput}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded transition duration-200"
          >
            이미지 선택
          </button>
        </div>
        
        {imageUrl && (
          <div className="w-full flex flex-col items-center">
            <div className="flex justify-center mb-4 flex-col items-center">
              <p className="text-lg font-medium mb-2">
                얼굴 영역 선택:
                {!faceSelection && " (얼굴이 있는 영역을 드래그하여 선택해주세요)"}
              </p>
              
              <div className="relative">
                <canvas
                  ref={selectionCanvasRef}
                  className="border border-gray-300 rounded"
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                />
                
                {faceSelection && !selectionMode && (
                  <div className="mt-2 flex gap-2">
                    <button
                      onClick={startFaceSelection}
                      className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-1 px-3 rounded text-sm"
                    >
                      다시 선택
                    </button>
                    
                    <button
                      onClick={analyzeFace}
                      disabled={analyzing}
                      className={`font-medium py-1 px-3 rounded text-sm ${
                        analyzing
                          ? 'bg-gray-400 cursor-not-allowed text-white'
                          : 'bg-green-500 hover:bg-green-600 text-white'
                      }`}
                    >
                      {analyzing ? '분석 중...' : '선택 영역 분석하기'}
                    </button>
                  </div>
                )}
                
                {!faceSelection && !selectionMode && (
                  <button
                    onClick={startFaceSelection}
                    className="mt-2 bg-blue-500 hover:bg-blue-600 text-white font-medium py-1 px-3 rounded text-sm"
                  >
                    얼굴 영역 선택하기
                  </button>
                )}
              </div>
            </div>
            
            {faceSelection && (
              <div className="mt-4 w-full">
                <h2 className="text-xl font-semibold mb-3">컬러 드레이핑 테스트</h2>
                <p className="text-gray-700 mb-2">
                  다양한 색상을 선택하여 얼굴 주변에 배치해보세요. 어울리는 색상이 퍼스널컬러를 판단하는 데 도움이 됩니다.
                </p>
                
                <div className="mb-4 flex gap-3">
                  <button
                    onClick={() => handleColorSetChange('warm')}
                    className={`py-1 px-3 rounded ${
                      colorSet === 'warm'
                        ? 'bg-orange-500 text-white'
                        : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                    }`}
                  >
                    웜톤 색상
                  </button>
                  <button
                    onClick={() => handleColorSetChange('cool')}
                    className={`py-1 px-3 rounded ${
                      colorSet === 'cool'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                    }`}
                  >
                    쿨톤 색상
                  </button>
                  <button
                    onClick={() => handleColorSetChange('neutral')}
                    className={`py-1 px-3 rounded ${
                      colorSet === 'neutral'
                        ? 'bg-gray-700 text-white'
                        : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                    }`}
                  >
                    중립 색상
                  </button>
                </div>
                
                <div className="flex flex-wrap gap-2 mb-4">
                  {drapingColors[colorSet].map((color, index) => (
                    <button
                      key={index}
                      onClick={() => handleColorSelect(color)}
                      className={`w-16 h-16 rounded-lg ${
                        selectedColor === color ? 'ring-4 ring-blue-500' : ''
                      }`}
                      style={{ backgroundColor: color.hex }}
                      title={color.name}
                    />
                  ))}
                </div>
                
                {selectedColor && (
                  <div className="mb-4">
                    <p className="text-gray-800 mb-2">
                      <span className="font-medium">{selectedColor.name}</span>
                      {selectedColor.season && (
                        <span className="ml-2 text-sm text-gray-600">
                          ({selectedColor.season.split('_')[0]} 계열)
                        </span>
                      )}
                    </p>
                    
                    <div className="relative">
                      <canvas
                        ref={simulationCanvasRef}
                        className="hidden"
                      />
                      
                      {simulationImageUrl && (
                        <img
                          src={simulationImageUrl}
                          alt="Color Draping Simulation"
                          className="border border-gray-300 rounded max-w-full"
                        />
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
      
      {result && (
        <div className="w-full max-w-4xl bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4 text-center">분석 결과</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">당신의 퍼스널컬러:</h3>
              <div className="text-2xl font-bold text-center mb-2">{result.colorInfo.name}</div>
              <p className="text-gray-700 mb-4">{result.colorInfo.characteristics}</p>
              
              <div className="mb-4">
                <h4 className="font-medium mb-2">피부톤 색상:</h4>
                <div
                  className="w-16 h-16 rounded-full mx-auto mb-1"
                  style={{ backgroundColor: `rgb(${result.skinTone.r}, ${result.skinTone.g}, ${result.skinTone.b})` }}
                ></div>
                <div className="text-center text-sm text-gray-500">
                  RGB({result.skinTone.r}, {result.skinTone.g}, {result.skinTone.b})
                </div>
              </div>
              
              <div className="mb-4">
                <h4 className="font-medium mb-2">색상 분석 상세:</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-gray-100 p-2 rounded">
                    <span className="font-medium">색조(Hue):</span> {result.debug.hue}°
                  </div>
                  <div className="bg-gray-100 p-2 rounded">
                    <span className="font-medium">채도(Saturation):</span> {result.debug.saturation}%
                  </div>
                  <div className="bg-gray-100 p-2 rounded">
                    <span className="font-medium">명도(Value):</span> {result.debug.value}%
                  </div>
                  <div className="bg-gray-100 p-2 rounded">
                    <span className="font-medium">웜/쿨 점수:</span> {result.debug.warmthScore}
                    <span className="ml-1 text-xs">
                      ({result.debug.warmthScore > 10 ? '웜톤' : result.debug.warmthScore < -10 ? '쿨톤' : '중립'})
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">추천 컬러 팔레트:</h3>
                <div className="grid grid-cols-5 gap-2">
                  {result.palette.map((color, index) => (
                    <div key={index} className="flex flex-col items-center">
                      <div
                        className="w-12 h-12 rounded-full mb-1"
                        style={{ backgroundColor: color }}
                      ></div>
                      <span className="text-xs text-gray-500">{color}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">메이크업 & 패션 팁:</h3>
                <div className="text-sm text-gray-700 p-3 bg-gray-50 rounded">
                  {result.colorCode.startsWith('SPRING') && (
                    <div>
                      <p className="mb-2"><strong>메이크업 팁:</strong> 따뜻하고 밝은 색상을 선택하세요. 코랄, 피치, 아이보리, 골드 톤의 메이크업이 어울립니다.</p>
                      <p><strong>패션 팁:</strong> 밝은 옐로우, 오렌지, 코랄, 아이보리, 베이지 계열의 의상이 잘 어울립니다. 선명하고 따뜻한 톤을 선택하세요.</p>
                    </div>
                  )}
                  {result.colorCode.startsWith('SUMMER') && (
                    <div>
                      <p className="mb-2"><strong>메이크업 팁:</strong> 부드러운 파스텔톤과 블루 베이스 색상을 선택하세요. 라벤더, 로즈, 소프트 핑크 메이크업이 어울립니다.</p>
                      <p><strong>패션 팁:</strong> 라벤더, 블루, 소프트 핑크, 그레이 계열의 의상이 잘 어울립니다. 채도가 낮고 부드러운 톤을 선택하세요.</p>
                    </div>
                  )}
                  {result.colorCode.startsWith('AUTUMN') && (
                    <div>
                      <p className="mb-2"><strong>메이크업 팁:</strong> 자연스럽고 따뜻한 어스톤 색상을 선택하세요. 테라코타, 황토색, 올리브 메이크업이 어울립니다.</p>
                      <p><strong>패션 팁:</strong> 카멜, 머스타드, 올리브 그린, 테라코타, 초콜릿 브라운 계열의 의상이 잘 어울립니다. 깊고 따뜻한 톤을 선택하세요.</p>
                    </div>
                  )}
                  {result.colorCode.startsWith('WINTER') && (
                    <div>
                      <p className="mb-2"><strong>메이크업 팁:</strong> 선명하고 차가운 색상을 선택하세요. 퓨어 화이트, 블랙, 실버, 핑크 및 블루 계열의 메이크업이 잘 어울립니다.</p>
                      <p><strong>패션 팁:</strong> 블랙, 화이트, 로얄 블루, 퍼플, 크림슨 레드 계열의 의상이 잘 어울립니다. 선명하고 차가운 톤을 선택하세요.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonalColorAnalyzer;