import React, { useState, useRef, useEffect } from 'react';

const WarmCoolToneAnalyzer = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [faceRect, setFaceRect] = useState(null);
  const [skinSamples, setSkinSamples] = useState([]);
  const [hairSamples, setHairSamples] = useState([]);
  const canvasRef = useRef(null);
  const displayCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // 웜톤/쿨톤 특성
  const toneCharacteristics = {
    warm: {
      skinTone: '노란 편이다',
      bloodVessel: '잘 안보인다',
      sunburn: '까맣게 탄다',
      hairColor: '진한 갈색',
      eyeColor: '색상이 옅고 부드럽다',
      clothesColor: '갈색, 베이지, 코랄, 레드, 블랙',
      blackClothes: '얼굴 윤곽 페이스라인이 또렷하게 살아나는 편',
      whiteClothes: '얼굴이 동동 뜨는 느낌',
      makeupColor: '코랄, 오렌지, 브라운, 베이지 등 옐로우 언더톤 색조 화장이 잘 어울린다'
    },
    cool: {
      skinTone: '붉거나 하얀 편이다',
      bloodVessel: '잘 보인다',
      sunburn: '붉게 탄다',
      hairColor: '딥 블랙',
      eyeColor: '색상이 진하고 채도 높다',
      clothesColor: '블루, 핑크, 연베이지, 아이보리',
      blackClothes: '얼굴빛이 어두워지고 칙칙해지는 편',
      whiteClothes: '얼굴이 안정적으로 어울리는 느낌',
      makeupColor: '핑크, 바이올렛, 모브핑크, 블랙, 오키드, 블루, 화이트 등 블루, 퍼플 언더톤 색조 화장이 잘 어울린다'
    }
  };

  // 추천 컬러 팔레트
  const colorPalettes = {
    warm: [
      { color: '#D4A76A', name: '골드 베이지' },
      { color: '#C38452', name: '캐러멜' },
      { color: '#FF6B35', name: '코랄' },
      { color: '#D9531E', name: '오렌지 레드' },
      { color: '#8B5D33', name: '초콜릿 브라운' },
      { color: '#ECBE7A', name: '머스타드' },
      { color: '#927C60', name: '올리브 브라운' },
      { color: '#C47E5E', name: '테라코타' }
    ],
    cool: [
      { color: '#5D3FD3', name: '라벤더' },
      { color: '#DB7093', name: '핑크' },
      { color: '#4682B4', name: '스틸 블루' },
      { color: '#483D8B', name: '다크 슬레이트 블루' },
      { color: '#BA55D3', name: '오키드' },
      { color: '#C71585', name: '미디엄 바이올렛 레드' },
      { color: '#778899', name: '라이트 슬레이트 그레이' },
      { color: '#E6E6FA', name: '라벤더 블러쉬' }
    ]
  };

  // 이미지 선택 처리
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setImageUrl(URL.createObjectURL(file));
      setResult(null);
      setFaceDetected(false);
      setFaceRect(null);
      setSkinSamples([]);
      setHairSamples([]);
    }
  };

  // 파일 선택창 트리거
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // 이미지 로드 시 처리
  useEffect(() => {
    if (imageUrl) {
      const img = new Image();
      img.src = imageUrl;
      img.onload = () => {
        const canvas = canvasRef.current;
        const displayCanvas = displayCanvasRef.current;
        
        if (!canvas || !displayCanvas) return;
        
        // 분석용 캔버스 설정
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
        
        // 표시용 캔버스 설정
        const maxWidth = 400;
        const scale = maxWidth / img.width;
        displayCanvas.width = maxWidth;
        displayCanvas.height = img.height * scale;
        
        const displayCtx = displayCanvas.getContext('2d');
        displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
        
        // 자동으로 얼굴 감지 시도
        detectFace();
      };
    }
  }, [imageUrl]);

  // 얼굴 영역 감지 (간단한 피부색 기반 감지)
  const detectFace = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // 피부색으로 추정되는 픽셀 찾기
    let skinPixels = [];
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        
        if (isSkinTone(r, g, b)) {
          skinPixels.push({ x, y });
        }
      }
    }
    
    if (skinPixels.length > 0) {
      // 피부 픽셀의 경계 찾기
      const minX = Math.max(0, skinPixels.reduce((min, p) => Math.min(min, p.x), canvas.width));
      const minY = Math.max(0, skinPixels.reduce((min, p) => Math.min(min, p.y), canvas.height));
      const maxX = Math.min(canvas.width, skinPixels.reduce((max, p) => Math.max(max, p.x), 0));
      const maxY = Math.min(canvas.height, skinPixels.reduce((max, p) => Math.max(max, p.y), 0));
      
      // 가로 세로 비율로 얼굴 영역 추정 (얼굴은 대략 1:1.5 비율)
      const width = maxX - minX;
      const height = maxY - minY;
      
      // 충분히 큰 피부색 영역이 있다면 얼굴로 간주
      if (width > canvas.width * 0.1 && height > canvas.height * 0.1) {
        const faceRect = { x: minX, y: minY, width, height };
        setFaceRect(faceRect);
        setFaceDetected(true);
        
        // 표시용 캔버스에 얼굴 영역 표시
        drawFaceRect(faceRect);
        
        // 피부 샘플 추출
        extractSkinSamples(faceRect);
        
        // 머리카락 샘플 추출
        extractHairSamples(faceRect);
      }
    }
  };

  // 얼굴 영역 표시
  const drawFaceRect = (rect) => {
    if (!displayCanvasRef.current) return;
    
    const displayCanvas = displayCanvasRef.current;
    const displayCtx = displayCanvas.getContext('2d');
    
    // 스케일 조정
    const scale = displayCanvas.width / canvasRef.current.width;
    const scaledRect = {
      x: rect.x * scale,
      y: rect.y * scale,
      width: rect.width * scale,
      height: rect.height * scale
    };
    
    // 얼굴 영역 표시
    displayCtx.strokeStyle = '#00FF00';
    displayCtx.lineWidth = 2;
    displayCtx.strokeRect(scaledRect.x, scaledRect.y, scaledRect.width, scaledRect.height);
  };

  // 피부 샘플 추출
  const extractSkinSamples = (rect) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 얼굴 중앙 부분에서 피부 샘플 추출
    const centerX = rect.x + rect.width / 2;
    const centerY = rect.y + rect.height / 2;
    
    const sampleSize = Math.min(rect.width, rect.height) * 0.1;
    const samples = [];
    
    // 얼굴 중앙 부위에서 여러 샘플 추출
    const samplePoints = [
      { x: centerX, y: centerY }, // 중앙
      { x: centerX - sampleSize, y: centerY }, // 왼쪽
      { x: centerX + sampleSize, y: centerY }, // 오른쪽
      { x: centerX, y: centerY - sampleSize }, // 위
      { x: centerX, y: centerY + sampleSize }  // 아래
    ];
    
    for (const point of samplePoints) {
      const sampleData = ctx.getImageData(
        point.x - sampleSize / 2, 
        point.y - sampleSize / 2, 
        sampleSize, 
        sampleSize
      );
      
      const avgColor = getAverageColor(sampleData.data);
      if (isSkinTone(avgColor.r, avgColor.g, avgColor.b)) {
        samples.push(avgColor);
      }
    }
    
    setSkinSamples(samples);
  };

  // 머리카락 샘플 추출
  const extractHairSamples = (rect) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 얼굴 윗부분에서 머리카락 샘플 추출
    const hairY = Math.max(0, rect.y - rect.height * 0.2);
    const sampleSize = rect.width * 0.1;
    const samples = [];
    
    // 헤어라인 부분에서 여러 샘플 추출
    const samplePoints = [
      { x: rect.x + rect.width / 2, y: hairY }, // 중앙
      { x: rect.x + rect.width / 4, y: hairY }, // 왼쪽
      { x: rect.x + rect.width * 3 / 4, y: hairY }  // 오른쪽
    ];
    
    for (const point of samplePoints) {
      const sampleData = ctx.getImageData(
        point.x - sampleSize / 2, 
        point.y - sampleSize / 2, 
        sampleSize, 
        sampleSize
      );
      
      const avgColor = getAverageColor(sampleData.data);
      if (isHairColor(avgColor.r, avgColor.g, avgColor.b)) {
        samples.push(avgColor);
      }
    }
    
    setHairSamples(samples);
  };

  // 피부톤 판별
  const isSkinTone = (r, g, b) => {
    // RGB를 YCbCr 색 공간으로 변환 (피부톤 감지에 더 효과적)
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    
    // YCbCr 색 공간에서 피부톤 범위 확인 (더 넓은 피부톤 범위 포함)
    return (
      y > 80 && 
      cb > 77 && cb < 135 && 
      cr > 130 && cr < 180
    );
  };

  // 머리카락 색상 판별
  const isHairColor = (r, g, b) => {
    // 머리카락 색상 범위 (검은색~갈색 계열)
    const brightness = (r + g + b) / 3;
    return (
      brightness < 150 && // 어두운 색상
      r >= g && r >= b    // 적색 계열이 강한 경향 (갈색, 검은색)
    );
  };

  // 이미지 데이터에서 평균 색상 추출
  const getAverageColor = (data) => {
    let r = 0, g = 0, b = 0;
    let count = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count++;
    }
    
    return {
      r: Math.round(r / count),
      g: Math.round(g / count),
      b: Math.round(b / count)
    };
  };

  // 웜톤/쿨톤 분석 실행
  const analyzeImage = () => {
    if (!faceDetected || skinSamples.length === 0) {
      alert('얼굴을 감지할 수 없습니다. 다른 이미지를 시도해보세요.');
      return;
    }
    
    setAnalyzing(true);
    
    // 피부톤 분석
    const skinAnalysis = analyzeSkinTone(skinSamples);
    
    // 머리카락 분석
    const hairAnalysis = analyzeHairColor(hairSamples);
    
    // 결과 종합
    const warmScore = skinAnalysis.warmScore + hairAnalysis.warmScore;
    const coolScore = skinAnalysis.coolScore + hairAnalysis.coolScore;
    
    const toneType = warmScore > coolScore ? 'warm' : 'cool';
    const confidence = Math.round((Math.max(warmScore, coolScore) / (warmScore + coolScore)) * 100);
    
    // RGB를 HSV로 변환
    const { h, s, v } = rgbToHsv(
      skinAnalysis.avgColor.r,
      skinAnalysis.avgColor.g,
      skinAnalysis.avgColor.b
    );
    
    setResult({
      toneType,
      confidence,
      skinAnalysis,
      hairAnalysis,
      debug: {
        warmScore,
        coolScore,
        hue: Math.round(h),
        saturation: Math.round(s * 100),
        value: Math.round(v * 100),
        yellowRatio: skinAnalysis.yellowRatio.toFixed(2),
        redRatio: skinAnalysis.redRatio.toFixed(2)
      }
    });
    
    setAnalyzing(false);
  };

  // 피부톤 분석
  const analyzeSkinTone = (samples) => {
    if (samples.length === 0) return { warmScore: 0, coolScore: 0, avgColor: { r: 0, g: 0, b: 0 } };
    
    // 평균 피부색 계산
    let totalR = 0, totalG = 0, totalB = 0;
    for (const sample of samples) {
      totalR += sample.r;
      totalG += sample.g;
      totalB += sample.b;
    }
    
    const avgR = totalR / samples.length;
    const avgG = totalG / samples.length;
    const avgB = totalB / samples.length;
    
    // 웜톤/쿨톤 점수 계산
    let warmScore = 0;
    let coolScore = 0;
    
    // 1. 노란 피부톤 vs 붉은/하얀 피부톤
    const yellowRatio = avgG / avgR; // 노란색 경향 (G/R 비율이 높을수록 노란 경향)
    const redRatio = avgR / avgB;    // 붉은색 경향 (R/B 비율이 높을수록 붉은 경향)
    
    if (yellowRatio > 0.85) {
      warmScore += 4; // 노란 경향 강함 (웜톤)
    } else {
      coolScore += 4; // 붉은/하얀 경향 강함 (쿨톤)
    }
    
    // 2. RGB 관계 분석
    if (avgR > avgG && avgG > avgB) {
      // 전형적인 웜톤 패턴
      warmScore += 3;
    }
    
    if (avgR > avgB && avgR - avgB < 30) {
      // 붉은색과 파란색 차이가 적으면 쿨톤 경향
      coolScore += 3;
    }
    
    // 3. 색조 분석
    const { h, s, v } = rgbToHsv(avgR, avgG, avgB);
    
    // 색조(Hue) 기반 분석
    if ((h >= 20 && h <= 50)) {
      // 노란색/황금색 경향 (웜톤)
      warmScore += 3;
    } else if ((h >= 0 && h < 20) || (h > 340 && h <= 360)) {
      // 붉은색 경향 (쿨톤)
      coolScore += 3;
    }
    
    return {
      warmScore,
      coolScore,
      avgColor: { r: Math.round(avgR), g: Math.round(avgG), b: Math.round(avgB) },
      yellowRatio: yellowRatio,
      redRatio: redRatio
    };
  };

  // 머리카락 색상 분석
  const analyzeHairColor = (samples) => {
    if (samples.length === 0) return { warmScore: 0, coolScore: 0, avgColor: { r: 0, g: 0, b: 0 } };
    
    // 평균 색상 계산
    let totalR = 0, totalG = 0, totalB = 0;
    for (const sample of samples) {
      totalR += sample.r;
      totalG += sample.g;
      totalB += sample.b;
    }
    
    const avgR = totalR / samples.length;
    const avgG = totalG / samples.length;
    const avgB = totalB / samples.length;
    
    // 웜톤/쿨톤 점수 계산
    let warmScore = 0;
    let coolScore = 0;
    
    // 갈색 vs 검은색 판별
    const brightness = (avgR + avgG + avgB) / 3;
    const redBrownRatio = avgR / avgB; // R/B 비율이 높을수록 갈색 경향
    
    if (brightness < 60 && avgR < 60 && avgG < 60 && avgB < 60) {
      // 딥 블랙 (쿨톤)
      coolScore += 2;
    } else if (redBrownRatio > 1.2 && brightness < 120) {
      // 갈색 경향 (웜톤)
      warmScore += 2;
    }
    
    return {
      warmScore,
      coolScore,
      avgColor: { r: Math.round(avgR), g: Math.round(avgG), b: Math.round(avgB) }
    };
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

  // 결과 페이지에서 다시 분석 시작
  const handleRestart = () => {
    setResult(null);
    setSelectedImage(null);
    setImageUrl(null);
    setFaceDetected(false);
    setFaceRect(null);
    setSkinSamples([]);
    setHairSamples([]);
  };

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center">이미지 기반 웜톤/쿨톤 분석기</h1>
      
      {!result ? (
        <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
          <div className="mb-4">
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
            <div className="mb-4">
              <div className="mb-4">
                <h2 className="text-lg font-semibold mb-2">선택된 이미지</h2>
                <div className="border border-gray-300 rounded overflow-hidden">
                  <canvas ref={displayCanvasRef} className="w-full h-auto" />
                </div>
                <canvas ref={canvasRef} className="hidden" />
              </div>
              
              {faceDetected ? (
                <div className="mb-4">
                  <p className="text-green-600 font-medium mb-2">✅ 얼굴이 감지되었습니다.</p>
                  
                  <div className="mb-4">
                    <h3 className="text-md font-medium mb-2">피부톤 샘플:</h3>
                    <div className="flex space-x-2">
                      {skinSamples.map((sample, index) => (
                        <div 
                          key={`skin-${index}`}
                          className="w-8 h-8 rounded-full"
                          style={{ backgroundColor: `rgb(${sample.r}, ${sample.g}, ${sample.b})` }}
                          title={`RGB(${sample.r}, ${sample.g}, ${sample.b})`}
                        />
                      ))}
                    </div>
                  </div>
                  
                  {hairSamples.length > 0 && (
                    <div className="mb-4">
                      <h3 className="text-md font-medium mb-2">머리카락 샘플:</h3>
                      <div className="flex space-x-2">
                        {hairSamples.map((sample, index) => (
                          <div 
                            key={`hair-${index}`}
                            className="w-8 h-8 rounded-full"
                            style={{ backgroundColor: `rgb(${sample.r}, ${sample.g}, ${sample.b})` }}
                            title={`RGB(${sample.r}, ${sample.g}, ${sample.b})`}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <button
                    onClick={analyzeImage}
                    disabled={analyzing}
                    className={`w-full font-medium py-2 px-4 rounded transition duration-200 ${
                      analyzing
                        ? 'bg-gray-400 cursor-not-allowed text-white'
                        : 'bg-green-500 hover:bg-green-600 text-white'
                    }`}
                  >
                    {analyzing ? '분석 중...' : '웜톤/쿨톤 분석하기'}
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <p className="text-yellow-600 mb-2">얼굴을 감지 중입니다...</p>
                  <p className="text-sm text-gray-600">
                    얼굴이 감지되지 않는다면, 밝은 조명에서 얼굴이 잘 보이는 사진을 선택해주세요.
                  </p>
                </div>
              )}
            </div>
          )}
          
          <div className="text-sm text-gray-600 mt-4">
            <p className="mb-2">🔍 <strong>사용 방법:</strong></p>
            <ol className="list-decimal pl-5 space-y-1">
              <li>얼굴이 또렷하게 보이는 정면 사진을 선택해주세요.</li>
              <li>자연광에서 촬영된 사진이 가장 정확한 결과를 제공합니다.</li>
              <li>과도한 메이크업이나 필터가 없는 사진이 좋습니다.</li>
              <li>얼굴이 감지되면 '분석하기' 버튼을 클릭하세요.</li>
            </ol>
          </div>
        </div>
      ) : (
        <div className="w-full max-w-2xl bg-white rounded-lg shadow-md p-6">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-2">
              당신의 퍼스널 컬러는
            </h2>
            <div className="text-4xl font-bold mb-4 relative inline-block">
              <span className={`
                ${result.toneType === 'warm' ? 'text-orange-500' : 'text-blue-500'} 
                relative z-10
              `}>
                {result.toneType === 'warm' ? '웜톤' : '쿨톤'}
              </span>
              <div className={`
                absolute -bottom-2 left-0 right-0 h-3 opacity-20
                ${result.toneType === 'warm' ? 'bg-orange-500' : 'bg-blue-500'}
                -z-10 transform -rotate-1
              `}></div>
            </div>
            <p className="text-gray-600 mb-2">
              정확도: {result.confidence}%
            </p>
            <p className="text-sm text-gray-500">
              웜톤 점수: {result.debug.warmScore}, 쿨톤 점수: {result.debug.coolScore}
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">분석 결과 상세</h3>
                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="flex flex-col items-center">
                    <div className="text-sm text-gray-600 mb-1">피부톤</div>
                    <div 
                      className="w-16 h-16 rounded-full"
                      style={{ backgroundColor: `rgb(${result.skinAnalysis.avgColor.r}, ${result.skinAnalysis.avgColor.g}, ${result.skinAnalysis.avgColor.b})` }}
                    />
                    <div className="text-xs text-gray-500 mt-1">
                      RGB({result.skinAnalysis.avgColor.r}, {result.skinAnalysis.avgColor.g}, {result.skinAnalysis.avgColor.b})
                    </div>
                  </div>
                  
                  {result.hairAnalysis.avgColor.r > 0 && (
                    <div className="flex flex-col items-center">
                      <div className="text-sm text-gray-600 mb-1">헤어컬러</div>
                      <div 
                        className="w-16 h-16 rounded-full"
                        style={{ backgroundColor: `rgb(${result.hairAnalysis.avgColor.r}, ${result.hairAnalysis.avgColor.g}, ${result.hairAnalysis.avgColor.b})` }}
                      />
                      <div className="text-xs text-gray-500 mt-1">
                        RGB({result.hairAnalysis.avgColor.r}, {result.hairAnalysis.avgColor.g}, {result.hairAnalysis.avgColor.b})
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="bg-gray-50 p-3 rounded-lg">
                  <h4 className="font-medium mb-2">피부톤 특성:</h4>
                  <ul className="text-sm space-y-1">
                    <li>
                      <span className="font-medium">노란 경향 (YR 비율):</span> {result.debug.yellowRatio}
                      {Number(result.debug.yellowRatio) > 0.85 ? 
                        ' (웜톤 경향)' : ' (쿨톤 경향)'}
                    </li>
                    <li>
                      <span className="font-medium">붉은 경향 (RB 비율):</span> {result.debug.redRatio}
                    </li>
                    <li>
                      <span className="font-medium">색조(Hue):</span> {result.debug.hue}°
                    </li>
                    <li>
                      <span className="font-medium">채도:</span> {result.debug.saturation}%
                    </li>
                    <li>
                      <span className="font-medium">명도:</span> {result.debug.value}%
                    </li>
                  </ul>
                </div>
              </div>
              
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">추천 컬러 팔레트</h3>
                <div className="grid grid-cols-4 gap-2">
                  {colorPalettes[result.toneType].map((color, index) => (
                    <div key={index} className="flex flex-col items-center">
                      <div 
                        className="w-12 h-12 rounded-full mb-1"
                        style={{ backgroundColor: color.color }}
                      ></div>
                      <span className="text-xs text-center">{color.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div>
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-2">
                  {result.toneType === 'warm' ? '웜톤' : '쿨톤'} 특성
                </h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <ul className="text-sm space-y-2">
                    <li>
                      <span className="font-medium">피부톤:</span> {toneCharacteristics[result.toneType].skinTone}
                    </li>
                    <li>
                      <span className="font-medium">혈관이 잘 보이는가:</span> {toneCharacteristics[result.toneType].bloodVessel}
                    </li>
                    <li>
                      <span className="font-medium">선크림을 안발랐을때:</span> {toneCharacteristics[result.toneType].sunburn}
                    </li>
                    <li>
                      <span className="font-medium">자연모 색깔:</span> {toneCharacteristics[result.toneType].hairColor}
                    </li>
                    <li>
                      <span className="font-medium">눈동자 홍채 색상:</span> {toneCharacteristics[result.toneType].eyeColor}
                    </li>
                    <li>
                      <span className="font-medium">즐겨입는 옷 컬러:</span> {toneCharacteristics[result.toneType].clothesColor}
                    </li>
                    <li>
                      <span className="font-medium">블랙 의상을 입었을때:</span> {toneCharacteristics[result.toneType].blackClothes}
                    </li>
                    <li>
                      <span className="font-medium">화이트 의상을 입었을 때:</span> {toneCharacteristics[result.toneType].whiteClothes}
                    </li>
                    <li>
                      <span className="font-medium">색조 메이크업 색상:</span> {toneCharacteristics[result.toneType].makeupColor}
                    </li>
                  </ul>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">
                  {result.toneType === 'warm' ? '웜톤' : '쿨톤'}을 위한 스타일링 팁
                </h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  {result.toneType === 'warm' ? (
                    <div className="text-sm space-y-2">
                      <p><strong>메이크업:</strong> 코랄, 오렌지, 브라운, 베이지 등 옐로우 언더톤 색상의 메이크업이 잘 어울립니다.</p>
                      <p><strong>의상:</strong> 갈색, 베이지, 코랄, 레드, 머스타드, 올리브 등의 따뜻한 계열 색상이 잘 어울립니다.</p>
                      <p><strong>악세서리:</strong> 골드 계열 액세서리가 잘 어울립니다.</p>
                      <p><strong>헤어 컬러:</strong> 골드 브라운, 캐러멜, 구리빛, 레드 브라운 등 따뜻한 느낌의 컬러가 잘 어울립니다.</p>
                    </div>
                  ) : (
                    <div className="text-sm space-y-2">
                      <p><strong>메이크업:</strong> 핑크, 바이올렛, 모브핑크, 블루 계열 등 블루, 퍼플 언더톤 색상의 메이크업이 잘 어울립니다.</p>
                      <p><strong>의상:</strong> 블루, 핑크, 연베이지, 아이보리, 버건디, 그레이 등의 차가운 계열 색상이 잘 어울립니다.</p>
                      <p><strong>악세서리:</strong> 실버 계열 액세서리가 잘 어울립니다.</p>
                      <p><strong>헤어 컬러:</strong> 애쉬 브라운, 블랙, 플래티넘, 블루 블랙 등 차가운 느낌의 컬러가 잘 어울립니다.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 flex justify-center">
            <button
              className="py-2 px-6 border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors"
              onClick={handleRestart}
            >
              다른 이미지로 다시 분석하기
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default WarmCoolToneAnalyzer;