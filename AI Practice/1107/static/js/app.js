// 전역 변수
let isLoggedIn = false;
let allContracts = [];

// HTML 이스케이프
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ========== 인증 관련 ==========
async function checkAuth() {
  try {
    const res = await fetch('/check-auth');
    const data = await res.json();
    isLoggedIn = data.logged_in;

    const authModal = document.getElementById('authModal');
    const appContent = document.getElementById('appContent');
    const userInfo = document.getElementById('userInfo');

    if (isLoggedIn) {
      authModal.classList.remove('show');
      appContent.style.display = 'block';
      userInfo.textContent = `${data.username}님 환영합니다`;
      loadContracts();
    } else {
      authModal.classList.add('show');
      appContent.style.display = 'none';
    }
  } catch (err) {
    console.error('인증 확인 오류:', err);
  }
}

function switchAuthTab(tab) {
  const tabs = document.querySelectorAll('.auth-tab');
  const forms = document.querySelectorAll('.auth-form');

  tabs.forEach(t => t.classList.remove('active'));
  forms.forEach(f => f.classList.remove('active'));

  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`${tab}Form`).classList.add('active');
}

async function handleLogin(e) {
  e.preventDefault();
  const username = document.getElementById('loginUsername').value;
  const password = document.getElementById('loginPassword').value;
  const errorDiv = document.getElementById('loginError');

  errorDiv.classList.remove('show');

  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });

    const data = await res.json();

    if (res.ok) {
      checkAuth();
    } else {
      errorDiv.textContent = data.error || '로그인 실패';
      errorDiv.classList.add('show');
    }
  } catch (err) {
    errorDiv.textContent = '서버 오류가 발생했습니다';
    errorDiv.classList.add('show');
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const username = document.getElementById('registerUsername').value;
  const password = document.getElementById('registerPassword').value;
  const email = document.getElementById('registerEmail').value;
  const errorDiv = document.getElementById('registerError');

  errorDiv.classList.remove('show');

  try {
    const res = await fetch('/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email })
    });

    const data = await res.json();

    if (res.ok) {
      alert('회원가입이 완료되었습니다! 로그인해주세요.');
      switchAuthTab('login');
      // 회원가입 폼 초기화
      document.getElementById('registerForm').reset();
    } else {
      errorDiv.textContent = data.error || '회원가입 실패';
      errorDiv.classList.add('show');
    }
  } catch (err) {
    errorDiv.textContent = '서버 오류가 발생했습니다';
    errorDiv.classList.add('show');
  }
}

async function handleLogout() {
  try {
    await fetch('/logout', { method: 'POST' });
    isLoggedIn = false;
    allContracts = [];
    
    // 모달 표시, 메인 컨텐츠 숨기기
    document.getElementById('authModal').classList.add('show');
    document.getElementById('appContent').style.display = 'none';
    
    // 로그인 탭으로 전환
    switchAuthTab('login');
    
    // 폼 초기화
    document.getElementById('loginForm').reset();
    document.getElementById('registerForm').reset();
    
    // 에러 메시지 숨기기
    document.getElementById('loginError').classList.remove('show');
    document.getElementById('registerError').classList.remove('show');
    
    console.log('✅ 로그아웃 완료');
  } catch (err) {
    console.error('로그아웃 오류:', err);
    alert('로그아웃 중 오류가 발생했습니다.');
  }
}

// ========== OCR 시각화 ==========
function drawOCRResult(imageBase64, ocrData) {
  console.log('\n=== OCR 시각화 시작 ===');
  console.log('이미지 base64 길이:', imageBase64 ? imageBase64.length : 0);
  console.log('OCR 데이터 개수:', ocrData ? ocrData.length : 0);

  if (!imageBase64) {
    console.error('❌ 이미지 데이터가 없습니다');
    return;
  }

  if (!ocrData || ocrData.length === 0) {
    console.warn('⚠️ OCR 데이터가 비어있습니다');
    return;
  }

  const canvas = document.getElementById('ocrCanvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();

  img.onload = function () {
    console.log(`✅ 이미지 로드 완료: ${img.width}x${img.height}`);

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    if (ocrData && ocrData.length > 0) {
      console.log(`🔍 ${ocrData.length}개 OCR 박스 그리기 시작`);

      ocrData.forEach((item, idx) => {
        try {
          const bbox = item.bbox;
          if (!bbox || bbox.length !== 4) {
            console.warn(`[${idx}] 잘못된 bbox:`, bbox);
            return;
          }

          const [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox;
          const minX = Math.min(x1, x2, x3, x4);
          const minY = Math.min(y1, y2, y3, y4);
          const maxX = Math.max(x1, x2, x3, x4);
          const maxY = Math.max(y1, y2, y3, y4);

          const conf = item.confidence || 0;
          let color;
          if (conf > 0.85) {
            color = 'rgba(0, 255, 0, 0.3)';
          } else if (conf > 0.7) {
            color = 'rgba(255, 165, 0, 0.3)';
          } else {
            color = 'rgba(255, 0, 0, 0.3)';
          }

          ctx.fillStyle = color;
          ctx.fillRect(minX, minY, maxX - minX, maxY - minY);

          ctx.strokeStyle = conf > 0.85 ? '#00ff00' : conf > 0.7 ? '#ffa500' : '#ff0000';
          ctx.lineWidth = 2;
          ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

          if (item.text && item.text.trim()) {
            const displayText = `${item.text} (${(conf * 100).toFixed(0)}%)`;
            ctx.font = '12px Arial';
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(minX, minY - 18, ctx.measureText(displayText).width + 4, 18);
            ctx.fillStyle = 'white';
            ctx.fillText(displayText, minX + 1, minY - 4);
          }
        } catch (err) {
          console.error(`[${idx}] 박스 그리기 오류:`, err, item);
        }
      });

      console.log('시각화 완료');

      // 통계 패널
      const highConf = ocrData.filter(d => d.confidence > 0.85).length;
      const medConf = ocrData.filter(d => d.confidence > 0.7 && d.confidence <= 0.85).length;
      const lowConf = ocrData.filter(d => d.confidence <= 0.7).length;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(10, 10, 280, 100);

      ctx.fillStyle = 'white';
      ctx.font = 'bold 16px Arial';
      ctx.fillText('📊 OCR 인식 결과', 20, 35);

      ctx.font = '13px Arial';
      ctx.fillText(`✅ 총 ${ocrData.length}개 텍스트 블록`, 20, 55);

      ctx.fillStyle = '#00ff00';
      ctx.fillText(`🟢 고신뢰도: ${highConf}`, 20, 73);

      ctx.fillStyle = '#ffa500';
      ctx.fillText(`🟠 중신뢰도: ${medConf}`, 150, 73);

      ctx.fillStyle = '#ff6666';
      ctx.fillText(`🔴 저신뢰도: ${lowConf}`, 20, 91);
    } else {
      console.warn('OCR 데이터가 없거나 비어있습니다');
    }

    document.getElementById('ocrVisual').classList.add('show');
    console.log('=== OCR 시각화 완료 ===\n');
  };

  img.onerror = function () {
    console.error('❌ 이미지 로드 실패');
    alert('이미지를 불러올 수 없습니다.');
  };

  img.src = 'data:image/png;base64,' + imageBase64;
}

// ========== 파일 업로드 ==========
function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    if (e.target.files.length > 1) {
      document.getElementById('fileName').textContent = '⚠️ 한 번에 한 파일만 선택해주세요!';
      document.getElementById('submitBtn').disabled = true;
      document.getElementById('fileInput').value = '';
    } else {
      document.getElementById('fileName').textContent = `선택된 파일: ${e.target.files[0].name}`;
      document.getElementById('submitBtn').disabled = false;
    }
  }
}

async function handleUpload(e) {
  e.preventDefault();
  const formData = new FormData(e.target);
  const statusBox = document.getElementById('status');
  const resultBox = document.getElementById('result');
  const submitBtn = document.getElementById('submitBtn');

  statusBox.textContent = "⏳ OCR 분석 중입니다... 잠시만 기다려주세요";
  statusBox.className = "status loading show";
  resultBox.className = "result";
  document.getElementById('ocrVisual').classList.remove('show');
  submitBtn.disabled = true;

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || "서버 오류가 발생했습니다");
    }

    const data = await res.json();

    console.log('========== 서버 응답 데이터 ==========');
    console.log('전체 응답:', data);
    console.log('OCR 데이터 존재:', !!data.ocr_data);
    console.log('OCR 데이터 타입:', typeof data.ocr_data);
    console.log('OCR 데이터 길이:', data.ocr_data ? data.ocr_data.length : 0);
    if (data.ocr_data && data.ocr_data.length > 0) {
      console.log('첫 번째 항목:', data.ocr_data[0]);
      console.log('첫 번째 bbox:', data.ocr_data[0].bbox);
    }
    console.log('이미지 존재:', !!data.image);
    console.log('=====================================');

    statusBox.textContent = "✅ 분석이 완료되었습니다!";
    statusBox.className = "status success show";

    // OCR 시각화
    if (data.image && data.ocr_data) {
      drawOCRResult(data.image, data.ocr_data);
    } else {
      console.warn('⚠️ OCR 시각화 건너뜀 - 이미지 또는 OCR 데이터 없음');
    }

    const displayText = data.text || '📭 추출된 텍스트가 없습니다';

    resultBox.innerHTML = `
      <h3>📊 분석 결과</h3>
      <div style="margin-bottom: 16px;">
        ${data.original_filename ? `<div style="color: #999; font-size: 13px; margin-bottom: 4px;">원본: ${escapeHtml(data.original_filename)}</div>` : ''}
        <strong>저장된 파일명:</strong> ${escapeHtml(data.filename)}<br>
        <strong>카테고리:</strong> <span class="contract-category">${escapeHtml(data.category)}</span>
      </div>
      ${data.summary && Object.keys(data.summary).length > 0 ? `
        <div class="contract-summary">
          <div class="summary-title">📝 주요 내용</div>
          <div class="summary-content">${formatSummary(data.summary)}</div>
        </div>
      ` : ''}
      <pre>${escapeHtml(displayText)}</pre>
    `;
    resultBox.className = "result show";

    e.target.reset();
    document.getElementById('fileName').textContent = '';

    setTimeout(() => loadContracts(), 500);

  } catch (err) {
    statusBox.textContent = "❌ 오류: " + err.message;
    statusBox.className = "status error show";
  } finally {
    submitBtn.disabled = false;
  }
}

// ========== 계약서 목록 ==========
async function loadContracts() {
  if (!isLoggedIn) return;

  try {
    const res = await fetch('/contracts');
    if (res.status === 401) {
      checkAuth();
      return;
    }

    allContracts = await res.json();
    updateStats(allContracts);
    filterContracts();
  } catch (err) {
    console.error('목록 조회 오류:', err);
  }
}

function updateStats(contracts) {
  const stats = {
    total: contracts.length,
    웨딩홀: contracts.filter(c => c.category === '웨딩홀').length,
    헤어메이크업: contracts.filter(c => c.category === '헤어메이크업').length,
    스냅: contracts.filter(c => c.category === '스냅').length,
    드레스: contracts.filter(c => c.category === '드레스').length,
  };

  document.getElementById('statsGrid').innerHTML = `
    <div class="stat-card">
      <div class="stat-number">${stats.total}</div>
      <div class="stat-label">전체 계약서</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">${stats.웨딩홀}</div>
      <div class="stat-label">웨딩홀</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">${stats.헤어메이크업}</div>
      <div class="stat-label">헤어메이크업</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">${stats.스냅}</div>
      <div class="stat-label">스냅</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">${stats.드레스}</div>
      <div class="stat-label">드레스</div>
    </div>
  `;
}

function filterContracts() {
  const searchQuery = document.getElementById('searchInput').value.toLowerCase().trim();
  const category = document.getElementById('filter').value;

  let filtered = allContracts;

  if (category) {
    filtered = filtered.filter(c => c.category === category);
  }

  if (searchQuery) {
    filtered = filtered.filter(c =>
      c.filename.toLowerCase().includes(searchQuery) ||
      (c.text && c.text.toLowerCase().includes(searchQuery)) ||
      c.category.toLowerCase().includes(searchQuery)
    );
  }

  renderContracts(filtered);
}

function renderContracts(contracts) {
  const listElement = document.getElementById('contractList');

  if (contracts.length === 0) {
    listElement.innerHTML = `
      <div class="empty-state">
        <div style="font-size: 48px; margin-bottom: 12px;">📭</div>
        <div style="font-size: 16px; font-weight: 500;">
          아직 저장된 계약서가 없습니다
        </div>
      </div>
    `;
    return;
  }

  listElement.innerHTML = contracts.map((c) => {
    const textPreview = c.text
      ? c.text.substring(0, 200).replace(/\n/g, ' ')
      : '내용 없음';

    return `
    <li class="contract-item">
      <div class="contract-item-header">
        <div class="contract-item-title">${escapeHtml(c.filename)}</div>
        <button class="expand-btn" onclick="toggleFullText(${c.id})">
          전체보기 ▼
        </button>
      </div>
      <div style="margin-bottom: 12px;">
        <span class="contract-category">${escapeHtml(c.category)}</span>
        <span style="color: #999; font-size: 13px; margin-left: 12px;">📅 ${c.upload_time}</span>
      </div>
      <div style="font-size: 13px; color: #666; padding: 12px; background: #f9f9f9; border-radius: 6px;">
        📄 ${escapeHtml(textPreview)}${c.text && c.text.length > 200 ? '...' : ''}
      </div>
      ${c.summary && Object.keys(c.summary).length > 0 ? `
        <div class="contract-summary">
          <div class="summary-title">📝 주요 내용</div>
          <div class="summary-content">${formatSummary(c.summary)}</div>
        </div>
      ` : ''}
      <div class="full-text" id="fulltext-${c.id}">
        <pre>${escapeHtml(c.text || '내용 없음')}</pre>
      </div>
    </li>
  `;
  }).join('');
}

function formatSummary(summary) {
  if (!summary) return '';
  const lines = [];
  
  // 순서대로 표시
  if (summary.company) lines.push(`<div><strong>🏢 업체:</strong> ${escapeHtml(summary.company)}</div>`);
  if (summary.amount) lines.push(`<div><strong>💰 금액:</strong> ${escapeHtml(summary.amount)}</div>`);
  if (summary.date) lines.push(`<div><strong>📅 날짜:</strong> ${escapeHtml(summary.date)}</div>`);
  if (summary.phone) lines.push(`<div><strong>📞 연락처:</strong> ${escapeHtml(summary.phone)}</div>`);
  if (summary.email) lines.push(`<div><strong>📧 이메일:</strong> ${escapeHtml(summary.email)}</div>`);
  if (summary.location) lines.push(`<div><strong>📍 장소:</strong> ${escapeHtml(summary.location)}</div>`);
  if (summary.business_number) lines.push(`<div><strong>🏷️ 사업자번호:</strong> ${escapeHtml(summary.business_number)}</div>`);
  
  return lines.join('');
}

function toggleFullText(id) {
  const element = document.getElementById(`fulltext-${id}`);
  element.classList.toggle('show');
  const btn = event.target;
  btn.textContent = element.classList.contains('show') ? '접기 ▲' : '전체보기 ▼';
}

// ========== 초기화 ==========
window.addEventListener('load', () => {
  checkAuth();

  // 이벤트 리스너 등록
  document.getElementById('fileInput').addEventListener('change', handleFileSelect);
  document.getElementById('uploadForm').addEventListener('submit', handleUpload);
  document.getElementById('searchInput').addEventListener('input', filterContracts);
  document.getElementById('filter').addEventListener('change', filterContracts);
  document.getElementById('loadList').addEventListener('click', loadContracts);
});