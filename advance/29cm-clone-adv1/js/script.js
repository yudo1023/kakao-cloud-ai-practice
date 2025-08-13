// script_plain_dom.js  — script2.js 스타일(순수 DOM API)로 변환 버전

// 1) 서브메뉴 데이터 (원본 그대로)
const submenuData = {
  women: [
    { title: "의류", items: ["해외브랜드","단독","상의","바지","셋업","점프수트","원피스","스커트","아우터","니트웨어","언더웨어","홈웨어","파티룩/행사복"] },
    { title: "가방", items: ["해외브랜드","단독","보스턴백","웨이스트백","기타가방","캐리어/여행가방","숄더백","토트백","크로스백","에코/캔버스백","백팩","클러치","가방 액세서리"] },
    { title: "신발", items: ["해외브랜드","단독","스니커즈","플랫 슈즈","부츠","로퍼","힐/펌프스","샌들","슬리퍼","신발 액세서리"] },
    { title: "액세서리", items: ["해외브랜드","단독","모자","아이웨어","헤어 액세서리","기타 액세서리","양말","벨트","시계","지갑/카드케이스","파우치","스카프/카라","넥타이","머플러","장갑"] },
    { title: "주얼리", items: ["단독","귀걸이","목걸이","반지","발찌","팔찌","주얼리 세트","주얼리 보관함","브로치/펜던트"] },
  ],
  men: [
    { title: "의류", items: [ "해외브랜드","단독","언더웨어","아우터","상의","하의","홈웨어","셋업","이너웨어","니트웨어"] },
    { title: "가방", items: [ "해외브랜드","단독","보스턴백","캐리어/여행가방","크로스백","토트백","웨이스트백","백팩","숄더백","랩탑백","에코/캔버스백","클러치","기타 가방","가방 액세서리"] },
    { title: "신발", items: [ "해외브랜드","단독","스니커즈","기능화","구두","부츠","로퍼","샌들","신발 액세서리","슬리퍼"] },
    { title: "액세서리", items: [ "해외브랜드","단독","모자","아이웨어","기타 액세서리","양말","벨트","시계","지갑/카드케이스","파우치","스카프","넥타이","머플러","장갑"] },
    { title: "주얼리", items: [ "목걸이","귀걸이","반지","팔찌"] },
  ],
  interior: [
    { title: "가구/인테리어", items: [ "해외브랜드", "단독", "홈패브릭", "가구", "홈데코", "침구", "조명", "아트/디자인", "스테이셔너리", "홈프레그런스", "가드닝", "책/음반", "선물하기"] },
  ],
  kitchen: [
    { title: "주방/생활", items: [ "단독", "저장용기/도시락", "냄비/팬/솥", "조리도구", "컵/잔/텀블러", "그릇/커트러리", "테이블데코", "주방잡화", "주방수납/정리", "생활수납/정리", "생필품", "욕실", "홈프래그런스", "기타생활", "반려동물", "선물하기"] },
  ],
  electronics: [
    { title: "가전", items: [ "영상가전", "주방가전", "세탁기/건조기", "청소기", "생활가전", "계절가전", "이미용가전", "건강가전", "리퍼브" ] },
  ],
  digital: [
    { title: "컴퓨터/디지털", items: [ "저장장치", "음향기기", "모바일/웨어러블", "모바일", "액세서리", "PC/노트북", "게임", "사진", "자동차용품/장난감", "스마트모빌리티" ] },
  ],
  beauty: [
    { title: "뷰티", items: [ "해외브랜드", "단독", "선물세트", "스킨케어", "선케어", "헤어케어", "메이크업", "바디케어", "프래그런스", "뷰티소품/기기", "맨즈뷰티", "건강식품" ] },
  ],
  food: [
    { title: "푸드", items: [ "이너뷰티/식단관리", "선물세트", "잼/꿀/오일", "음료", "가공식품", "신선/냉장" ] },
  ],
  leisure: [
    { title: "레저", items: [ "캠핑", "요가/필라테스", "등산/하이킹", "골프", "수영", "러닝", "피트니스", "여행", "테니스", "자전거" ] },
  ],
  kids: [
    { title: "유아/아동", items: ["유아 용품","수유 용품","유아식/분유","외출 용품","임산부/태교용품","베이비","남아 의류","여아 의류","모자","가방","잡화","신발","유아동 식기","유아동 침구","유아동 가구","토이/교육","선물하기"] },
  ],
  culture: [
    { title: "컬쳐", items: ["티켓"] }
  ],
  earth: [
    { title: "어스", items: ["더 나은 소재","비건","동물권 존중","사회적 가치"] }
  ],
};

document.addEventListener("DOMContentLoaded", function () {
  var headerNavMenu = document.querySelector(".header-nav-menu");
  var menuWrap = document.querySelector(".header-menu");
  var submenuBox = document.getElementById("submenu-container");
  if (!menuWrap || !submenuBox || !headerNavMenu) return;

  var openKey = null;

  // 서브메뉴 구성
  function buildSubmenu(key) {
    var sections = submenuData[key];
    if (!sections) return null;

    var mega = document.createElement("div");
    mega.setAttribute("class", "mega");
    mega.setAttribute("role", "region");
    mega.setAttribute("aria-label", key);

    var grid = document.createElement("div");
    grid.setAttribute("class", "mega-grid");

    for (var i = 0; i < sections.length; i++) {
      var sec = sections[i];

      var section = document.createElement("section");
      section.setAttribute("class", "mega-section");

      var h2 = document.createElement("h2");
      h2.setAttribute("class", "mega-title");
      h2.appendChild(document.createTextNode(sec.title));

      var ul = document.createElement("ul");
      ul.setAttribute("class", "mega-list");

      for (var j = 0; j < sec.items.length; j++) {
        var it = sec.items[j];

        var li = document.createElement("li");
        li.setAttribute("class", "mega-item");

        var a = document.createElement("a");
        a.setAttribute("href", "/" + key + "/" + encodeURIComponent(it));
        a.appendChild(document.createTextNode(it));

        li.appendChild(a);
        ul.appendChild(li);
      }

      section.appendChild(h2);
      section.appendChild(ul);
      grid.appendChild(section);
    }

    mega.appendChild(grid);
    return mega;
  }

  function open(key) {
    if (openKey === key) return;
    close();

    var content = buildSubmenu(key);
    if (!content) return;

    submenuBox.appendChild(content);
    submenuBox.classList.add("active");
    openKey = key;

    var selector = '.header-menu-item[data-menu="' + key + '"] .header-menu-link';
    var current = menuWrap.querySelector(selector);
    if (current) current.setAttribute("aria-expanded", "true");
  }

  function close() {
    if (openKey) {
      var prevSelector = '.header-menu-item[data-menu="' + openKey + '"] .header-menu-link';
      var prev = menuWrap.querySelector(prevSelector);
      if (prev) prev.setAttribute("aria-expanded", "false");
    }
    submenuBox.classList.remove("active");
    while (submenuBox.firstChild) {
      submenuBox.removeChild(submenuBox.firstChild);
    }
    openKey = null;
  }

  // 이벤트 바인딩
  menuWrap.addEventListener("pointerover", function (e) {
    var li = e.target.closest(".header-menu-item[data-menu]");
    if (!li || !menuWrap.contains(li)) return;
    open(li.dataset.menu);
  });

  menuWrap.addEventListener("click", function (e) {
    var link = e.target.closest(".header-menu-item[data-menu] .header-menu-link");
    if (!link) return;
    e.preventDefault();
    var li = link.closest(".header-menu-item[data-menu]");
    open(li.dataset.menu);
  });

  headerNavMenu.addEventListener("pointerleave", function () {
    close();
  });

  // 바깥 클릭 시 닫기
  document.addEventListener("click", function (e) {
    if (!headerNavMenu.contains(e.target)) close();
  });

});