document.addEventListener("DOMContentLoaded", function(){
    // DOM 요소 접근(정의)
    const header = document.getElementsByClassName("header")[0];
    const nav = document.getElementsByClassName("navigation")[0];
    const nav_ul = document.getElementsByTagName("ul")[0];
    const first_btn = document.getElementsByClassName("btn")[0];

    // DOM 요소 생성(속성)
    let newAttribute = document.createAttribute("style");
    newAttribute.value = "color : black";
    header.setAttributeNode(newAttribute);

    // DOM 요소 생성(자식 노드)
    // appendChild 첫번째 방법
    let newList = document.createElement("li");
    let newContent = document.createTextNode("새로운 메뉴");
    
    newList.appendChild(newContent);
    nav_ul.appendChild(newList);

    // appendChild 두번째 방법
    //let newContent2 = "<li>새로운 메뉴</li>";
    //nav_ul.appendChild(newContent2);

    // 이벤트 리스너
    first_btn.addEventListener("click", function () {       // html에서 <button class="btn" onClick="alert("버튼이 클릭이되었습니다.")">버튼1</button>
        alert("버튼이 클릭되었습니다.")
    });

    first_btn.removeEventListener("click", function () {    // addEventListener를 생성하면 반드시 removeEventListner도 작성
        alert("버튼이 클릭되었습니다.")
    });
});