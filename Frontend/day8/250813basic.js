//
// React 기초 1. JSX
//

// 특징 1. 변수와 함수 호출
// 변수 사용
const name = 'Player';
const hello = <h1>Hello, {name}!</h1>;

// 함수 호출 사용
function formatName(user) {
    return user.fristName + ' ' + user.lastName;
}

const user = {
    fristName: 'Harry',
    lastName: 'Potter'
};

const element = <h1>Hello, {formatName(user)}!</h1>;

// 조건부 연산자
const isLogIn = true;
const element = (
    <div>
        {isLogIn ? <h1>Welcome back!</h1> : <h1>Please sign in.</h1>}
    </div>
);

// 특징 2. 속성 전달
// JavaScript 표기법
const element = <div className="my-class">Content</div>;
const label = <label htmlFor="inputId">Enter your name:</label>;

// 속성 값에 JavaScript 표현식 사용
const imageUrl = 'https://example.com/image.png';
const element = <img src={imageUrl} alt="Something's Image"/>;

// 특징 3. 조건부 렌더링
// 삼항 연산자
function Greeting(props) {
    const isLogIn = props.isLogIn;
    return (
        <div>
            {isLogIn ? <h1>Welcome back!</h1> : <h1>Please sign up.</h1>}
        </div>
    );
}

// AND(&&) 연산자
function Notification(props) {
    const unreadNotifications = props.unreadNotifications;
    return (
        <div>
            <h1>Hello!</h1>
            {unreadNotifications.length > 0 && (
                <h2>You have {unreadNotifications.length} unread notification.</h2>
            )}
        </div>
    );
}

// 특징 4. 표현식으로 간주
// 함수에서 JSX 반환
function greetings(user) {
    if(user) {
        return <h1>Hello, {user.name}!</h1>;
    }
    return <h1>Hello, Anonymous.</h1>;
}

// 배열에 JSX 저장
const items = ['Apple', 'Banana', 'Pear'];
const itemElements = items.map((item) =>
    <li key={item}>{item}</li>
);

const element = <ul>{itemElements}</ul>;

// 객체에 JSX 저장
const arrs = [
    { id: 1, name: "홍길동", age: 30 },
    { id: 2, name: "홍길동", age: 30 },
    { id: 3, name: "홍길동", age: 30 },
    { id: 4, name: "홍길동", age: 30 },
    { id: 5, name: "홍길동", age: 30 },
]

const arrElements = arrs.map((arrItem) =>
    <li key={arrItem.id}>{arrItem.name}</li>
);

const element2 = <ul>{arrElements}</ul>;

// 특징 5. 자식 요소
// 여러 자식 요소
const element = (
    <div>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <button>Click here!</button>
    </div>
);

// 중첩된 요소
const element = (
    <div>
        <header>
            <h1>Welcome</h1>
        </header>
        <main>
            <p>Here are main contents.</p>
        </main>
        <footer>
            <p>© 2024 Education</p>
        </footer>
    </div>
);

//
// 기초 2. 엘리먼트 렌더링
//

// 1. 엘리먼트
// JSX로 엘리먼트 생성
const element = <h1>Hello, world!</h1>;

// 생성된 엘리먼트가 JavaScript 객체로 변환
const element = {
    type: 'h1',
    props: {
        children: 'Hello, world!'
    }
};

// 2. 불변성
const element = <h1>Hello, world!</h1>;

// 엘리먼트를 변경하는 대신 새로운 엘리먼트를 생성
const updatedElement = <h1>Hello, element!</h1>;

// 3. 렌더링
// ReactDOM.render()
import React from 'react';
import ReactDOM from 'react-dom';

const element = <h1>Hello, world!</h1>;
ReactDOM.render(element, document.getElementById('root')); // 엘리먼트를 id가 'root'인 DOM노드에 렌더링

// 3. 업데이트
// 3-1. 초기상태
import React from 'react';
import ReactDOM from 'react-dom';

let count = 0;

function render() {
    const element = (
        <div>
            <h1>Counter: {count}</h1>
            <button onClick={increment}>Increment</button>
        </div>
    );
    ReactDOM.render(element, document.getElementById('root'));
}

function increment() {
    count += 1;
    render(); // 새로운 엘리먼틀르 생성하고 다시 렌더링
}

render();

// 3-2. 업데이트 과정
// 초기 렌더링
//<div>
//    <h1>Counter: 0</h1>
//    <button>증가</button>
//</div>

// 3-3. 버튼 클릭
// increment()함수 호출 -> count 1 증가 -> render() 함수 재호출

// 3-4. 새로운 엘리먼트 생성
//<div>
//    <h1>Counter: 1</h1>
//    <button>증가</button>
//</div>

// 3-5. React의 비교 및 업데이트
// h1 태그의 텍스트가 Counter: 0 -> Counter: 1로 변경되었음을 감지

// 3-6. DOM 업데이트
// 변경된 부분만 실제 DOM에 반영. h1 태그의 텍스트만 업데이트

// 4. Virtual DOM
// 중첩 및 구성
const element = (
    <div>
        <h1>Hello, world!</h1>
        <p>This is a paragraph.</p>
        <ul>
            <li>First item</li>
            <li>Secon item</li>
            <li>Third item</li>
        </ul>
    </div>
);
