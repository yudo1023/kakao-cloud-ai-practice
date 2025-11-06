//
// 기초 3. 표현식과 문장
//

console.log('Hello World!');    //Hello World!

let data = 0;
console.log('data type is : ' + typeof data);   //data type is : number

console.log('Hello World!');    // console.log('Hello World!');

//
// 기초 4. 변수와 상수
//

// 예제 1. let, const
let age = 25;
console.log(age); // 25

age = 30;
console.log(age); // 30

const PI = 3.14;
console.log(PI); // 3.14

PI = 3.14159; // 오류 발생 : const로 선언된 상수는 값 변경 불가

let c = 1;
let c = 2; // 오류 발생 : 이미 만든 이름의 변수/상수는 다시 생성 불가

const A = 1;
console.log(A); // 1

const B; // 오류 발생 : 상수는 선언만 불가능
let c; // 변수는 선언만 가능

//
// 기초 5. 자료형과 연산자
//

// 예제 1. 기본 자료형
// 숫자(number)
let age = 25;
let pi = 3.14;

// 문자열(string)
let name = "Alice";
let greeting = "Hello, World!";

let hong = "홍길동";
console.log(`안녕하세요, ${hong}입니다.`); // 안녕하세요, 홍길동입니다.

let message = '그는 "이건 \'예제\'입니다"라고 말했다.'; // 작은따옴표 포함 \'
console.log(message); // 그는 "이건 '예제'입니다"라고 말했다.

let message = "그녀가 '이건 \"예제\"입니다'라고 말했다."; // 큰따옴표 포함 \"
console.log(message); // 그녀가 '이건 "예제"입니다'라고 말했다.

let message = "안녕하세요,\n자바스크립트 세계에 오신 것을 환영합니다!"; // 줄바꿈 포함 \n
console.log(message);
// * 출력결과 *
// 안녕하세요,
// 자바스크립트 세계에 오신 것을 환영합니다!

let message = "이름\t나이\t직업\n홍길동\t30\t프로그래머"; // 탭 포함 \t
console.log(message);
// * 출력결과 *
// 이름     나이    직업
// 홍길동    30     프로그래머

let path = "C:\\사용자\\문서\\파일.txt"; // 백슬래시 포함 \\
console.log(path); // C:\사용자\문서\파일.txt

// 불리언(boolean)
let isStudent = true;
let isGraduated = false;

const boolTrue = 1 < 2; // true
const boolFalse = 1 > 2; // false

// ! 기호를 변수 앞에 붙여 참 거짓 값을 반전시킬 수 있습니다.
!isStudent // false
!isGraduated // true

// 예제 2. 연산자
// 산술 연산자
