//
// 심화 4-2. 상속
//

// 예제 1. 동물과 개 클래스
// 부모 클래스 : Animal
class Animal {
    constructor(name) {
        this.name = name;
    }
    speak() {
        console.log(`${this.name}이(가) 소리를 냅니다.`);
    }
}

// 자식 클래스 : Dog (Animal 클래스 상속)
class Dog extends Animal {
    constructor(name, breed) {
        super(name); // 부모 클래스의 생성자 호출하여 name 설정
        this.breed = breed;
    }

    // 부모 클래스의 메서드를 재정의 (오버라이딩)
    speak() {
        console.log(`${this.name} (품종 : ${this.breed})이(가) 멍멍 소리를 냅니다.`);
    }
}

// Animal 클래스를 상속받은 Dog 클래스의 인스턴스 생성
const myDog = new Dog('바둑이', '진돗개');

// 부모 클래스에서 상속받은 메서드 호출
myDog.speak(); // 바둑이 (품종 : 진돗개)이(가) 멍멍 소리를 냅니다.

// 예제 2. Animal 클래스를 부모로 하는 자식 클래스들
// 부모 클래스 : Animal
class Animal {
    constructor(name) {
        this.name = name;
    }
    speak() {
        console.log(`${this.name}이(가) 소리를 냅니다.`);
    }
}

// 자식 클래스 : Dog (Animal 클래스 상속)
class Dog extends Animal {
    constructor(name, breed) {
        super(name); // 부모 클래스의 생성자 호출하여 name 설정
        this.breed = breed;
    }

    // 부모 클래스의 메서드를 재정의 (오버라이딩)
    speak() {
        console.log(`${this.name} (품종 : ${this.breed})이(가) 멍멍 소리를 냅니다.`);
    }
}

// 자식 클래스 : Cat (Animal 클래스 상속)
class Cat extends Animal{
    // 부모 클래스의 메서드를 재정의 (오버라이딩)
    speak() {
        console.log(`${this.name}이(가) 야옹 소리를 냅니다.`);
    }
}

// 자식 클래스 : Bird (Animal 클래스를 상속받음)
class Bird extends Animal {
    // 부모 클래스의 메서드를 재정의 (오버라이딩)
    speak() {
        console.log(`${this.name}이(가) 짹짹 소리를 냅니다.`);
    }
}

// Animal 클래스의 인스턴스 생성
const initAnimal = new Animal('강아지');
initAnimal.speak();

// Animal 클래스를 상속받은 Dog, Cat, Bird 클래스의 인스턴스 생성
const myDog = new Dog('바둑이', '진돗개');
const myCat = new Cat('나비');
const myBird = new Bird('참새');

// 각각의 speak 메서드 호출
myDog.speak(); // 바둑이 (품종 : 진돗개)이(가) 멍멍 소리를 냅니다.
myCat.speak(); // 나비이(가) 야옹 소리를 냅니다.
myBird.speak(); // 참새이(가) 짹짹 소시를 냅니다.

// 예제 3. Dog 클래스를 부모로 하는 자식 클래스
// 자식 클래스 : GuardDog (Dog 클래스를 상속 받음)
class GuardDog extends Dog {
    constructor(name, breed, trainingLevel) {
        super(name, breed); // Dog 클래스의 생성자를 호출하여 name과 breed를 설정
        this.trainingLevel = trainingLevel; // GuardDog만의 고유 속성
    }

    // 부모 클래스의 메서드를 재정의 (오버라이딩)
    speak() {
        console.log(`${this.name} (품종 : ${this.breed}, 훈련 레벨 : ${this.trainingLevel})이(가) 경계 소리를 냅니다.`);
    }

    // GuardDog만의 새로운 메소드
    guard() {
        console.log(`${this.name}이(가) 경계를 섭니다!`);
    }
}

// GuardDog 클래스의 인스턴스 생성
const myGuardDog = new GuardDog('백구', '시베리안 허스키', '고급');

// GuardDog의 메서드 호출
myGuardDog.speak(); // 백구 (품종 : 시베리안 허스키, 훈련 레벨 : 고급)이(가) 경계 소리를 냅니다.
myGuardDog.guard(); // 백구이(가) 경계를 섭니다!

//
// 심화 5. 접근자 프로퍼티와 은닉
//

// 예제 1. 객체 리터럴에서의 접근자 프로퍼티
const inventory = {
    quantity : 10, // 실제 개수

    // Getter : 개수를 그대로 반환
    get totalQuantity() {
        return this.quantity;
    },

    // Setter : 개수를 설정(음수가 되지 않도록 처리)
    set totalQuantity(value) {
        if(value < 0) {
            console.log("개수는 음수가 될 수 없습니다.");
        } else {
            this.quantity = value;
        }
    }
};

// Getter 사용
console.log(inventory.totalQuantity); // 10
// Setter 사용
inventory.totalQuantity = 15;
console.log(inventory.totalQuantity); //15
//잘못된 개수 설정 시도
inventory.totalQuantity = -5; // 개수는 음수가 될 수 없습니다.
console.log(inventory.totalQuantity); // 15

// 예제 2. 클래스에서의 접근자 프로퍼티
class Inventory {
    constructor(quantity) {
        this.quantity = quantity; // 실제 개수
    }

    // Getter : 개수를 그대로 반환
    get totalQuantity() {
        return this.quantity;
    },

    // Setter : 개수를 설정(음수가 되지 않도록 처리)
    set totalQuantity(value) {
        if(value < 0) {
            console.log("개수는 음수가 될 수 없습니다.");
        } else {
            this.quantity = value;
        }
    }
};

const inventory = new Inventory(10);

// Getter 사용
console.log(inventory.totalQuantity); // 10
// Setter 사용
inventory.totalQuantity = 15;
console.log(inventory.totalQuantity); //15
//잘못된 개수 설정 시도
inventory.totalQuantity = -5; // 개수는 음수가 될 수 없습니다.
console.log(inventory.totalQuantity); // 15


// 예제 3. 캡슐화
class Student {
    // private 필드 정의
    #studentName = '';
    #studentID = 0;

    constructor(studentName, studentID) {
        this.#studentName = studentName;
        this.#studentID = studentID;
    }

    // Getter : studentName을 읽을 때 호출
    get name() {
        return this.#studentName;
    }

    // Setter : studentName을 설정할 때 호출
    set name(newName) {
        this.#studentName = newName;
    }

    // Getter : studentID를 읽을 때 호출
    get id() {
        return this.#studentID;
    }

    // Setter : studentID를 설정할 때 호출
    set id(newId) {
        this.#studentID = newId;
    }

    // private 필드에 접그하여 출력하는 메서드
    displayInfo() {
        console.log(`학생 이름 : ${this.name}, 학번 : ${this.id}`);
    }
}

// Student 클래스의 인스턴스 생성
const student1 = new Student('이영희' , 2023001);

// Getter를 통해 private 필드에 접근
console.log(student1.name); // 이영희
// Setter을 통해 privat 필드의 값을 변경
student1.name = '김철수';
console.log(student1.name); // 김철수

// Getter와 Setter를 통해 학번 변경 및 접근
console.log(student1.id); // 2023001
student1.id = 20230002;
console.log(student1.id); // 2023002

// 클래스 내부의 메서드를 통해 private 필드를 출력
student1.displayInfo(); // 학생 이름 : 김철수, 학번 : 2023002

console.log(student1.#name); // privat 접근 시 오류
console.log(student1['#name']); // undefined 반환

//
// 심화 6. 프로퍼티 어트리뷰트
//

// 예제 1. 데이터 프로퍼티 어트리뷰트 설정
const person = {};

// 데이터 프로퍼티 'name'을 정의
Object.defineProperty(person, 'name', {
    value : 'Alice',
    writable : false, // 값이 수정 불가능하게 설정
    enumerable : true, // 열거 가능하게 설정
    configurable : false // 프로퍼티 삭제 또는 속성 변경 불가능하게 설정
});

console.log(person.name); // Alice

person.name = 'Bob'; // writable : false 이므로 값 수정 안됨
console.log(person.name); // Alice

for(let key in person) {
    console.log(key); // name
}

// 'name' 프로퍼티 삭제 시도 (configurable : false 이므로 삭제 안됨)
delete person.name;
console.log(person.name); // Alice

// 예제 2. 접근자 프로퍼티 어트리뷰트 설정
const product = {
    _price : 100, // 실제 가격을 저장하는 내부 변수
    // 접근자 프로퍼티 'price' 정의
    get price() {
        return `$${this._price}`; // 가격을 문자열로 반환
    },
    set price(value) {
        if(value > 0) {
            this._price = value; // 값이 0보다 큰 경우에만 가격 설정
        } else {
            console.log('가격은 0보다 커야 합니다.');
        }
    }
};

console.log(product.price); // $100

product.price = 150;
console.log(product.price); // $150

product.price = -50; // 가격은 0보다 커야 합니다.
console.log(product.price); // $150

//
// 심화 7. 스프레드
//

// 예제 1. 배열에서의 스프레드 문법 사용
// 배열의 복사
const arr1 = [1, 2, 3];
const arr2 = [...arr1]; // arr1의 요소를 복사하여 arr2에 할당
console.log(arr2); // [1, 2, 3]

// 배열의 결합
const arr3 = [4, 5, 6];
const combineArray = [...arr1, ...arr3]; // arr1과 arr3을 결합
console.log(combineArray); // [1, 2, 3, 4, 5, 6]

// 함수 호출에서의 사용
function sum(a, b, c) {
    return a+b+c;
}
const numbers = [1, 2, 3];
console.log(sum(...numbers)); // 6

// 예제 2. 객체에서의 스프레드 문법 사용
// 객체의 복사
const obj1 = { name : "Alice", age : 25 };
const obj2 = { ...obj1 }; // obj1의 속성을 복사하여 obj2에 할당
console.log(obj2); // { name : "Alice", age : 25 }

// 객체의 병합
const obj3 = { job : "Developer" };
const combinedObj = { ...obj1, ...obj3 }; // obj1과 obj3 병합
console.log(combinedObj); // { name : "Alice", age : 25, job : "Developer" }

// 객체의 속성 덮어쓰기
const obj4 = { name : "Bob", age : 30 };
const updatedObj = { ...obj1, ...obj4 }; // obj1의 속성들이 obj4의 속성들로 변경
console.log(updatedObj); // { name : "Bob", age : 30}

// 예제 3. 리액트 참고
const state = {
    id: "hong30",
    name: "홍길동",
    age: 30
};

// react에서는 이렇게 사용 안함
state.age = 20;
state.des = "유저 정보입니다.";

// react에서 사용하는 형태
setState({
    ...state,
    state.age = 20;
});

//
// 심화 8. 디스트럭쳐링
//

// 예제 1. 배열 디스트럭쳐링
// 기본 배열 디스트럭쳐링
const numbers = [1, 2, 3];
const [first, second, third] = numbers;
console.log(first); // 1
console.log(second); // 2
console.log(third); // 3

// 일부 요소만 디스트럭쳐링
const [one, , three] = numbers; // 두 번째 요소는 무시
console.log(one); // 1
console.log(three); // 3

// 기본값 설정
const [a, b, c = 10, d = 20] = [1, 2];
console.log(a); // 1
console.log(b); // 2
console.log(c); // 10
console.log(d); // 20

const [e, f = 2 g, h = 20] = [1, , 10];
console.log(e); // 1
console.log(f); // 2
console.log(g); // 10
console.log(h); // 20

// 예제 2. 객체 디스트럭쳐링
// 기본 객체 디스트럭쳐링
const person = {
    name : "Alice",
    age : 25,
    job : "Developer"
};

const { name, age, job } = person;
console.log(name); // Alice
console.log(age); // 25
console.log(job); // Developer

// 다른 이름으로 변수 할당
const { name : personName, age : personAge } = person;
console.log(personName); // Alice
console.log(personAge); // 25

// 기본값 설정
const { name : n, age : a, hobby = "Reading" } = person;
console.log(n); // Alice
console.log(a); // 25
console.log(hobby); // Reading


// 예제 3. 함수 매개변수에서의 디스트럭쳐링
// 함수의 매개변수에서 객체 디스트럭쳐링 사용
function printPerson({ name, age }) {
    console.log(`이름 : ${name}, 나이 : ${age}`);
}

const user = { name : "Bob", age : 30 };
printPerson(user); // 이름 : Bob, 나이 : 30

//
// 심화 9. 표준 내장 객체
//

// 예제 1. 기본 자료형과 객체 자료형의 차이
// 기본 자료형
let number = 273;
let string = '안녕하세요';
let boolean = true;

console.log(typeof number); // number
console.log(typeof string); // string
console.log(typeof boolean); // boolean

// 객체 자료형
let number = new Number(273);
let string = new String('안녕하세요');
let boolean = new Boolean(true);

console.log(typeof number); // object
console.log(typeof string); // object
console.log(typeof boolean); // object

// 기본 자료형, 객체 자료형 속성과 메서드
let stringA = '음료, 1800원';
console.log(stringA.split(',')); // 음료 1800원
let stringB = new String('음료, 1800원');
console.log(stringB.split(',')); // 음료 1800원

// 기본 자료형 속성, 메서드 추가 불가
let primitiveNumer = 123;
primitiveNumer.method = function() {
    return 'Primitive Method';
}
console.log(primitiveNumer.method()); // 메서드 실행시 에러 발생

// 프로토타입
let primitiveNumer = 456;
Number.prototype.method = function() {
    return 'Primitive Method';
}
console.log(primitiveNumer.method()); // Primitive Method

// 예시 2. 주요 표준 내장 객체 - Number 객체
let numberFromLiteral = 123;
let numberFromConstructor = new Number(123);

// Number 객체 메서드
let num = 12345.6789;
// toExponential() : 지수 표기법 - 숫자를 소수점 이하 자리와 10의 거듭제곱으로 표현
console.log(num.toExponential()); // 1.23456789e+4
console.log(num.toExponential(2)); // 1.23e+4
console.log(num.toExponential(4)); // 1.2346e+4
// toFixed() : 고정 소수점 표기범 - 소수점 이하 자릿수를 정확히 맞출때
console.log(num.toFixed(0)); // 12346
console.log(num.toFixed(2)); // 12345.68
console.log(num.toFixed(5)); // 12345.67890
// toPrecision() : 지정한 전체 자릿수로 변환
console.log(num.toPrecision(2)); // 1.2e+4
console.log(num.toPrecision(4)); // 1.235e+4
console.log(num.toPrecision(7)); // 1.2345.68

// Number 클래스의 속성
class Test {};

Test.property = 123;
Test.method = function() {
    return "This is a method.";
}

console.log(Test.property); // 123
console.log(Test.method()); // This is a method

// Number 클래스 정적 속성
// Number.MAX_VALUE : js에서 표현 가능한 가장 큰 숫자 값, 이 이상은 Infinity로 표현
console.log(Number.MAX_VALUE); // 1.7976931348623157e+308

// Number.MIN_VALUE : js에서 표현 가능한 가장 작은 양수 값, 이보다 작은 숫자는 0으로 간주
console.log(Number.MIN_VALUE); // 5e-324

// Number.NaN : 숫자가 아닌 값
console.log(Number.NaN); // NaN
console.log( 0 / 0 ); // NaN

// Number.POSITIVE_INFINITY : 양의 무한대 값
console.log(Number.POSITIVE_INFINITY); // Infinity
console.log( 1 / 0 ); // Infinity

// Number.NEGATIVE_INFINITY : 음의 무한대 값
console.log(Number.NEGATIVE_INFINITY); // -Infinity
console.log( -1 / 0 ); // -Infinity

// 예제 2. 주요 표준 내장 객체 - String 객체
let stringFromLiteral = '안녕하세요';
let stringFromConstructor = new String('안녕하세요');

// String 객체 메서드
let string = 'abc';
string.toUpperCase();
console.log(string); // abc

console.log(string.toUpperCase()); // ABC

string = string.toUpperCase();
console.log(string); // ABC

// 메서드 체이닝
let string = 'Hello World';

string = string.toLowerCase(); // 대문자->소문자
string = string.replace(' ', '|'); // 첫 번째 인자를 두 번째 인자로 변경
let array = string.split('|'); // '|'기반으로 문자열 분해

console.log(string); // hello|world
console.log(arrrayy); [ "hello", "world" ]

let array2 = string.toLowerCase().replace(' ', '|').split('|');

console.log(array2); // [ "hello", "world" ]

// 예제 3. 주요 표준 내장 객체 - Date 객체
let now = new Date();
let specificDate = new Date('2023-12-25');
let specificDate2 = new Date(2023, 11, 25);
let dateFromTimestamp = new Date(1609459200000);

console.log(now); // 2025-08-11
console.log(specificDate); // 2023-12-25
console.log(specificDate2); // 2023-12-25
console.log(dateFromTimestamp); // 2023-12-25

// 예제 4. 주요 표준 내장 객체 - Math 객체
// 0이상 1미만의 랜덤한 소수 생성
let randomNumber = Math.random();
console.log("랜덤한 소수 : ", randomNumber);

// 1부터 100까지의 랜덤한 정수 생성
let randomInt = Math.floor(Math.random() * 100)+1;
console.log("1부터 100까지의 랜덤한 정수 : ", randomInt);

let num = 7.6;

let rounded = Math.round(num);
console.log("반올림 : ", rounded); // 8

let ceiled = Math.ceil(num);
console.log("올림 : ", ceiled); // 8

let floored = Math.floor(num);
console.log("내림 : ", floored); // 7

// 예제 5. 주요 표준 내장 객체 - Array 객체
// 배열 생성
let fruits = ["Apple", "Banana", "Cherry"];
let numbers = new Array(10);
let moreFruits = new Array("Apple", "Banana", "Cherry");
let emptyArray = [];

// push, pop, splice : 배열 조작 메서드
let fruits = ["Apple", "Banana"];

// 1. push() 메서드 : 배열의 끝에 새로운 요소 추가
fruits.push("Cherry");
console.log(fruits); // ["Apple", "Banana", "Cherry"]

// 여러 요소를 한 번에 추가
fruits.push("Date", "Elderberry");
console.log(fruits); // ["Apple", "Banana", "Cherry", "Date", "Elderberry"]

// 2. pop() 메서드 : 배열의 마지막 요소를 제거하고 반환
let lastFruit = fruits.pop();
console.log(lastFruit); // Elderberry
console.log(fruits); // ["Apple", "Banana", "Cherry", "Date"]

// 3. splice() 메서드 : 특정 위치에 요소 추가/삭제/교체(시작위치, 삭제갯수, 요소들...)
fruits.splice(1, 0, "Fig", "Grape"); // 배열 [1]부터 "Fig", "Grape" 요소를 추가
console.log(fruits); // ["Apple", "Fig", "Grape", "Banana", "Chery", "Date"]

let removedFruits = fruits.splice(2,2); // 배열 [2]부터 2개 요소 삭제
console.log(removedFruits); // ["Grape", "Banana"]
console.log(fruits); // ["Apple", "Fig", "Cherry", "Date"]

fruits.splice(1, 1, "Honeydew");
console.log(fruits); // ["Apple", "Honeydew", "Cherry", "Date"]

// includes : 배열 검색 및 검사 메서드.(특성 요소, 특정 위치)
let fruits = ["Apple", "Banana", "Cherry", "Date"];

// 1. 배열에 특정 요소가 포함되어 있는지 확인
let hasBanana = fruits.includes("Banana");
console.log(hasBanana); // true

// 2. 배열에 특정 요소가 포함되어 있지 않은 경우
let hasGrape = fruits.includes("Grape");
console.log(hasGrape); // false

// 3. 특정 위치부터 검색
let hasCherryAfterIndex2 = fruits.includes("Cherry", 2);
console.log(hasCherryAfterIndex2); // true

let hasAppleAfterIndex2 = fruits.includes("Apple", 2);
console.log(hasAppleAfterIndex2); // false

// sort, reverse, concat, slice : 배열 정렬 및 변형 메서드
let fruits = ["Banana", "Apple", "Cherry", "Date"];

// 1. sort() : 배열을 사전순으로 정렬
fruits.sort();
console.log("Sorted : ", fruits); // Sorted : (4) ["Apple", "Banana", "Cherry", "Date"]

// 2. reverse() : 배열의 요소 순서를 역순으로 정렬
fruits.reverse();
console.log("Reversed : ", fruits); // Reversed : (4) ["Date", "Cherry", "Banana", "Apple"]

// 3. join() : 배열의 모든 요소를 연결해 하나의 문자열로 만듦
let fruitString = fruits.join(", ");
console.log("Joined : ", fruitString); // Joined : Date, Cherry, Banana, Apple

// 4. concat() : 배열과 다른 배열 또는 값을 결합하여 새로운 배열을 만듦
let moreFruits = ["Elderberry", "Fig"];
let allFruits = fruits.concat(moreFruits);
console.log("Concatenated : ", allFruits); // Concatenated : (6) ["Date", "Cherry", "Banana", "Apple", "Elderberry", "Fig"]

// 5. slice() : 배열의 일부를 선택하여 새로운 배열을 반화
let slicedFruits = allFruits.slice(1, 4);
console.log("Sliced : ", slicedFruits); // Sliced : (3) ["Cherry", "Banana", "Apple"]

// forEach, map, filter : 콜백 함수와 함께 사용하는 메서드
let numbers = [1, 2, 3, 4, 5];

// 1. forEach() : 각 요소에 대한 함수 실행
console.log("Using forEach : ");
numbers.forEach(function(num) {
    console.log(num * 2); // 2, 4, 6, 8, 10
});

// 2. map() : 각 요소에 대해 함수 실행한 결과로 새로운 배열 생성
let doubledNumbers = numbers.map(function(num) {
    return num * 2;
});
console.log("Using map : ", doubledNumbers); // Using map : [2, 4, 6, 8, 10]

// 3. filter() : 각 요소에 대해 함수 실행하여 true를 반환하는 요소로 새로운 배열 생성
let evenNumbers = numbers.filter(function(num) {
    return num % 2 === 0;
});
console.log("Using filter : ", evenNumbers); // Using Filter : [2, 4]

// 예제 6. 주요 표준 내장 객체 - Object 클래스
// Object 클래스 생성
let obj1 = {
    key1 : "value1",
    key2 : "value2"
};

let obj2 = new Object();
obj2.key1 = "value1";
obj2.key2 = "value2";

let proto = {greet : function() { console.log("Hello!"); }};
let obj3 = Object.create(proto);
obj3.name = "John";

// Object.assign() : 파괴적 메서드
const target = { a : 1, b : 2};
const source = { b : 4, c : 5};

const returnedTarget = Object.assign(target, source);

console.log(returnedTarget); // { a: 1, b : 4, c : 5}
console.log(target); // { a : 1, b : 4, c : 5}

// Object.keys(), Object.values(), Object.entries()
const person = {
    name : "Alice",
    age : 25,
    occupation : "Engineer"
};

// 1. Object.keys() : 객체의 모든 키를 배열로 반환
const keys = Object.keys(person);
console.log("Keys : ", keys); // Keys : ["name", "age", "occupation"]

// 2. Object.values() : 객체의 모든 값을 배열로 반환
const values = Object.values(person);
console.log("Values : ", values); // Values : ["Alice", 25, "Engineer"]

// 3. Object.entries() : 객체의 키-값 쌍을 배열의 배열로 반환
const entries = Object.entries(person);
console.log("Enteries : ", entries); // Enteries : [["name", "Alice"], ["Age", 25], ["occupation", "Engineer"]]

// Object.preventExtenstions() : 객체가 더 이상 확장되지 않도록. 새로운 속성 추가 x
const person = {
    name : "Alice",
    age : 25
};

Object.preventExtensions(person);

person.occupation = "Engineer";
console.log(person.occupation); // undefined

console.log(Object.isExtensible(person)); // false

// Object.isSealed() : 객체 봉인 확인. 봉인된 객체는 새로운 속성 추가, 기존 속성 변경 가능하나 기존 속성 제거 불가
const person = {
    name : "Alice",
    age : 25
};

Object.seal(person);

console.log(Object.isSealed(person)); // true

person.age = 26;
console.log(person.age); // 26

delete person.name;
console.log(person.name); // "Alice"

// Object.freeze() : 객체 동결. 속성 추가, 제거, 수정 불가능
const person = {
    name : "Alice",
    age : 25
};

Object.freeze(person);

person.age = 26;
console.log(person.age); // 25

delete person.name;
console.log(person.name); // Alice

person.occupation = "Engineer";
console.log(person.occupation); // undefined

console.log(Object.isFrozen(person)); // true

//
// 심화 10. JSON
//

// 예제 1. JSON 예제
// 1. JSON.stringify() : JSON 문자열로 변환
const person = {
    name : "Alice",
    age : 25,
    job : "Developer"
};

const jsonString = JSON.stringify(person);

console.log(jsonString); // {"name" : "Alice", "age" : 25, "job" : "Developer"}

// 2. JSON.parse() : JSON 문자열을 자바스크립트 객체로 변환
const jsonString = '{"name" : "Alice", "age" : 25, "job" : "Developer"}';

const jsonObject = JSON.parse(jsonString);

console.log(jsonObject); // {"name" : "Alice", "age" : 25, "job" : "Developer"}
console.log(jsonObject.name); // Alice

//
// 심화 11. Symbol
//

// 예제 1. Symbol 기본
// Symbol 생성
const sym1 = Symbol();
const sym2 = Symbol('description'); // 설명을 가진 Symbol 생성
const sym3 = Symbol('description'); // 같은 설명을 가진 Symbol이라도 서로 다름

console.log(sym1 === sym2); // false
console.log(sym2 === sym3); // false

// Symbol을 객체의 프로퍼티 키로 사용
const sym = Symbol('uniqueKey');

const obj = {
    [sym] : 'value'
};

console.log(obj[sym]); // value

for (let key in obj) {
    console.log(key); // 출력 안됨(Symbol 프로퍼티는 열거되지 않음)
}

console.log(Object.keys(obj)); // 빈 배열(Symbol키는 Object.keys로 접근 불가)

// 예제 2. Symbol 접근 방법
// 1. 대괄호 표기법 : Symbol키에 접근
const sym = Symbol('mySymbol');
const obj = {
    [sym] : 'value'
};
console.log(obj[sym]); // value

// 2. Object.getOwnPropertySymbols() 메서드 : 객체의 모든 Symbol 프로퍼티 키를 배열로 반환
const sym1 = Symbol('symbol1');
const sym2 = Symbol('symbol2');
const obj = {
    [sym1] : 'value1',
    [sym2] : 'value2'
};
const symbols = Object.getOwnPropertySymbols(obj);
console.log(symbols); // [Symbol(symbol1), Symbol(symbol2)]

console.log(obj[symbols[0]]); // value1
console.log(obj[symbols[1]]); // value2

// 3. Reflect.ownKeys() 메서드 : 객체의 모든키를 배열로 반환
const sym = Symbol('symbolKey');
const obj = {
    [sym] : 'value',
    normalKey : 'normalValue'
};

const keys = Reflect.ownKeys(obj);
console.log(keys); // ["normalKey", Symbol(symbolKey)]

console.log(obj[keys[0]]); // normalValue
console.log(obj[keys[1]]); // value

//
// 심화 12. 이터러블과 제너레이터
//

// 예제 1. Set 사용
const mySet = new Set();

mySet.add(1);
mySet.add(2);
mySet.add(2); // 중복된 값은 무시
mySet.add(3);

console.log(mySet.size); // 3

console.log(mySet.has(2)); // true
mySet.delete(2);
console.log(mySet.has(2)); // false

for(let value of mySet) {
    console.log(value); // 1, 3
}

// 예제 2. Map 사용
const myMap = new Map();

const keyObj = {};
const keyFunc = function() {};
const keyString = 'a string';

myMap.set(keyString, "value associatted with 'a string'");
myMap.set(keyObj, "value associated with keyObj");
myMap.set(keyFunc, "value associated with keyFunc");

console.log(myMap.get(keyString)); // value associatted with 'a string'
console.log(myMap.get(keyObj)); // value associatted with keyObj
console.log(myMap.get(keyFunc)); // value associatted with keyFunc

console.log(myMap.size); // 3

for(let[key, value] of myMap) {
    console.log(key, value); 
    // a string value associatted with 'a string'
    // {} 'value associated with keyObj'
    // ƒ () {} 'value associated with keyFunc'
}