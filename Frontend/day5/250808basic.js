//
// 기초 10. 예외처리
//

// try-catch 구조
try {
    // 오류가 발생할 가능성이 있는 코드
} catch (error) {
    // 오류가 발생했을 때 실행할 코드
}

// finally 블록
try {
    // 오류가 발생할 가능성이 있는 코드
} catch (error) {
    // 오류가 발생했을 때 실행할 코드
} finally {
    // 오류 발생 여부와 관계없이 항상 실행되는 코드
}

// 예제 : 기본적인 try-catch 나눗셈 예제
function divide(a,b) {
    try {
        if (b === 0) {
            throw new Error("Division by zero is not allowed"); // 강제로 예외를 발생시킴
        }
        let result = a / b;
        console.log(`Result: ${result}`);
    } catch (error) {
        console.log(`Error: ${error.message}`);
    }
}

divide(10, 2); // 정상적인 출력 : Result: 5
divide(10, 0); // 오류 발생 : Error: Division by zero is not allowed

// 예제 : finally 블록 사용
function readFile(file) {
    try {
        // 파일을 읽는 작업 (실제로는 이 부분에 파일 읽기 코드가 읽을 수 있음)
        console.log(`Reading file: ${file}`);
        if (!file) {
            throw new Error("File not found");
        }
    } catch (error) {
        console.log(`Error: ${error.message}`);
    } finally {
        console.log("Cleaning up resources...");
    }
}

readFile("data.txt"); // 정상적인 파일 읽기 시나리오
readFile("");         // 오류 발생 시나리오

// 실행 결과 예측
function 함수() {
    try {
        return;
        console.log("try 위치");
    } catch {
        console.log("catch 위치");
    } finally {
        console.log("finally 위치");
    }
}
함수() // finally 위치

//
// 기초 11. 블록문과 스코프
//

// 예제 1 : 전역 스코프와 지역 스코프
let globalVar = "이것은 전역 변수입니다.";

function myFunction() {
    let localVar = "이것은 지역 변수입니다.";
    console.log(globalVar); // 전역 변수에 접근 가능
    console.log(localVar);  // 지역 변수에 접근 가능
}

myFunction();

console.log(globalVar);     // 전역 변수에 접근 가능
console.log(localVar);      // 오류 발생 : 지역 변수에 접근 불가

// 예제 2 : 블록 스코프
if (true) {
    let blockScopedVar = "이것은 블록 스코프 변수입니다.";
    console.log(blockScopedVar); // 블록 내에서 접근 가능
}

console.log(blockScopedVar);     // 오류 발생 : 블록 밖에서 접근 불가

// 예제 3 : var의 함수 스코프
function varTest() {
    var functionScopedVar = "이것은 함수 스코프 변수입니다.";

    if (true) {
        var functionScopedVar = "이것은 함수 스코프에서 재할당된 변수입니다.";
        console.log(functionScopedVar);
    }
    console.log(functionScopedVar);
}

varTest();

// *출력 결과*
// 이것은 함수 스코프에서 재할당된 변수입니다.
// 이것은 함수 스코프에서 재할당된 변수입니다.

//
// 심화 1-1. 함수와 일급 객체
//

// 1) 함수의 할당
// 예제 1. 함수가 변수에 할당
// 함수를 변수에 할당
const greet = function(name) {
    return "안녕하세요, " + name + "님!";
};
// 할당된 함수를 호출
console.log(greet("홍길동")); // 안녕하세요, 홍길동님!

// 예제 2. 함수가 다른 함수에 할당
function greet(name) {
    return "안녕하세요, " + name + "님!";
}
// greet 함수를 greetAgain 변수에 할당
const greetAgain = greet;
console.log(greetAgain("이순신")); // 안녕하세요, 이순신님!

// 예제 3. 함수가 객체의 속성으로 할당
const person1 = {
    age: 25,
    greet1: function(name) {
        return "안녕하세요, " + name + "님!";
    }
};
console.log(person1.greet1("홍길동")); // 안녕하세요, 홍길동님!

const person2 = {
    name: "이순신",
    position: "장군",
    greet2: function() {
        return "안녕하세요, " + this.name + " " + this.position + "님!";
    }
};
console.log(person2.greet2()); // 안녕하세요, 이순신 장군님!

// 예제 4. 함수가 배열의 요소로 할당
const functions = [
    function(name) { return "안녕하세요, " + name + "님!"; },
    function(name) { return "반갑습니다, " + name + "님!"; }
];
// 배열의 첫 번째 함수를 호출
console.log(functions[0]("홍길동")); // 안녕하세요, 홍길동님!
// 배열의 두 번째 함수를 호출
console.log(functions[1]("이순신")); // 반갑습니다, 이순신님!

// 2) 함수가 다른 함수의 인자로 전달
// 예제 1
// 인자로 전달될 함수(콜백함수)
function greet(name) {
    return "안녕하세요, " + name + "님!";
}
// 함수를 인자로 받는 함수(고차함수)
function processUserInput(callback) {
    const name = "홍길동";
    console.log(callback(name));
}
// 함수 'greet'를 인자로 전달
processUserInput(greet); // 안녕하세요, 홍길동님!

// 예제 2. 고차함수에 콜백함수 전달
// calculate
function add(a, b) {
    return a + b;
}
function substract(a, b) {
    return a - b;
}
function multiply(a, b) {
    return a * b;
}
// evaluate
function isOdd(number) {
    return !!(number % 2);
}
function isPositive(number) {
    return number > 0;
}
function calcAndEval(calc, eval, x, y) {
    return eval(calc(x, y));
}
console.log(
    calcAndEval(add, isOdd, 3, 9),              // false: (3+9=12) 짝수
    calcAndEval(substract, isPositive, 3, 9),   // false: (3-9=-6) 음수
    calcAndEval(multiply, isOdd, 3, 9)          // true: (3*9=27) 홀수
)

// 3)함수가 다른 함수의 결과값으로 반환
// 예제 1
// 함수를 반환하는 함수
function createGreeting(greeting) {
    return function(name) {
        return greeting + ", " + name + "님!";
    };
}
// 함수 createGreeting을 호출하여 함수 반환
const sayHello = createGreeting("안녕하세요");
console.log(sayHello("홍길동")); // 안녕하세요, 홍길동님!

const sayGoodbye = createGreeting("안녕히 가세요");
console.log(sayGoodbye("홍길동")); // 안녕히 가세요, 홍길동님!

// 예제 2
function getIntroFunc(name, formal) {
    return formal
        ? function () {
            console.log(`안녕하십니까, ${name}입니다.`);
        }
        : function () {
            console.log(`안녕하세요~ ${name}라고 해요~`);
        };
}

const hong = getIntroFunc('홍길동', true);
const jeon = getIntroFunc('전우치', false);

hong(); // 안녕하십니까, 홍길동입니다.
jeon(); // 안녕하세요~ 전우치라고 해요~

//
// 심화 1-2. 함수와 매개변수
//

// 예제 1. 함수의 매개변수 갯수를 넘어가는 인수
function add(x, y) {
    return x + y;
}
console.log(
    add(2, 4),      // 6
    add(2, 4, 6),   // 6
    add(2, 4, 6, 8) // 6
);

// 예제 2. 기본값 매개변수
function add(x = 1, y = 3) {
    console.log(`${x} + ${y}`);
    return x + y;
}
console.log(
    add(),      // 4: (1+3)
    add(2),     // 5: (2+3)
    add(2, 4)   // 6: (2+4)
);

// 예제 3. arguments : 함수 내에서 사용가능한 지역 변수
function add(x, y) {
    console.log('1번', arguments);
    console.log('2번', arguments[0]);
    console.log('3번', typeof arguments);
    return x + y;
}
console.log(
    '4번', add(2, 4, 6, 8)
);
// 출력 결과
// 1번 Arguments(4) [2, 4, 6, 8, callee: ƒ, Symbol(Symbol.iterator): ƒ]0: 21: 42: 63: 8callee: ƒ add(x, y)length: 4Symbol(Symbol.iterator): ƒ values()[[Prototype]]: Object
// 2번 2
// 3번 object
// 4번 6

function add(x, y) {
    for (const arg of arguments) {
        console.log(arg);
    }
    return x + y;
}
console.log(add(2,4,6,8));
// 출력 결과
// 2
// 4
// 6
// 8
// 6

function getAverage() {
    let result = 0;
    for (const num of arguments) {
        result += num;
    }
    return result / arguments.length;
}
console.log(
    getAverage(15.99, 25.50, 9.75),
    getAverage(100, 200, 150, 300),
    getAverage(50, 60, 70, 80, 90)
);

// 예제 4. 문자열 변환 함수 결합

//첫번째 방법

function toUpperCase(str) {
    return str.toUpperCase();
}
function addExclamation(str) {
    return str + '!';
}
function reverseString(str) {
    return str.split('').reverse().join('');
}
function surroundWithBrackets(str) {
    return `[${str}]`;
}
// combineStringTransforms 함수 정의
function combineStringTransforms(...transforms) { // (...)나머지 매개변수 이용
    return function (initialString) {
        let result = initialString;
        for(const transform of transforms) {
            if(typeof transform !== 'function') continue;
            result = transform(result);
        }
        return result;
    }
}

// 문자열 변환 함수 결합
const transform1 = combineStringTransforms(toUpperCase, addExclamation);
const transform2 = combineStringTransforms(toUpperCase, reverseString);
const transform3 = combineStringTransforms(addExclamation, reverseString, surroundWithBrackets);
// 함수 사용
console.log(transform1('hello'));
console.log(transform2('hello'));
console.log(transform3('hello'));

// ------------------------------------------------------

// 두번째 방법

// 기본 문자열 변환 함수들 정의
function toUpperCase(str) {
    return str.toUpperCase();
}
function addExclamation(str) {
    return str + '!';
}
function reverseString(str) {
    return str.split('').reverse().join('');
}
function surroundWithBrackets(str) {
    return `[${str}]`;
}
// combineStringTransforms 함수 정의
function combineStringTransforms() {
    return function(initialString) {
        let result = initialString;
        for (var i = 0; i < arguments.length; i++) {
            const transform = arguments[i];
            if (typeof transform !== 'function') continue;
            result = transform(result);
        }
        return result;
    };
}

// 문자열 변환 함수 생성
const transform = combineStringTransforms();

// 함수 사용
console.log(transform('hello', toUpperCase, addExclamation));
console.log(transform('hello', toUpperCase, reverseString));
console.log(transform('hello', addExclamation, reverseString));

//
// 심화 1-3. 함수 더 알아보기
//

// 예제 1. 중첨된 함수의 간단한 예제
function outerFunction() {
    let functionName = "외부 함수";
    console.log(functionName, "입니다.");

    function innerFunction() {
        let functionName = "내부 함수";
        console.log(functionName, "입니다.");
    }
    innerFunction(); // 내부 함수 호출
}
outerFunction(); // 외부 함수 호출

// *출력 결과*
// 외부 함수입니다.
// 내부 함수입니다.

// 예제 2. 재귀함수 예제 : 1부터 n까지의 숫자 합 계산
function sum(n) {
    // 기본 사례: n이 1이면, 합은 1입니다.
    if (n === 1) {
        return 1;
    }
    // 재귀 사례 : n + sum(n-1)을 계산합니다.
    return n + sum(n - 1);
}
console.log(sum(5)); // 15

// 예제 3. 불변성 - 원시 타입
let a = 10;
let b = a; // b에 a의 값(10)이 복사됨
b = 20;
console.log(a); // 10
console.log(b); // 20

// 예제 4. 불변성 - 참조 타임
let person = {
    name: "Alice",
    age: 30
}; // 객체
let numbers = [1, 2, 3, 4, 5]; // 배열
function greet(a, b) {
    a.name = "Jhon"
    a.age = 26
    b[0]++
}
greet(person, numbers);
console.log(person, numbers);

// *출력 결과*
// { name: "Jhon", age: 26 }
// [2, 2, 3, 4, 5]

//
// 심화 2. 객체 더 알아보기
//

// 예제 1. 기본 예제
// 객체 생성
let person = {
    name: "Alice",
    age: 30,
    job: "Developer"
};

console.log(person); // { name: "Alice", age: 30, job: "Developer"}

// 객체 프로퍼티 삭제
delete person.job;

console.log(person); // { name: "Alice", age: 30 }
console.log(person.job); // undefined

// 예제 2. 키의 동적 사용
const item = {
    name: "휴대폰",
    color: "white",
    price: 1000000
};

function addModifyProperty(obj, key, value) {
    obj[key] = value;
}

function deleteProperty(obj, key) {
    delete obj[key];
}

addModifyProperty(item, 'inch', 16);
console.log(item); // { name: "휴대폰", color: "white", price : 1000000, inch: 16 }

addModifyProperty(item, 'price', 750000);
console.log(item); // { name: "휴대폰", color: "white", price : 750000, inch: 16 }

deleteProperty(item, 'color');
console.log(item); // { name: "휴대폰", price : 750000, inch: 16 }

// 예제 3. 알아두면 좋은 문법
const a = 10; b = 20;

const obj1 = {
    a: a,
    b: b
};

console.log(obj1); // { a: 10, b: 20}

const obj2 = { a, b };

console.log(obj2); // { 10, 20 }

function createItem(name, cost, stock) {
    return { name, cost, stock };
}

const item1 = createItem('의자', 25000, 10);
const item2 = createItem('책상', 50000, 5);

console.log(item1, item2); // { name: '의자', cost: 25000, stock: 10 } { name: '책상', cost : 50000, stock : 5 }

// 예제 4. 메서드
// 일반 함수 표현으로 정의된 메서드
const user = {
    firstName: "이순신",
    greet: function(formal) {
        return formal
            ? `안녕하십니까, ${this.firstName}입니다.`
            : `안녕하세요, ${this.firstName}입니다.`;
    }
};

console.log(user.greet(true)); // 안녕하십니까, 이순신입니다.

// 축약된 표현으로 정의된 메서드
const user = {
    firstName: "이순신",
    greet(formal) {
        return formal
            ? `안녕하십니까, ${this.firstName}입니다.`
            : `안녕하세요, ${this.firstName}입니다.`
    }
};

console.log(user.greet(true)); // 안녕하십니까, 이순신입니다.

//
// 심화 3. 생성자 함수
//

// 예제 1. 생성자 함수를 사용한 객체 생성
function Car(model, year) {
    this.model = model;
    this.year = year;
    this.getInfo = function () {
      return `${this.year} ${this.model}`;
    };
  }
  
  const car1 = new Car("Toyota", 2021);
  const car2 = new Car("Honda", 2020);
  
  console.log(car1.getInfo()); // 2021 Toyota
  console.log(car2.getInfo()); // 2020 Honda

// 예제 2. 객체 리터럴
const animal1 = {
    type: "Dog",
    sound: "Woof",
    makeSound: function () {
        return `${this.type} says ${this.sound}!`;
    }
};
console.log(animal1.makeSound()); // Dog says Woof!

// 예제 3. 객체 반환 함수
function createAnimal(type, sound) {
    return {
      type: type,
      sound: sound,
      makeSound: function () {
        return `${this.type} says ${this.sound}!`;
      },
    };
  }
  
  const animal2 = createAnimal("Cat", "Meow");
  const animal3 = createAnimal("Cow", "Moo");
  
  console.log(animal2.makeSound()); // Cat says Meow!
  console.log(animal3.makeSound()); // Cow says Moo!

  // 예제 4. 생성자 함수
  function Animal(type, sound) {
    this.type = type;
    this.sound = sound;
  }
  
  Animal.prototype.makeSound = function () {
    return `${this.type} says ${this.sound}!`;
  };
  
  const animal4 = new Animal("Lion", "Roar");
  const animal5 = new Animal("Sheep", "Baa");
  
  console.log(animal4.makeSound()); // Lion says Roar!
  console.log(animal5.makeSound()); // Sheep says Baa!

  // 예제 4. 
  console.log(animal1, animal1 instanceof Animal); // { type: 'Dog', sound: 'Woof', makeSoun: f } false
  console.log(animal2, animal2 instanceof Animal); // { type: 'Cat', sound: 'Meow', makeSoun: f } false
  console.log(animal3, animal3 instanceof Animal); // { type: 'Cow', sound: 'Moo', makeSoun: f } false
  console.log(animal4, animal4 instanceof Animal); // Animal { type: 'Lion', sound: 'Roar' } true
  console.log(animal5, animal5 instanceof Animal); // Animal { type: 'Sheep', sound: 'Baa' } true

  //
  // 심화 4-1. 클래스
  //

  // 예제 1. 기본 예제
  // 클래스 정의
  class Animal {
    // 생성자 메서드
    constructor(type, sound) {
        this.type = type;
        this.sound = sound;
    }
    // 매서드 정의
    makeSound() {
        return `${this.type} says ${thiis.sound}!`;
    }
  }
  // 클래스 인스턴스 생성
  const dog = new Animal('Dog', 'Woof');
  const cat = new Animal('Cat', 'Meow');

  console.log(dog.makeSound()); // Dog says Woof!
  console.log(cat.makeSound()); // Cat sysa Meow!

  


  