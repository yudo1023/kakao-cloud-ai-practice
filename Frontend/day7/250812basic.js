//
// 심화 12. 이터러블과 제너레이터
//

// 예제 3. 이터러블과 이터레이터
const iterable = [1, 2, 3]; // 배열은 기본적으로 이터러블

// 이터레이터를 얻기 위해 Symbol.iterator 메서드 호출
const iterator = iterable[Symbol.iterator]();

console.log(iterator.next()); // { value : 1, done : false }
console.log(iterator.next()); // { value : 2, done : false }
console.log(iterator.next()); // { value : 3, done : false }
console.log(iterator.next()); // { value : undefined, done : true }

// 예제 4. 제너레이터
function* generatorFunction() {
    yield 1;
    yield 2;
    yield 3;
}

const generator = generatorFunction();

console.log(generator.next()); // { value : 1, done : false }
console.log(generator.next()); // { value : 2, done : false }
console.log(generator.next()); // { value : 3, done : false }
console.log(generator.next()); // { value : undefined, done : true }

//
// 심화 13. 옵셔널 체이닝
//

// 예제 1. 객체 속성에 안전하게 접근하기
const user = {
    name: 'Alice',
    address: {
        city: 'Wonderland'
    }
};

console.log(user?.address?.city); // Wonderland
console.log(user?.cotact?.email); // undefined

console.log(user.contact); // undefined
console.log(user.contact.email); // 에러 발생

// 예제 2. 배열 요소에 안전하게 접근하기
const users = [{ name: 'Alice' }, null, { name: 'Bob' }];

console.log(users[0]?.name); // Alice
console.log(users[1]?.name); // undefined
console.log(users[2].name); // Bob

// 예제 3. 함수 또는 메서드 호출에 사용하기
const user = {
    greet: function() {
        return 'Hello';
    }
};

console.log(user.greet?.()); // Hello
console.log(user.nonExistentMethod?.()); // undefined

// 예제 4. 옵셔널 체이닝과 null 병합 연산자
const user = null;

const name = user?.name ?? 'Anonymous';
console.log(name); // Anonymous

//
// 심화 14. 렉시컬과 클로저
//

// 예제 1. 렉시컬 스코프 예제
// 기본 예제
const a = 1;
const b = 1;
const c = 1;

function funcA(){
    const b = 2;
    const c = 2;

    console.log("2", a, b, c);
    funcB();
}

function funcB() {
    const c = 3;
    
    console.log("3", a, b, c);
}

console.log("1", a, b, c);
funcA();

// * 출력결과*
// 1, 1, 1, 1
// 2, 1, 2, 2
// 3, 1, 1, 3

// 심화 예제
const a = 1;
const b = 1;
const c = 1;

function funcA() {
    const b = 2;
    const c = 2;

    console.log("2", a, b, c);

    return function funcB() {
        const c = 3;
        
        console.log("3", a, b, c);
    }
    
    funcB();
}

console.log("1", a, b, c);
funcA();

// * 출력결과 *
// 1, 1, 1, 1
// 2, 1, 2, 2
// 3, 1, 2, 3

// 예제 2. 클로저 예제
function outerFunction() {
    let outerVariable = 'I am outside!';

    function innerFunction() {
        console.log(outerVariable); // outerVariable에 접근 가능
    }
    return innerFunction;
}

const myClosure = outerFunction();
myClosure(); // I am outside!

// 예제 3. 상태 유지를 위한 클로저 예제
function createCounter() {
    let count = 0;

    return function() {
        count++;
        return count;
    };
}

const counter = createCounter();

console.log(counter()); // 1
console.log(counter()); // 2
console.log(counter()); // 3

//
// 심화 15. this
//

// 예제 1. this의 동작 방식
// 1. 전역 컨텍스트
console.log(this); // 브라우저에서는 window 객체 출력

// 2. 일반 함수 호출
function showThis() {
    console.log(this);
}

showThis(); // 전역 객체(브라우저에서는 window) 출력

// 3. 메서드 호출
const obj = {
    name: 'Alice',
    showThis: function() {
        console.log(this.name);
    }
};

obj.showThis(); // Alice

// 4. 생성자 함수 호출
function Person(name) {
    this.name = name;
}

const person = new Person('Alice');
console.log(person.name); // Alice

// 5. 클래스 선언
class Person {
    constructor(name, age) {
        // this는 새로 생성된 인스턴스 객체를 가리킴
        this.name = name;
        this.age = age;
    }

    // 클래스 메서드
    greet() {
        console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
    }
}

const alice = new Person('Alice', 30);
alice.greet(); // Hello, my name is Alice and I am 30 years old.

const bob = new Person('Bob', 25);
bob.greet(); // Hello, my name is Bob and I am 25 years old.

// 예제 2. this의 동적 바인딩 
// 일반 함수에서의 this
function showThis() {
    console.log(this);
}

const obj = {
    name: 'Object',
    showThis: showThis
};

showThis(); // 글로벌 객체 (브라우저에서는 window)
obj.showThis(); // obj 객체

// 동적 바인딩을 활용하는 메서드들
function showThis() {
    console.log(this.name);
};

const obj1 = { name: 'Object 1' };
const obj2 = { name: 'Object 2' };

showThis.call(obj1); // Object 1
showThis.apply(obj2); // Object 2

const boundShowThis = showThis.bind(obj1); // 새로운 함수를 반환, 반환된 함수의 this값이 영구적으로 바인딩
boundShowThis(); // Object 1

// 예제 3. this의 정적 바인딩
// 화살표 함수에서의 this : 함수가 정의된 위치의 this값 유지
const obj = {
    name: 'Object',
    showThis: () => {
        console.log(this);
    }
};

obj.showThis(); // 글로벌 객체 window

// 예제 4. 화살표 함수와 일반 함수의 this 차이
function Outer() {
    this.name = 'Outer';

    this.showThisRegular = function() {
        console.log(this.name);
    };

    this.showThistArrow = () => {
        console.log(this.name);
    };
};

const outer = new Outer();
outer.showThisRegular(); // Outher
outer.showThistArrow(); // Outer

const detachedRegular = outer.showThisRegular;
detachedRegular(); // undefined

const detachedArrow = outer.showThistArrow;
detachedArrow(); // Outer

//
// 심화 16. 프로토타입
//

// 예제 1. Object.getPrototypeOf를 사용하여 객체의 프로토타입 접근
// 생성자 함수 정의
function Person(name, age) {
    this.name = name;
    this.age = age;
};

// 프로토타입에 메서드 추가
Person.prototype.greet = function() {
    console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
};

// 새로운 객체 생성
const alice = new Person('Alice', 30);

// 프로토타입 확인
console.log(Object.getPrototypeOf(alice) === Person.prototype); // true

// 프로토타입 체인 확인
console.log(Object.getPrototypeOf(Person.prototype) === Object.prototype); // true
console.log(Object.getPrototypeOf(Object.prototype) === null); // true

// 메서드 호출
alice.greet(); // Hello, my name is Alice and I am 30 years old.

// 예제 2. Object.setPrototypeOf 메서드
const obj = {};
consst proto = { greet: function() { console.log('Hello!');}};

Object.setPrototypeOf(obj, proto);

obj.greet(); // Hello

// 예제 3. 생성자 함수에서의 .prototype
function Person(name, age) {
    this.name = name;
    this.age = age;
};

// Person의 프로토타입에 메서드 추가
Person.prototype.greet = funtion() {
    console.log(`Hello, my name is ${this.name} and I am ${this.age} years odl.`);
};

const alice = new Person('Alice', 30);
const bob = new Person('Bob', 25);

// 두 객체는 Person.prototype에 정의된 메서드를 공유
alice.greet(); // Hello, my name is Alice and I am 30 years old.
bob.greet(); // Hello, my name is Bob and I am 25 years old.

// 예제 4. 클래스에서의 .prototype
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    // 클래스 메서드는 자동으로 Person.prototype에 추가됨
    greet() {
        console.log(`Hello, my name is ${this.name} and I am ${this.age} years old.`);
    }
}

const alice = new Person('Alice', 30);
const bob = new Person('Bob', 25);

// 두 객체는 Person.prototype에 정의된 메서드를 공유
alice.greet(); // Hello, my name is Alice and I am 30 years old.
bob.greet(); // Hello, my name is Bob and I am 25 years old.

// 예제 5. 프로토타입 상속
function Animal(type) {
    this.type = type;
}

// 프로토타입에 메서드 추가
Animal.prototype.makeSound = function() {
    console.log(`${this.type} makes a sound.`);
};

function Dog(name) {
    Animal.call(this, 'Dog'); // Animal 생성자 호출하여 상속
    this.name = name;
}

// 프로토타입 상속 설정
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

Dog.prototype.bark = function() {
    console.log(`${this.name} barks!`);
};

const rex = new Dog('Rex');

rex.makeSound(); // Dog makes a sound. (Animal의 메서드 상속)
rex.bark(); // Rex barks! (Dog의 메서드)

//
// 17. 비동기 프로그래밍
//

// 예제 1. setTimeout 사용
console.log('Start');

setTimeout(() => {
    console.log('This message is shown after 2 seconds');
}, 2000); // 2초 후에 메시지 출력

console.log('End');

// 예제 2. Promise 사용
// 70% 확률로 성공(resolve)하고 30% 확률로 실패(reject)하는 Promise 반환
function randomOpertaion() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            const randomValue = Math.random();

            if(randomValue < 0.3) {
                reject('Operation failed.');
            } else {
                resolve(`Operation succeeded. value is ${randomValue}`);
            }
        }, 1000);
    });
}

function repeatUntilFailure() {
    randomOpertaion()
        // Promise가 이행 상태가 되면 호출
        .then(result => {
            console.log(result);

            repeatUntilFailure();
        })
        // Promise가 실패(reject) 상태가 되면 호출
        .catch(error => {
            console.error(error);
            console.log('Stopping execution');
        });
}

repeatUntilFailure(); 

// * 출력 결과 *
// 작업이 실패할 때까지 계속 반복
// 작업이 실패할 경우 : Operation failed. 와 Stopping execution.
// 성공할 경우 : Operation succeeded.

// 예제 3. Promise.all을 사용한 병렬 진행 : 모든 작업이 성공하면 결과가 반환, 하나라도 작업 실패하면 catch 실행
function taskA() {
    return new Promise(resolve => {
        setTimeout(() => resolve('Task A completed'), 1000);
    });
}

function taskB() {
    return new Promise(resolve => {
        setTimeout(() => resolve('Task B completed'), 2000);
    });
}

function taskC() {
    return new Promise(resolve => {
        setTimeout(() => resolve('Task C completed'), 3000);
    });
}

Promise.all([taskA(), taskB(), taskC()])
    .then(results => {
        console.log('All tasks completed: ', results);
    })
    .catch(error => {
        console.error('One of the tasks failed: ', error);
    });

// * 출력 결과 *
// 모든 작업이 성공하면 3초 후 출력
// All tasks completed:  (3) ['Task A completed', 'Task B completed', 'Task C completed']
// 하나의 작업이라도 실패하면 즉시
// One of the tasks failed

// 예제 4. Promise.allSettled을 사용한 병렬 진행 : Promise의 성공 또는 실패 여부 상관없이 모든 Promise 결과 반환
function taskA() {
    return new Promise(resolve => {
        setTimeout(() => resolve('Task A completed'), 1000);
    });
}

function taskB() {
    return new Promise((resolve, reject) => {
        setTimeout(() => resolve('Task B failed'), 2000);
    });
}

function taskC() {
    return new Promise(resolve => {
        setTimeout(() => resolve('Task C completed'), 3000);
    });
}

Promise.allSettled([taskA(), taskB(), taskC()])
    .then(results => {
        console.log('All tasks completed: ');
        results.forEach(result => console.log(result.status, result.value || result.reason));
    });

// * 출력 결과 *
// All tasks complete:
// Task A completed
// Task B failed
// Task C completed

// 예제 4. Promise.race : Promise 중 가장 먼저 완료된(이행 또는 거부) 결과만 반환
function createRandomPromise() {
    return new Promise((resolve, reject) => {
        const randomValue = Math.random();
        const delay = Math.random() * 1000;

        setTimeout(() => {
            if(randomValue > 0.5) {
                resolve('Promise succeeded');
            } else {
                reject('Promise failed');
            }
        }, delay);
    });
}

function racePromises() {
    const promise1 = createRandomPromise();
    const promise2 = createRandomPromise();

    Promise.race([promise1, promise2])
        .then(result => {
            console.log('First promise fulfilled: ', result);
        })
        .catch(error => {
            console.error('First promise rejected: ', error);
        });
}

racePromises();
racePromises();
racePromises();

// * 출력 결과*
// promise1과 promise2 중 먼저 성공하거나 실패한 결과가 반환

// 예제 5. async 함수, await 키워드
// async 함수
async function fetchData() {
    return "Data fetched!";
}

fetchData().then(data => console.log(data)); // Data fetched!

// await 키워드
function fetchData2() {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve("Data fetched!");
        }, 2000);
    });
}

async function processData() {
    console.log("Fetching data...");
    const data = await fetchData(); // fetchData함수가 반환한 Promise가 이행될 때까지 기다림
    console.log(data);
}

processData(); // Data fetched!

// 예제 6. try...catch를 사용한 에러 처리
async function fetchDataWithError() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            reject("Error fetching data!");
        }, 2000);
    });
}

async function processData() {
    try {
        const data = await fetchDataWithError();
        console.log(data);
    } catch (error) {
        console.error(error);
    }
}

processData(); // Error fetching data!