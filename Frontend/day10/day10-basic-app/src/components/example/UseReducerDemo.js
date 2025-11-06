// ./components/example/UseReducerDemo.js

import { useState, useReducer } from "react";

// 예제 1. 기본
// const reducer = (state, action) => {
//     console.log("Reducer 실행");
//     switch(action.type) {
//         case "PLUS":
//             return state + 1;
//         case "MINUSE":
//             return state - 1;
//         default:
//             return state;
//     }
// };
//
// const UseReducerDemo = () => {
//     const [count, dispatch] = useReducer(reducer, 0);
//
//     const plusHandler = () => {
//         dispatch({type: "PLUS"});
//     };
//
//     const minusHandler = () => {
//         dispatch({type: "MINUS"});
//     };
//
//     return (
//         <div style={{ textAlign: "center" }}>
//             <h1>{count}</h1>
//             <button onClick={plusHandler}>더하기</button>
//             <br />
//             <br />
//             <button onClick={minusHandler}>빼기</button>
//         </div>
//     );
// };

// 예제 2-1. 객체 관리
// const initialState = { count: 0 };

// const reducer = (state, action) => {
//     switch (action.type) {
//         case "INCREMENT":
//             return { count: state.count + 1 };
//         case "DECREMENT":
//             return { count: state.count - 1 };
//         default:
//             throw new Error();
//     }
// };

// const UseReducerDemo = () => {
//     const [state, dispatch] = useReducer(reducer, initialState);

//     return (
//         <div>
//             <p>Count: {state.count}</p>
//             <button onClick={() => dispatch({ type: "INCREMENT" })}>Increment</button>
//             <button onClick={() => dispatch({ type: "DECREMENT" })}>Decrement</button>
//         </div>
//     );
// };

// 예제 2-2. 객체에 여러 값 넣어 관리
const initialState = { count1: 0 , count2: 0};

const reducer = (state, action) => {
    switch (action.type) {
        case "INCREMENT":
            return { count1: state.count1 + 1, count2: state.count2 + 2};
        case "DECREMENT":
            return { count1: state.count1 - 1, count2: state.count2 - 2};
        default:
            throw new Error();
    }
};

const UseReducerDemo = () => {
    const [state, dispatch] = useReducer(reducer, initialState);

    return (
        <div>
            <p>Count1: {state.count1}</p>
            <p>Count2: {state.count2}</p>
            <button onClick={() => dispatch({ type: "INCREMENT" })}>Increment</button>
            <button onClick={() => dispatch({ type: "DECREMENT" })}>Decrement</button>
        </div>
    );
};

export default UseReducerDemo;
