// ./example/CounterDemo.js

import React from "react";
import { useSelector, useDispatch } from "react-redux";
import { increment, decrement, toggleCounter } from "../reducers/counterSlice";


const CounterDemo = () => {
    const counter = useSelector((state) => state.counter);
    const showCounter = useSelector((state) => state.showCounter);
    const dispatch = useDispatch();

    return (
        <div className="App">
          <h1>리덕스 카운터</h1>
          {showCounter && <p>카운터 값: {counter}</p>}
          <button onClick={() => dispatch(increment(5))}>카운터 +5</button>
          <button onClick={() => dispatch(decrement(3))}>카운터 -3</button>
          <br />
          <br />
          <button onClick={() => dispatch(toggleCounter())}>
            {showCounter ? "카운터 숨기기" : "카운터 보이기"}
          </button>
        </div>
      );
};

export default CounterDemo;