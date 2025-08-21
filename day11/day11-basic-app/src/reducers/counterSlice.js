// ./reducers/counterSlice.js

import { createSlice } from "@reduxjs/toolkit";

const counterSlice = createSlice({
  name: "counter",
  initialState: {
    counter: 0,
    showCounter: true,
  },
  reducers: {
    increment(state, action) {
      // Redux Toolkit을 사용하면 상태 변경에서 불변성을 직접 관리할 필요 없음 (immer 사용)
      state.counter += action.payload;
    },
    decrement(state, action) {
      state.counter -= action.payload;
    },
    toggleCounter(state) {
      state.showCounter = !state.showCounter;
    },
  },
});

// 액션과 리듀서를 자동으로 생성
export const { increment, decrement, toggleCounter } = counterSlice.actions;
export default counterSlice.reducer;
