import './App.css';
import React, { useState } from 'react';
import UseReducerDemo from './components/example/UseReducerDemo';
import { ThemeContext } from './store/ThemeContext';
import { UserContext } from './store/UserContext';
import Main from './components/pages/Main';
import UseFetchOrigin from './components/example/UseFetchOrgin';
import Fetch from './components/common/fetch';


function App() {
  const user = useState({ name: "홍길동", age: 25});
  const [theme, setTheme] = useState("light");


  return (
    // 예제) useReducer 기본
    //<UseReducerDemo/>

    // 예제) useContext 라이트/다크 모드
    // <ThemeContext.Provider value={{ theme, setTheme }}>
    //   <Theme />
    // </ThemeContext.Provider>

    // 예제) useContext 응용 버전(헤더 테마 적용, 나이 추가)
    // <UserContext.Provider value={user}>
    //   <ThemeContext.Provider value={{ theme, setTheme }}>
    //     <Main />
    //   </ThemeContext.Provider>
    // </UserContext.Provider>

    // 커스텀 훅 응용
    // 예제) useFetch로 데이터를 가져오는 커스텀 훅 만들어보기
    // useFetch 사용 전 
    // <UseFetchOrigin />
    // useFetch 사용 후
    <Fetch />
    
  );
}

export default App;
