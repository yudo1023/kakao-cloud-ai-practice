import logo from './logo.svg';
import './App.css';
import React, { useState } from "react";
import CounterClass from './components/example/CounterClass';
import CounterFunction from './components/example/CounterFunction';

function App() {
  // 4. State &  라이프사이클 - 간단한 Class 컴포넌트 작성 및 라이프사이클 메서드 사용
  const [isClassComponent, setIsClassComponent] = useState(true);

  const switchComponent = () => {
    setIsClassComponent((prevState) => !prevState);
  };

  return (
    // 4. State &  라이프사이클 - 간단한 Class 컴포넌트 작성 및 라이프사이클 메서드 사용
    <div>
      <button onClick={switchComponent}>컴포넌트 변경</button>
      {isClassComponent ? <CounterClass /> : <CounterFunction />}
    </div>
  );
}

export default App;








//import ProfileCard from './components/example/ProfileCard';
// 3. 컴포넌트 & Props - 간단한 사용자 프로필 카드
    // <div>
    //     <ProfileCard
    //         avatarUrl="https://example.com/avatar1.png"
    //         name="Jack"
    //         bio="Frontend Developer"
    //     />
    //     <ProfileCard
    //         avatarUrl="https://example.com/avatar2.png"
    //         name="Jane"
    //         bio="Backend Developer"
    //     />
    // </div>
