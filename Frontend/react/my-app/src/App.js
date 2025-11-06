import './App.css';
import { useState } from 'react';
import MainHeader from "./components/MainPage/MainHeader";
import CardComponent from "./components/common/cards";
import NameChanger from './components/example/NameChanger';
import LifecycleDemo from './components/example/LifecycleDemo';
import UseStateDemo from './components/example/UseStateDemo';
import UseEffectDemo from './components/example/UseEffectDemo';
import UseMemoDemo from './components/example/UseMemoDemo';
import UseCallbackDemo from './components/example/UseCallbackDemo';
import ReactMemoDemo from './components/example/ReactMemoDemo';
import UseRefDemo from './components/example/UseRefDemo';
import { UserContext } from './store/UserContext';
import Header from './components/pages/Header';
//import Input from './components/common/input';

function App() {
  // const listData = [
  //   { imgURL : "./image1.jpg", title: "상품명1" },
  //   { imgURL : "./image2.jpg", title: "상품명2" },
  //   { imgURL : "./image3.jpg", title: "상품명3" },
  //   { imgURL : "./image4.jpg", title: "상품명4" },
  // ]

  // const [count, setCount] = useState(0);

  

  return (
    // <div className='card-list'>
    //   {/* 1번 방법 */}
    //   {/* <CardComponent imgURL={listData[0].imgURL} title={listData[0].title}/> */}

    //   {/* 2번 방법 */}
    //   {/* <CardComponent data={listData[0]}/> */}

    //   {/* 3번 방법 */}
    //   {/* {
    //     listData.map((data) => 
    //       <CardComponent key={data.imgURL} data={data}/>
    //     )
    //   } */}
    // </div>

    <UseMemoDemo/>

    // <div>
    //   <h1>부모 컴포넌트</h1>
    //   <p>Count: {count}</p>

    //   <button onClick={() => setCount((prev) => prev + 1)}>부모버튼</button>
    //   <ReactMemoDemo count={count}/>
    // </div>

    // <UseRefDemo />
    

    // <UserContext.Provider value="홍길동">
    //   <Header />
    // </UserContext.Provider>

    //  <Input />

  );
}

export default App;
