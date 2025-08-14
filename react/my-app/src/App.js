import './App.css';
import MainHeader from "./components/MainPage/MainHeader";
import CardComponent from "./components/common/cards";
import NameChanger from './components/example/NameChanger';
import LifecycleDemo from './components/example/LifecycleDemo';
import UseStateDemo from './components/example/UseStateDemo';
import UseEffectDemo from './components/example/UseEffectDemo';

function App() {
  const listData = [
    { imgURL : "./image1.jpg", title: "상품명1" },
    { imgURL : "./image2.jpg", title: "상품명2" },
    { imgURL : "./image3.jpg", title: "상품명3" },
    { imgURL : "./image4.jpg", title: "상품명4" },
  ]
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

    <UseEffectDemo/>

  );
}

export default App;
